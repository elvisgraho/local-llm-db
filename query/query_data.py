import argparse
import logging
from typing import Any, Optional, List, Dict, Union, Tuple, Set

import networkx as nx # Used in _retrieve_graph
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from sklearn.metrics.pairwise import cosine_similarity # Used in _retrieve_graph

from query.database_paths import DEFAULT_DB_NAME
from query.data_service import data_service
from query.templates import (
    RAG_ONLY_TEMPLATE,
    KAG_TEMPLATE,
    HYBRID_TEMPLATE, # Generic hybrid template
    DIRECT_TEMPLATE,
    LIGHTRAG_HYBRID_TEMPLATE, # Specific hybrid for lightrag if needed
    KAG_HYBRID_TEMPLATE # Specific hybrid for KAG
)
from query.llm_service import get_llm_response, optimize_query, get_model_context_length, truncate_history
from query.global_vars import (
    RAG_SIMILARITY_THRESHOLD, # Used for lightrag filtering
    RAG_MAX_DOCUMENTS, # Target for final context document count
    GRAPH_MAX_DEPTH, # Used for KAG traversal
    GRAPH_MAX_NODES, # Used for KAG node selection/context limit
    GRAPH_MIN_SIMILARITY # Used for KAG node selection
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants for Token Estimation ---
TOKEN_ESTIMATION_FACTOR = 4 # Simple approximation: 1 token ~= 4 characters
CONTEXT_SAFETY_MARGIN = 200 # Reserve tokens for safety margin, prompt variations
RESERVED_FOR_RETRIEVAL_DEFAULT = 3000 # Default tokens reserved for context (can be overridden)
INITIAL_RETRIEVAL_MULTIPLIER = 2 # Retrieve more docs initially for better reranking
MAX_INITIAL_RETRIEVAL_LIMIT = 50 # Hard limit on initial document retrieval count

# --- Helper Functions ---

def _estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string."""
    if not text: return 0
    return len(text) // TOKEN_ESTIMATION_FACTOR

def _reorder_documents_for_context(docs: List[Document]) -> List[Document]:
    """Reorders documents to place most relevant at start and end for LLM context."""
    if len(docs) >= 3:
        logger.debug(f"Reordering {len(docs)} documents for 'lost in the middle'.")
        most_relevant, second_most_relevant, *middle_docs = docs
        return [most_relevant] + middle_docs + [second_most_relevant]
    return docs

def _apply_metadata_filter(docs_with_scores: List[Tuple[Document, float]], metadata_filter: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
    """Applies metadata filter to a list of (Document, score) tuples."""
    if not metadata_filter: return docs_with_scores
    filtered = []
    for doc, score in docs_with_scores:
        match = all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
        if match: filtered.append((doc, score))
    logger.info(f"Applied metadata filter: {len(docs_with_scores)} -> {len(filtered)} documents.")
    return filtered

def _calculate_available_context(
    query_text: str,
    conversation_history: Optional[List[Dict[str, str]]],
    llm_config: Optional[Dict],
    base_template_str: str,
    reserved_for_context: int = RESERVED_FOR_RETRIEVAL_DEFAULT
) -> Tuple[Optional[List[Dict[str, str]]], int]:
    """Calculates available tokens for retrieval context after accounting for other elements."""
    context_length = get_model_context_length(llm_config)
    query_tokens = _estimate_tokens(query_text)

    # Estimate tokens used by the prompt template structure itself
    template_for_estimation = base_template_str.replace("{context}", "").replace("{question}", "").replace("{sources}", "").replace("{relationships}", "")
    prompt_tokens = _estimate_tokens(template_for_estimation)

    # Calculate max tokens for history based on remaining space
    max_history_tokens = context_length - query_tokens - prompt_tokens - reserved_for_context - CONTEXT_SAFETY_MARGIN
    max_history_tokens = max(0, max_history_tokens) # Ensure non-negative

    truncated_history, history_tokens = truncate_history(conversation_history, max_history_tokens)

    # Calculate final available tokens for the retrieved context itself
    available_tokens_for_context = context_length - query_tokens - history_tokens - prompt_tokens - CONTEXT_SAFETY_MARGIN
    available_tokens_for_context = max(0, available_tokens_for_context) # Ensure non-negative

    logger.info(f"Context Calc: Total={context_length}, Query={query_tokens}, History={history_tokens} (Max={max_history_tokens}), Prompt={prompt_tokens}, Reserved={reserved_for_context} -> Available for Context={available_tokens_for_context}")
    return truncated_history, available_tokens_for_context

def _retrieve_semantic(query_text: str, db: VectorStore, k: int, metadata_filter: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
    """Performs semantic search."""
    try:
        logger.info(f"Performing semantic search (k={k}) with filter: {metadata_filter}")
        results = db.similarity_search_with_score(
            query_text, k=k, filter=metadata_filter # Chroma uses 'filter' which maps to 'where'
        )
        return results
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        return []

def _retrieve_keyword(query_text: str, db: VectorStore, rag_type: str, db_name: str, k: int, metadata_filter: Optional[Dict[str, Any]]) -> List[Tuple[Document, float]]:
    """Performs keyword search using BM25."""
    bm25 = data_service.get_bm25_index(rag_type, db_name)
    _, bm25_corpus, bm25_doc_ids = data_service._bm25_cache.get((rag_type, db_name), (None, None, None))

    if not bm25 or not bm25_corpus or not bm25_doc_ids:
        logger.warning(f"BM25 index/data not available for {rag_type}/{db_name}. Skipping keyword search.")
        return []

    try:
        logger.info(f"Performing keyword search (BM25, k={k}).")
        tokenized_query = query_text.split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        bm25_scored_docs_info = sorted(zip(bm25_doc_ids, bm25_corpus, bm25_scores), key=lambda x: x[2], reverse=True)
        top_bm25_candidates = [(doc_id, score) for doc_id, _, score in bm25_scored_docs_info if score > 0][:k]
        top_bm25_ids = [doc_id for doc_id, _ in top_bm25_candidates]

        if not top_bm25_ids: return []

        chroma_bm25_docs_data = db.get(ids=top_bm25_ids, include=["metadatas", "documents"])
        bm25_docs_with_scores = []
        id_to_score = dict(top_bm25_candidates)
        for doc_id, content, metadata in zip(chroma_bm25_docs_data["ids"], chroma_bm25_docs_data["documents"], chroma_bm25_docs_data["metadatas"]):
             doc = Document(page_content=content, metadata=metadata)
             score = id_to_score.get(doc_id, 0.0)
             bm25_docs_with_scores.append((doc, score))

        # Apply metadata filter *after* retrieving BM25 candidates
        return _apply_metadata_filter(bm25_docs_with_scores, metadata_filter)
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}", exc_info=True)
        return []

def _retrieve_graph(query_text: str, rag_type: str, db_name: str, k: int, metadata_filter: Optional[Dict[str, Any]]) -> Tuple[List[Tuple[Document, float]], List[str]]:
    """
    Performs graph traversal for KAG.
    Returns a tuple containing:
        - List of (Document, score) tuples.
        - List of formatted relationship strings.
    """
    try:
        graph = data_service.get_kag_graph(rag_type, db_name) # Corrected method name
        db = data_service.get_chroma_db(rag_type, db_name) # Need Chroma for node content/metadata
        if not graph or not db:
            logger.warning(f"KAG graph or DB not available for {rag_type}/{db_name}.")
            return []

        logger.info(f"Performing graph retrieval (KAG, k={k}).")
        embedding_function = data_service.embedding_function # Ensure loaded

        # Find initial nodes by comparing query embedding to node embeddings in the graph
        query_embedding = embedding_function.embed_query(query_text)
        candidate_nodes_scores = []
        # Filter for nodes of type 'chunk' that also have embeddings
        chunk_nodes_with_embeddings = [
            (node_id, data) for node_id, data in graph.nodes(data=True)
            if data.get('type') == 'chunk' and 'embedding' in data
        ]

        logger.debug(f"Found {len(chunk_nodes_with_embeddings)} chunk nodes with embeddings in KAG graph.")

        for node_id, node_data in chunk_nodes_with_embeddings: # Iterate filtered nodes
            try:
                similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
                # Use original '>' operator and check threshold
                if similarity > GRAPH_MIN_SIMILARITY:
                    # Check metadata filter *before* adding to candidates
                    metadata = node_data.get('metadata', {})
                    match = True
                    if metadata_filter:
                        match = all(metadata.get(key) == value for key, value in metadata_filter.items())

                    if match:
                        candidate_nodes_scores.append((node_id, similarity, metadata)) # Store ID, score, and metadata for sorting/fetching

            except Exception as e:
                 logger.warning(f"Error calculating similarity for node {node_id}: {e}")
                 continue # Skip node on error

        if not candidate_nodes_scores:
            logger.warning("No KAG nodes met the similarity threshold or metadata filter.")
            return []

        # Sort candidates by similarity and select top K
        candidate_nodes_scores.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidate_nodes_scores[:k]
        start_node_ids = [node_id for node_id, _, _ in top_candidates]
        initial_node_scores = {node_id: score for node_id, score, _ in top_candidates} # Store scores for later use

        logger.info(f"Selected {len(start_node_ids)} starting nodes for KAG traversal.")

        # Traverse graph from initial nodes, collecting Document objects and relationships
        visited_nodes = set()
        collected_docs_scores = {} # Use dict to store {node_id: (Document, score)}
        relationships = [] # List to store formatted relationship strings
        max_nodes_to_visit = GRAPH_MAX_NODES * INITIAL_RETRIEVAL_MULTIPLIER # Visit more for better selection

        def traverse(node_id, depth=0):
            # Corrected indentation and logic for traverse function
            if node_id in visited_nodes or depth >= GRAPH_MAX_DEPTH or len(visited_nodes) >= max_nodes_to_visit:
                return
            visited_nodes.add(node_id)

            if node_id in graph: # Check if node exists in graph
                node_data = graph.nodes[node_id]
                content = node_data.get('content')
                metadata = node_data.get('metadata', {})

                if content: # Only collect nodes with content
                    # Use initial score if available, otherwise default to 0
                    score = initial_node_scores.get(node_id, 0.0)
                    doc = Document(page_content=content, metadata=metadata)
                    # Store doc and score, potentially overwriting if visited via shorter path with lower score
                    if node_id not in collected_docs_scores or score > collected_docs_scores[node_id][1]:
                         collected_docs_scores[node_id] = (doc, score)
                         # Log successful addition HERE
                         logger.debug(f"Collected doc for node {node_id} with score {score:.4f}.")
                # else: # Don't log skipping collection if node simply has no content, might be a relationship hub
                #    logger.debug(f"Node {node_id} has no content, skipping doc collection.")

                # Process edges/relationships REGARDLESS of current node content
                for neighbor_id in list(graph.neighbors(node_id)): # Use list for safe iteration
                    if neighbor_id not in graph:
                        logger.warning(f"Neighbor '{neighbor_id}' of node '{node_id}' not found in graph. Skipping edge processing.")
                        continue

                    edge_data = graph.get_edge_data(node_id, neighbor_id)
                    if not edge_data:
                        logger.warning(f"No edge data found between {node_id} and {neighbor_id}")
                        continue

                    # Format relationship if type exists
                    if edge_data.get('relation'):
                        neighbor_data = graph.nodes[neighbor_id]
                        relation_type = edge_data['relation']
                        similarity_score = edge_data.get('similarity', 0)

                        node_content_preview = node_data.get('content', '[Node has no content]')[:100] + ('...' if len(node_data.get('content', '')) > 100 else '')
                        neighbor_content_preview = neighbor_data.get('content', '[Node has no content]')[:100] + ('...' if len(neighbor_data.get('content', '')) > 100 else '')

                        rel_str = f"- Relation '{relation_type}'"
                        if relation_type == 'semantically_similar':
                             rel_str += f" (similarity: {similarity_score:.2f})"
                        elif relation_type == 'same_section':
                             section_type = node_data.get('metadata', {}).get('section_type', 'unknown')
                             rel_str += f" (section: {section_type})"
                        # Add more relation types here if needed

                        rel_str += f":\n  From Node {node_id}: {node_content_preview}\n  To Node {neighbor_id}: {neighbor_content_preview}"
                        relationships.append(rel_str)

                    # Recursively traverse neighbors
                    traverse(neighbor_id, depth + 1) # Correct indentation for traversal call

            else:
                 logger.warning(f"Node {node_id} encountered during traversal but not found in graph keys.")

        # Limit starting points for traversal
        num_start_nodes = min(len(start_node_ids), GRAPH_MAX_NODES)
        for start_node_id in start_node_ids[:num_start_nodes]:
            traverse(start_node_id)

        logger.info(f"Graph traversal finished. Collected {len(collected_docs_scores)} candidate documents and {len(relationships)} relationships.")
        if not collected_docs_scores:
             logger.warning("Traversal completed but collected_docs_scores is empty.")
             # Still return empty relationships if no docs found
             return [], []

        # Convert collected data to list of tuples
        final_graph_docs_scores = list(collected_docs_scores.values())

        # Sort by score (descending) before returning
        final_graph_docs_scores.sort(key=lambda x: x[1], reverse=True)
        # Return top K results based on the original request `k`, and the collected relationships
        return final_graph_docs_scores[:k], relationships

    except Exception as e:
        logger.error(f"Error during graph retrieval: {e}", exc_info=True)
        return []

def _rerank_results(query_text: str, results: List[Tuple[Document, float]], k: int) -> List[Tuple[Document, float]]:
    """Reranks results using a CrossEncoder model if available."""
    reranker = data_service.reranker
    if not reranker or not results:
        # logger.info("Reranker not available or no results to rerank. Returning top K original.")
        # Sort by original score just in case they weren't already
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    try:
        logger.info(f"Reranking {len(results)} results.")
        pairs = [[query_text, doc.page_content] for doc, _ in results]
        rerank_scores = reranker.predict(pairs)
        reranked_results = [(results[i][0], rerank_scores[i]) for i in range(len(results))]
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Selected top {k} results after reranking.")
        return reranked_results[:k]
    except Exception as e:
        logger.error(f"Error during reranking: {e}. Falling back to top K original scores.", exc_info=True)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

def _select_docs_for_context(docs_with_scores: List[Tuple[Document, float]], available_tokens: int) -> Tuple[List[Document], List[str], int]:
    """Selects documents from a sorted list to fit within the token limit."""
    selected_docs = []
    selected_sources = set()
    current_tokens = 0
    separator_tokens = _estimate_tokens("\n\n---\n\n")

    logger.info(f"Selecting documents to fit within {available_tokens} estimated tokens.")
    for doc, score in docs_with_scores:
        doc_tokens = _estimate_tokens(doc.page_content)
        if current_tokens + doc_tokens + separator_tokens <= available_tokens:
            selected_docs.append(doc)
            current_tokens += doc_tokens + separator_tokens
            source = doc.metadata.get("source")
            if source and isinstance(source, str) and source.strip():
                selected_sources.add(source)
            # else: logger.debug(f"Missing source in selected doc: {doc.metadata}")
        else:
            # logger.debug(f"Token limit reached adding doc. Added {len(selected_docs)} docs.")
            break # Stop adding docs once limit is exceeded

    if not selected_docs:
        logger.warning("No documents could be selected within the available token limit.")
        return [], [], 0

    estimated_tokens_used = current_tokens
    unique_sources = sorted(list(selected_sources))
    logger.info(f"Selected {len(selected_docs)} documents using ~{estimated_tokens_used} tokens. Sources: {len(unique_sources)}")
    return selected_docs, unique_sources, estimated_tokens_used

def _generate_response(
    query_text: str,
    final_docs: List[Document],
    sources: List[str],
    rag_type: str,
    hybrid: bool,
    llm_config: Optional[Dict],
    truncated_history: Optional[List[Dict[str, str]]],
    estimated_context_tokens: int,
    formatted_relationships: Optional[List[str]] = None # Add relationships parameter
) -> Dict[str, Union[str, List[str], int]]:
    """Formats context, selects template, generates prompt, and calls LLM."""

    if not final_docs:
        logger.warning(f"No documents provided to _generate_response for {rag_type} mode.")
        no_info_msg = "No relevant information found in the knowledge graph." if rag_type == 'kag' else "No relevant information found in the database."
        return {"text": no_info_msg, "sources": [], "estimated_context_tokens": 0}

    reordered_docs = _reorder_documents_for_context(final_docs)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reordered_docs])

    # Select template
    if hybrid:
        template_str = KAG_HYBRID_TEMPLATE if rag_type == 'kag' else \
                       LIGHTRAG_HYBRID_TEMPLATE if rag_type == 'lightrag' else \
                       HYBRID_TEMPLATE # Default RAG hybrid
    else:
        template_str = KAG_TEMPLATE if rag_type == 'kag' else RAG_ONLY_TEMPLATE # RAG/LightRAG non-hybrid

    prompt_template = ChatPromptTemplate.from_template(template_str)

    # Format sources
    sources_text = "\n".join(f"- {s}" for s in sources if s and s != 'unknown')
    if not sources_text: sources_text = "No specific sources identified for this context."

    # Prepare prompt arguments
    prompt_args = {"question": query_text, "context": context_text}
    if "{sources}" in template_str: prompt_args["sources"] = sources_text
    if "{relationships}" in template_str:
        if rag_type == 'kag' and formatted_relationships:
            relationships_text = "\n\n".join(formatted_relationships)
            if not relationships_text: # Handle case where list is empty
                 relationships_text = "No specific relationships were identified for this context."
        else:
            relationships_text = "Relationship information is not applicable or was not provided." # Default/fallback
        prompt_args["relationships"] = relationships_text

    prompt = prompt_template.format(**prompt_args)
    logger.debug(f"Using template: {'Hybrid' if hybrid else 'Standard'} for {rag_type}")

    # Get response from LLM, passing truncated history
    response_text = get_llm_response(prompt, llm_config=llm_config, conversation_history=truncated_history)

    return {"text": response_text, "sources": sources, "estimated_context_tokens": estimated_context_tokens}

# --- Query Functions ---

def query_direct(query_text: str, llm_config: Optional[Dict] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Union[str, List[str]]]:
    """Query the model directly without using RAG. History truncation happens in get_llm_response."""
    try:
        logger.debug("Processing direct query")
        prompt_template = ChatPromptTemplate.from_template(DIRECT_TEMPLATE)
        prompt = prompt_template.format(question=query_text)
        # get_llm_response handles history truncation internally based on available space *after* the prompt
        response_text = get_llm_response(prompt, llm_config=llm_config, conversation_history=conversation_history)
        return {"text": response_text, "sources": []}
    except Exception as e:
        logger.error(f"Error in direct query: {str(e)}", exc_info=True)
        raise

def query_rag(
    query_text: str,
    hybrid: bool = False,
    rag_type: str = 'rag', # Keep for consistency, logic uses 'rag'
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """Query using standard RAG (Semantic + Keyword + Reranking + Context Limit)."""
    try:
        logger.info(f"Processing RAG query for {db_name} (Hybrid: {hybrid})")
        db = data_service.get_chroma_db('rag', db_name)
        if not db: raise ValueError(f"Chroma DB 'rag/{db_name}' not loaded.")
        _ = data_service.embedding_function # Ensure loaded

        # 1. Calculate available context space
        template_str = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
        truncated_history, available_tokens = _calculate_available_context(
            query_text, conversation_history, llm_config, template_str
        )

        # 2. Retrieve initial candidates (Semantic + Keyword)
        # Apply hard limit to initial retrieval count
        k_initial = min(RAG_MAX_DOCUMENTS * INITIAL_RETRIEVAL_MULTIPLIER, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"RAG initial retrieval k = {k_initial}")
        semantic_results = _retrieve_semantic(query_text, db, k_initial, metadata_filter)
        keyword_results = _retrieve_keyword(query_text, db, 'rag', db_name, k_initial, metadata_filter)

        # Combine results (deduplicate based on content)
        combined_dict = {doc.page_content: (doc, score) for doc, score in semantic_results}
        for doc, score in keyword_results:
            if doc.page_content not in combined_dict or score > combined_dict[doc.page_content][1]:
                combined_dict[doc.page_content] = (doc, score)
        initial_docs_scores = list(combined_dict.values())
        initial_docs_scores.sort(key=lambda x: x[1], reverse=True) # Sort before reranking

        # 3. Rerank the initially retrieved pool, aiming for the final target count
        reranked_docs_scores = _rerank_results(query_text, initial_docs_scores, RAG_MAX_DOCUMENTS)

        # 4. Select final documents based on token limit
        final_docs, sources, estimated_tokens = _select_docs_for_context(reranked_docs_scores, available_tokens)

        # 5. Generate response
        return _generate_response(
            query_text, final_docs, sources, 'rag', hybrid, llm_config, truncated_history, estimated_tokens
        )
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise

def query_lightrag(
    query_text: str,
    hybrid: bool = False,
    rag_type: str = 'lightrag', # Keep for consistency
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """Query using LightRAG (Semantic Search + Threshold + Reranking + Context Limit)."""
    try: # Ensure try is at correct indentation
        logger.info(f"Processing LightRAG query for {db_name} (Hybrid: {hybrid})")
        db = data_service.get_chroma_db('lightrag', db_name)
        if not db: raise ValueError(f"Chroma DB 'lightrag/{db_name}' not loaded.")
        _ = data_service.embedding_function # Ensure loaded

        # 1. Calculate available context space
        template_str = LIGHTRAG_HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
        truncated_history, available_tokens = _calculate_available_context(
            query_text, conversation_history, llm_config, template_str
        )

        # 2. Retrieve initial candidates (Semantic only for LightRAG)
        # Apply hard limit to initial retrieval count
        k_initial = min(RAG_MAX_DOCUMENTS * INITIAL_RETRIEVAL_MULTIPLIER, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"LightRAG initial retrieval k = {k_initial}")
        semantic_results = _retrieve_semantic(query_text, db, k_initial, metadata_filter)

        # 3. Filter by similarity threshold
        threshold_filtered_docs = [(doc, score) for doc, score in semantic_results if score >= RAG_SIMILARITY_THRESHOLD]
        threshold_filtered_docs.sort(key=lambda x: x[1], reverse=True) # Sort before reranking

        if not threshold_filtered_docs:
             logger.warning("No documents met the similarity threshold for LightRAG.")
             return {"text": "No relevant information found matching the similarity criteria.", "sources": [], "estimated_context_tokens": 0}

        # 4. Rerank (Applied after thresholding for LightRAG), aiming for the final target count
        reranked_docs_scores = _rerank_results(query_text, threshold_filtered_docs, RAG_MAX_DOCUMENTS)

        # 5. Select final documents based on token limit
        final_docs, sources, estimated_tokens = _select_docs_for_context(reranked_docs_scores, available_tokens)

        # 6. Generate response
        return _generate_response(
            query_text, final_docs, sources, 'lightrag', hybrid, llm_config, truncated_history, estimated_tokens
        )
    except Exception as e: # Ensure except block is correctly aligned with try
        logger.error(f"Error in LightRAG query: {str(e)}", exc_info=True)
        raise

def query_kag( # Ensure def is at correct indentation
    query_text: str,
    hybrid: bool = False,
    rag_type: str = 'kag', # Keep for consistency
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """Query using KAG (Graph Traversal + Reranking + Context Limit)."""
    try:
        logger.info(f"Processing KAG query for {db_name} (Hybrid: {hybrid})")
        # KAG requires both graph and DB
        graph = data_service.get_kag_graph('kag', db_name) # Corrected method name
        db = data_service.get_chroma_db('kag', db_name)
        if not graph or not db: raise ValueError(f"Graph or Chroma DB 'kag/{db_name}' not loaded.")
        _ = data_service.embedding_function # Ensure loaded

        # 1. Calculate available context space
        template_str = KAG_HYBRID_TEMPLATE if hybrid else KAG_TEMPLATE
        truncated_history, available_tokens = _calculate_available_context(
            query_text, conversation_history, llm_config, template_str
        )

        # 2. Retrieve initial candidates and relationships (Graph Traversal)
        # Apply hard limit to initial retrieval count
        k_initial = min(GRAPH_MAX_NODES * INITIAL_RETRIEVAL_MULTIPLIER, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"KAG initial retrieval k = {k_initial}")
        # Unpack documents and relationships
        initial_docs_scores, formatted_relationships = _retrieve_graph(query_text, 'kag', db_name, k_initial, metadata_filter)

        if not initial_docs_scores:
             logger.warning("Graph retrieval yielded no documents for KAG.")
             # Pass empty lists and relationships to generate response
             return _generate_response(query_text, [], [], 'kag', hybrid, llm_config, truncated_history, 0, formatted_relationships=[])


        # 3. Rerank graph results, aiming for the final target count (GRAPH_MAX_NODES acts like RAG_MAX_DOCUMENTS here)
        reranked_docs_scores = _rerank_results(query_text, initial_docs_scores, GRAPH_MAX_NODES)

        # 4. Select final documents based on token limit
        final_docs, sources, estimated_tokens = _select_docs_for_context(reranked_docs_scores, available_tokens)

        # 5. Generate response, passing the formatted relationships
        return _generate_response(
            query_text, final_docs, sources, 'kag', hybrid, llm_config, truncated_history, estimated_tokens, formatted_relationships=formatted_relationships
        )
    except Exception as e:
        logger.error(f"Error in KAG query: {str(e)}", exc_info=True)
        raise

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--rag_type", type=str, choices=['rag', 'direct', 'lightrag', 'kag'], default='rag', help="Query mode")
    parser.add_argument("--db_name", type=str, default=DEFAULT_DB_NAME, help="Database name.")
    parser.add_argument("--optimize", action="store_true", help="Optimize query")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid prompt template (if applicable)")
    args = parser.parse_args()

    try:
        # Assume default llm_config for CLI for now
        default_llm_config = {'provider': 'local', 'modelName': ''} # Adjust as needed
        query_text = optimize_query(args.query_text, llm_config=default_llm_config) if args.optimize else args.query_text
        logger.info(f"Processing query in {args.rag_type} mode for DB '{args.db_name}' (Hybrid: {args.hybrid}): {query_text}")

        query_func_map = {
            'direct': query_direct,
            'rag': query_rag,
            'lightrag': query_lightrag,
            'kag': query_kag
        }

        if args.rag_type in query_func_map:
            query_func = query_func_map[args.rag_type]
            # Prepare arguments for the specific function
            func_args = {
                "query_text": query_text,
                "llm_config": default_llm_config,
                "conversation_history": None # CLI doesn't handle history
            }
            if args.rag_type != 'direct':
                func_args["hybrid"] = args.hybrid
                func_args["rag_type"] = args.rag_type # Pass rag_type for consistency if needed internally
                func_args["db_name"] = args.db_name
                # func_args["metadata_filter"] = None # Add if CLI supports filters

            result = query_func(**func_args)
        else:
             raise ValueError(f"Invalid rag_type: {args.rag_type}")


        print(result["text"])
        if result.get("sources"): print("\nSources:", result["sources"])
        if "estimated_context_tokens" in result: print(f"\nEstimated Context Tokens Used: {result['estimated_context_tokens']}")

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
