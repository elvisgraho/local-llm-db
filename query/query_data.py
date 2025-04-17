import argparse
import networkx as nx
from typing import Any, Optional, List, Dict, Union # Added for type hinting
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document # Added import
from query.database_paths import CHROMA_PATH
from query.data_service import data_service
from query.templates import (
    RAG_ONLY_TEMPLATE,
    KAG_TEMPLATE,
    HYBRID_TEMPLATE,
    DIRECT_TEMPLATE,
    LIGHTRAG_HYBRID_TEMPLATE,
    KAG_HYBRID_TEMPLATE
)
from query.llm_service import get_llm_response, optimize_query
from query.global_vars import (
    RAG_SIMILARITY_THRESHOLD,
    RAG_MAX_DOCUMENTS,
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_NODES,
    GRAPH_MIN_SIMILARITY
)
import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from langchain_core.vectorstores import VectorStore # Added for type hinting

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--mode", type=str, choices=['rag', 'direct', 'hybrid', 'graph', 'lightrag', 'kag'], 
                       default='rag', help="Query mode: rag, direct, hybrid, graph, lightrag, or kag")
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the query before processing")
    args = parser.parse_args()
    
    try:
        # Optimize query if requested
        query_text = optimize_query(args.query_text) if args.optimize else args.query_text
        logger.info(f"Processing query in {args.mode} mode: {query_text}")
        
        if args.mode == 'direct':
            result = query_direct(query_text)
        elif args.mode == 'hybrid':
            result = query_hybrid(query_text)
        elif args.mode == 'graph':
            result = query_graph(query_text)
        elif args.mode == 'lightrag':
            result = query_lightrag(query_text)
        elif args.mode == 'kag':
            result = query_kag(query_text)
        else:
            result = query_rag(query_text)
        
        print(result["text"])
        if result.get("sources"):
            print("\nSources:", result["sources"])
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

def query_direct(query_text: str, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query the model directly without using RAG.
    
    Args:
        query_text (str): The query text.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        query_text (str): The query text.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing direct query")
        prompt_template = ChatPromptTemplate.from_template(DIRECT_TEMPLATE)
        prompt = prompt_template.format(question=query_text)
        
        response_text = get_llm_response(prompt, llm_config=llm_config)
        return {"text": response_text, "sources": []}
    except Exception as e:
        logger.error(f"Error in direct query: {str(e)}", exc_info=True)
        raise  # Re-raise the exception to be caught by the main handler

def _perform_hybrid_retrieval_and_rerank(
    query_text: str,
    db: VectorStore,
    bm25: Optional[Any], # Using Any for BM25Okapi as it's not a LangChain type
    reranker: Optional[Any] # Using Any for CrossEncoder
) -> Tuple[List[Document], List[str]]:
    """
    Performs hybrid retrieval (semantic + keyword) and reranks the results.

    Args:
        query_text (str): The user's query.
        db (VectorStore): The vector store (e.g., Chroma) for semantic search.
        bm25 (Optional[BM25Okapi]): The initialized BM25 index.
        reranker (Optional[CrossEncoder]): The initialized reranker model.

    Returns:
        Tuple[List[Document], List[str]]: A tuple containing the list of final
                                          reranked/sorted documents and a list of their sources.
    """
    final_docs = []
    sources = []

    # --- Semantic Search ---
    try:
        results = db.similarity_search_with_score(query_text, k=RAG_MAX_DOCUMENTS)
    except Exception as e:
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        results = []

    # --- Keyword Search (BM25) ---
    bm25_results = []
    if bm25 and data_service._bm25_corpus and data_service._bm25_doc_ids:
        try:
            tokenized_query = query_text.split(" ")
            bm25_scores = bm25.get_scores(tokenized_query)
            bm25_scored_docs = sorted(
                zip(data_service._bm25_doc_ids, data_service._bm25_corpus, bm25_scores),
                key=lambda x: x[2], reverse=True
            )
            top_bm25_ids = [doc_id for doc_id, _, score in bm25_scored_docs[:RAG_MAX_DOCUMENTS] if score > 0]
            if top_bm25_ids:
                # Fetch metadata only for the top BM25 results to optimize
                chroma_bm25_docs = db.get(ids=top_bm25_ids, include=["metadatas", "documents"])
                id_to_metadata = {id: meta for id, meta in zip(chroma_bm25_docs["ids"], chroma_bm25_docs["metadatas"])}
                id_to_content = {id: content for id, content in zip(chroma_bm25_docs["ids"], chroma_bm25_docs["documents"])}
                for doc_id, _content, score in bm25_scored_docs:
                    if doc_id in top_bm25_ids: # Ensure we only process top IDs with metadata
                        bm25_results.append(
                            (Document(page_content=id_to_content[doc_id], metadata=id_to_metadata[doc_id]), score)
                        )
        except Exception as e:
            logger.error(f"Error during BM25 search: {e}", exc_info=True)
    else:
        logger.warning("BM25 index not available, skipping keyword search.")

    # --- Combine Semantic and Keyword Results ---
    combined_results_dict = {}
    for doc, score in results:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        combined_results_dict[doc_id] = (doc, score, "semantic")
    for doc, score in bm25_results:
        doc_id = doc.metadata.get("id", doc.page_content[:50])
        if doc_id not in combined_results_dict:
            combined_results_dict[doc_id] = (doc, score, "keyword")
    combined_results = list(combined_results_dict.values())

    if not combined_results:
        logger.warning("No relevant information found from semantic or keyword search.")
        return [], [] # Return empty lists

    # --- Reranking ---
    final_results_tuples = []
    if reranker:
        try:
            pairs = [[query_text, doc.page_content] for doc, _, _ in combined_results]
            rerank_scores = reranker.predict(pairs)
            reranked_results = []
            for i, (doc, original_score, search_type) in enumerate(combined_results):
                reranked_results.append((doc, rerank_scores[i], search_type))
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            final_results_tuples = reranked_results[:RAG_MAX_DOCUMENTS]
            logger.info(f"Reranked {len(combined_results)} results, selected top {len(final_results_tuples)}")
        except Exception as e:
            logger.error(f"Error during reranking: {e}. Falling back.", exc_info=True)
            # Fallback: sort by original score (semantic preferred)
            combined_results.sort(key=lambda x: (x[2] == 'semantic', x[1]), reverse=True)
            final_results_tuples = combined_results[:RAG_MAX_DOCUMENTS]
    else:
        logger.warning("Reranker not available, using combined semantic/keyword results without reranking.")
        combined_results.sort(key=lambda x: (x[2] == 'semantic', x[1]), reverse=True)
        final_results_tuples = combined_results[:RAG_MAX_DOCUMENTS]

    # Extract final documents and sources
    final_docs = [doc for doc, _, _ in final_results_tuples]
    sources = []
    for doc in final_docs:
        source = doc.metadata.get("source")
        if source and isinstance(source, str) and source.strip():
            sources.append(source)
        else:
            logger.warning(f"Invalid or missing source in final document metadata: {doc.metadata}")
            
    return final_docs, sources

def query_hybrid(query_text: str, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query using both RAG context and the model's knowledge.
    Applies Hybrid Search (Semantic + Keyword) and Reranking to the retrieved context.
    
    Args:
        query_text (str): The query text.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        _ = data_service.embedding_function # Ensure embedding function is loaded
        db = data_service.chroma_db
        bm25 = data_service.bm25_index
        reranker = data_service.reranker
        
        # Perform retrieval and reranking using the helper function
        final_docs, sources = _perform_hybrid_retrieval_and_rerank(
            query_text, db, bm25, reranker
        )
        
        # Format context
        if not final_docs:
            context_text = "No relevant context found."
            # Sources list is already empty from the helper function
        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
            
        # Create prompt using the HYBRID_TEMPLATE
        prompt_template = ChatPromptTemplate.from_template(HYBRID_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get response
        response_text = get_llm_response(prompt, llm_config=llm_config)
        # Sources are obtained from the helper function
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in hybrid query: {str(e)}", exc_info=True)
        raise # Re-raise the exception

def query_graph(query_text: str, hybrid: bool = False, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query using the graph structure with semantic search.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing graph query")
        # Initialize only GraphRAG graph and embedding function
        _ = data_service.embedding_function
        G = data_service.graphrag_graph
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}", exc_info=True)
        raise # Re-raise the exception

    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    # Get query embedding
    query_embedding = data_service.embedding_function.embed_query(query_text)
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > GRAPH_MIN_SIMILARITY:
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    relevant_nodes = [node for node, _ in relevant_nodes[:GRAPH_MAX_NODES]]

    if not relevant_nodes:
        logger.warning("No relevant information found in the graph structure")
        return {"text": "No relevant information found in the graph structure", "sources": []}

    # Collect context from relevant nodes and their neighbors
    context_parts = []
    sources = set()
    
    for node in relevant_nodes:
        node_data = G.nodes[node]
        context_parts.append(node_data.get('content', ''))
        sources.add(node_data.get('metadata', {}).get('source', 'unknown'))
        
        # Add content from semantically similar nodes
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            if edge_data.get('relation') == 'semantically_similar':
                neighbor_data = G.nodes[neighbor]
                context_parts.append(neighbor_data.get('content', ''))
                sources.add(neighbor_data.get('metadata', {}).get('source', 'unknown'))
        
        # Add content from code blocks and payloads
        for neighbor in G.neighbors(node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('type') in ['code_block', 'payload']:
                context_parts.append(neighbor_data.get('content', ''))

    context_text = "\n\n---\n\n".join(context_parts)
    
    # Use hybrid template if hybrid mode is enabled
    template = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = get_llm_response(prompt, llm_config=llm_config)
    return {"text": response_text, "sources": list(sources)}

def query_rag(query_text: str, hybrid: bool = False, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query using RAG with Hybrid Search (Semantic + Keyword) and Reranking.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use the hybrid LLM prompt template
                     (combining retrieved context with LLM's internal knowledge).
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing RAG query")
        # Initialize necessary services
        _ = data_service.embedding_function # Ensure embedding function is loaded
        db = data_service.chroma_db
        # BM25 and reranker will be accessed by the helper function
        
        # Verify database exists and has data
        # Note: BM25 index loading depends on ChromaDB, so this check is important
        if not os.path.exists(os.path.join(str(CHROMA_PATH), "chroma.sqlite3")):
            logger.error("RAG database not found")
            return {"text": "RAG database not found. Please run populate_rag.py first.", "sources": []}
            
        # Perform retrieval and reranking using the helper function
        final_docs, sources = _perform_hybrid_retrieval_and_rerank(
            query_text, db, data_service.bm25_index, data_service.reranker
        )
        
        if not final_docs:
            # If no documents are found after retrieval/reranking, return a specific message
            logger.warning("No sufficiently relevant documents found after combining/reranking in RAG mode.")
            return {"text": "No relevant information found in the database to answer the query.", "sources": []}

        # Format context from the final documents
        context_text = "\n\n---\n\n".join([doc.page_content for doc in final_docs])
        
        # Use hybrid template if hybrid mode is enabled
        template = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Get response
        response_text = get_llm_response(prompt, llm_config=llm_config)
        
        # Sources are obtained from the helper function
        # Ensure database is persisted after query
        data_service.persist_chroma_db()
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise # Re-raise the exception

def query_lightrag(query_text: str, hybrid: bool = False, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query using the light RAG implementation.
    
    LightRAG is a simplified version of RAG that focuses on speed and efficiency:
    1. Uses FAISS for faster similarity search
    2. Simpler document processing
    3. Optional QA chain for faster responses
    4. Less strict filtering and validation
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing light RAG query")
        # Initialize only vectorstore and QA chain
        _ = data_service.vectorstore
        _ = data_service.qa_chain
        
        # Get relevant documents
        results = data_service.vectorstore.similarity_search_with_score(query_text, k=RAG_MAX_DOCUMENTS)
        
        if not results:
            logger.warning("No relevant information found in the database")
            return {"text": "No relevant information found in the database", "sources": []}
        
        # Filter results by similarity score (optional for LightRAG)
        filtered_results = [(doc, score) for doc, score in results if score >= RAG_SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            logger.warning("No sufficiently relevant information found in the database")
            return {"text": "No sufficiently relevant information found in the database", "sources": []}
        
        # Format context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        
        # Extract sources with basic validation
        sources = []
        for doc, _score in filtered_results:
            source = doc.metadata.get("source")
            if source and isinstance(source, str):
                sources.append(source)
            else:
                logger.debug(f"Missing or invalid source in document metadata: {doc.metadata}")
        
        # Use hybrid template if enabled
        if hybrid:
            prompt_template = ChatPromptTemplate.from_template(LIGHTRAG_HYBRID_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response_text = get_llm_response(prompt, llm_config=llm_config)
        else:
            # Use regular QA chain for non-hybrid mode (faster)
            # TODO: Consider how to pass llm_config to the QA chain if needed
            response_text = data_service.qa_chain.invoke({"query": query_text})["result"]
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in light RAG query: {str(e)}", exc_info=True)
        raise # Re-raise the exception

def query_kag(query_text: str, hybrid: bool = False, llm_config: Optional[Dict] = None) -> Dict[str, Union[str, List[str]]]:
    """Query using Knowledge-Augmented Generation (KAG) approach.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing KAG query")
        # Initialize only KAG graph and embedding function
        _ = data_service.embedding_function
        G = data_service.kag_graph
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}", exc_info=True)
        raise # Re-raise the exception

    # Get query embedding
    query_embedding = data_service.embedding_function.embed_query(query_text)
    
    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > GRAPH_MIN_SIMILARITY:
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    initial_nodes = [node for node, _ in relevant_nodes[:GRAPH_MAX_NODES]]

    if not initial_nodes:
        logger.warning("No relevant information found in the knowledge graph")
        return {"text": "No relevant information found in the knowledge graph", "sources": []}

    # Collect context and relationships through graph traversal
    context_parts = []
    relationships = []
    sources = set()
    visited_nodes = set()
    
    def traverse_graph(node, depth=0):
        """Recursively traverse the graph to collect related information."""
        if depth > GRAPH_MAX_DEPTH or node in visited_nodes:
            return
        
        visited_nodes.add(node)
        node_data = G.nodes[node]
        
        # Add node content
        context_parts.append(node_data.get('content', ''))
        sources.add(node_data.get('metadata', {}).get('source', 'unknown'))
        
        # Process edges
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            
            # Add relationship information
            if edge_data.get('relation'):
                neighbor_data = G.nodes[neighbor]
                relation_type = edge_data['relation']
                similarity = edge_data.get('similarity', 0)
                
                # Format relationship based on type
                if relation_type == 'semantically_similar':
                    relationships.append(
                        f"- Similar content (similarity: {similarity:.2f}):\n"
                        f"  From: {node_data.get('content', '')[:100]}...\n"
                        f"  To: {neighbor_data.get('content', '')[:100]}..."
                    )
                elif relation_type == 'same_section':
                    relationships.append(
                        f"- Same section ({node_data.get('metadata', {}).get('section_type', 'unknown')}):\n"
                        f"  From: {node_data.get('content', '')[:100]}...\n"
                        f"  To: {neighbor_data.get('content', '')[:100]}..."
                    )
            
            # Recursively traverse
            traverse_graph(neighbor, depth + 1)

    # Start traversal from initial nodes
    for node in initial_nodes:
        traverse_graph(node)

    # Sort context parts by relevance
    context_parts.sort(key=lambda x: cosine_similarity([query_embedding], [data_service.embedding_function.embed_query(x)])[0][0], reverse=True)
    
    # Format the final context
    context_text = "\n\n---\n\n".join(context_parts[:GRAPH_MAX_NODES])
    relationships_text = "\n\n".join(relationships)
    
    # Use hybrid template if enabled
    template = KAG_HYBRID_TEMPLATE if hybrid else KAG_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(
        context=context_text,
        relationships=relationships_text,
        question=query_text
    )
    
    response_text = get_llm_response(prompt, llm_config=llm_config)
    return {"text": response_text, "sources": list(sources)}

if __name__ == "__main__":
    main()
