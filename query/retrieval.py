import logging
from typing import Any, Optional, List, Dict, Tuple

import networkx as nx
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
from sklearn.metrics.pairwise import cosine_similarity

from query.data_service import data_service
from query.global_vars import (
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_NODES,
    GRAPH_MIN_SIMILARITY
)
from query.query_helpers import _apply_metadata_filter # Import from helpers

logger = logging.getLogger(__name__)

# Constants moved from query_data.py
INITIAL_RETRIEVAL_MULTIPLIER = 2 # Retrieve more docs initially for better reranking
MAX_INITIAL_RETRIEVAL_LIMIT = 50 # Hard limit on initial document retrieval count

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
            return [], [] # Return empty lists for both docs and relationships

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
            return [], [] # Return empty lists

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
        return [], [] # Return empty lists on error