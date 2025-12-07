import logging
from typing import Any, Optional, List, Dict, Tuple, Set

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Modern LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Local Imports
from query.data_service import data_service
from query.global_vars import (
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_NODES
)
from query.query_helpers import _apply_metadata_filter

logger = logging.getLogger(__name__)

# Constants
INITIAL_RETRIEVAL_MULTIPLIER = 2  # Retrieve more docs initially for better reranking
MAX_INITIAL_RETRIEVAL_LIMIT = 50  # Hard limit on initial document retrieval count

def _retrieve_semantic(
    query_text: str, 
    db: VectorStore, 
    k: int, 
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Tuple[Document, float]]:
    """
    Performs semantic search using ChromaDB with dimension mismatch safeguard.
    """
    try:
        logger.info(f"Performing semantic search (k={k}) with filter: {metadata_filter}")
        
        results = db.similarity_search_with_score(
            query_text, 
            k=k, 
            filter=metadata_filter 
        )
        return results

    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg and ("expecting" in error_msg or "got" in error_msg):
            logger.error(
                f"CONFIGURATION ERROR: Embedding Dimension Mismatch.\n"
                f"The Database expects a different model than the one currently active.\n"
                f"Details: {error_msg}\n"
                f"Fix: Update 'global_vars.py' to use the embedding model that matches your DB."
            )
            return []
            
        logger.error(f"Error during semantic search: {e}", exc_info=True)
        return []

def _retrieve_keyword(
    query_text: str, 
    db: VectorStore, 
    rag_type: str, 
    db_name: str, 
    k: int, 
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Tuple[Document, float]]:
    """
    Performs keyword search using the cached BM25 index.
    """
    # Access cache directly (Acknowledging internal implementation detail of DataService)
    # Ideally DataService would expose a method like `get_bm25_search_results`
    bm25_data = data_service._bm25_cache.get((rag_type, db_name))
    
    if not bm25_data or bm25_data[0] is None:
        # Try to trigger a load if missing, or fail gracefully
        data_service.get_bm25_index(rag_type, db_name)
        bm25_data = data_service._bm25_cache.get((rag_type, db_name))
        
    if not bm25_data or bm25_data[0] is None:
        logger.warning(f"BM25 index not available for {rag_type}/{db_name}. Skipping keyword search.")
        return []

    bm25, bm25_corpus, bm25_doc_ids = bm25_data

    try:
        logger.info(f"Performing keyword search (BM25, k={k}).")
        
        # Simple tokenization: Lowercase and split
        tokenized_query = query_text.lower().split()
        
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Zip and Sort: (ID, Corpus, Score)
        # We assume bm25_doc_ids maps 1:1 to corpus indices
        bm25_scored_docs_info = sorted(
            zip(bm25_doc_ids, bm25_corpus, bm25_scores), 
            key=lambda x: x[2], 
            reverse=True
        )
        
        # Select Top K candidates (Score > 0)
        top_bm25_candidates = [
            (doc_id, score) 
            for doc_id, _, score in bm25_scored_docs_info 
            if score > 0
        ][:k]
        
        top_bm25_ids = [doc_id for doc_id, _ in top_bm25_candidates]

        if not top_bm25_ids:
            return []

        # Retrieve full documents from Chroma by ID
        chroma_bm25_docs_data = db.get(ids=top_bm25_ids, include=["metadatas", "documents"])
        
        bm25_docs_with_scores = []
        id_to_score = dict(top_bm25_candidates)
        
        # Reconstruct Documents with Scores
        if chroma_bm25_docs_data and chroma_bm25_docs_data.get("ids"):
            for doc_id, content, metadata in zip(
                chroma_bm25_docs_data["ids"], 
                chroma_bm25_docs_data["documents"], 
                chroma_bm25_docs_data["metadatas"]
            ):
                doc = Document(page_content=content, metadata=metadata)
                score = id_to_score.get(doc_id, 0.0)
                bm25_docs_with_scores.append((doc, score))

        # Apply metadata filter *after* retrieving BM25 candidates
        return _apply_metadata_filter(bm25_docs_with_scores, metadata_filter)
        
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}", exc_info=True)
        return []

def _retrieve_graph(
    query_text: str, 
    rag_type: str, 
    db_name: str, 
    k: int, 
    metadata_filter: Optional[Dict[str, Any]]
) -> Tuple[List[Tuple[Document, float]], List[str]]:
    """
    Optimized Graph Traversal.
    Uses VectorDB for O(1) entry point finding instead of O(N) graph iteration.
    """
    try:
        graph = data_service.get_kag_graph(rag_type, db_name)
        db = data_service.get_chroma_db(rag_type, db_name)
        
        if not graph or not db:
            logger.warning(f"KAG graph or DB not available for {rag_type}/{db_name}.")
            return [], []

        logger.info(f"Performing graph retrieval (KAG, k={k}).")

        # --- OPTIMIZATION START ---
        # 1. Find Entry Nodes via Vector Search (O(log N)) instead of Graph Scan (O(N))
        # We retrieve more candidates (k*3) to ensure we hit nodes that exist in the graph
        entry_candidates = db.similarity_search_with_score(
            query_text, 
            k=min(k * 3, MAX_INITIAL_RETRIEVAL_LIMIT), 
            filter=metadata_filter
        )
        
        start_nodes = []
        initial_scores = {}

        # Map vector docs back to graph nodes using 'id' (assuming 1:1 mapping)
        # In KAG/LightRAG, Document IDs usually match Graph Node IDs
        for doc, score in entry_candidates:
            node_id = doc.metadata.get("id") or doc.metadata.get("doc_id")
            
            # Fallback: Check if page_content maps to a node (rare but possible in some implementations)
            if not node_id and graph.has_node(doc.page_content):
                 node_id = doc.page_content

            # Verify node exists in graph before adding
            if node_id and graph.has_node(node_id):
                start_nodes.append(node_id)
                initial_scores[node_id] = score

        if not start_nodes:
            logger.warning("No vector search results matched graph nodes.")
            return [], []

        logger.info(f"Identified {len(start_nodes)} entry points via VectorDB.")
        # --- OPTIMIZATION END ---

        # 2. Iterative Traversal (Stack-based DFS)
        visited_nodes: Set[str] = set()
        collected_docs: Dict[str, Tuple[Document, float]] = {}
        relationships: List[str] = []
        
        # Max nodes to touch to prevent explosion in dense graphs
        max_visit_count = GRAPH_MAX_NODES * INITIAL_RETRIEVAL_MULTIPLIER
        
        # Initialize stack with entry nodes
        stack = [(node_id, 0) for node_id in start_nodes]
        
        while stack and len(visited_nodes) < max_visit_count:
            current_id, depth = stack.pop()
            
            if current_id in visited_nodes:
                continue
            
            visited_nodes.add(current_id)
            
            if current_id not in graph:
                continue
                
            node_data = graph.nodes[current_id]
            content = node_data.get('content')
            metadata = node_data.get('metadata', {})
            
            # A. Collect Document
            if content:
                # Use vector score if available, otherwise decay score based on hop distance
                # Score = Initial_Score * (decay_factor ^ depth)
                base_score = initial_scores.get(current_id, 0.8) # Default 0.8 for non-entry nodes
                decayed_score = base_score * (0.9 ** depth)
                
                doc = Document(page_content=content, metadata=metadata)
                
                # Update if better score found
                if current_id not in collected_docs or decayed_score > collected_docs[current_id][1]:
                    collected_docs[current_id] = (doc, decayed_score)

            # B. Stop traversal if max depth reached
            if depth >= GRAPH_MAX_DEPTH:
                continue

            # C. Process Neighbors
            neighbors = list(graph.neighbors(current_id))
            for neighbor_id in neighbors:
                edge_data = graph.get_edge_data(current_id, neighbor_id)
                if not edge_data: continue
                
                # Record Relationship (Only if within depth limit)
                if edge_data.get('relation'):
                    _format_relationship(
                        graph, current_id, neighbor_id, edge_data, relationships
                    )
                
                if neighbor_id not in visited_nodes:
                    stack.append((neighbor_id, depth + 1))

        # 3. Finalize
        final_docs = list(collected_docs.values())
        final_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Deduplicate relationships
        unique_relationships = sorted(list(set(relationships)))
        
        return final_docs[:k], unique_relationships[:15] # Limit relationships count

    except Exception as e:
        logger.error(f"Error during graph retrieval: {e}", exc_info=True)
        return [], []
    
    
def _format_relationship(
    graph: nx.Graph, 
    source_id: str, 
    target_id: str, 
    edge_data: Dict, 
    relationships_list: List[str]
) -> None:
    """Helper to format a relationship string for the prompt."""
    try:
        source_node = graph.nodes[source_id]
        target_node = graph.nodes[target_id]
        
        rel_type = edge_data.get('relation', 'connected_to')
        sim_score = edge_data.get('similarity', 0)
        
        # Snippets for context
        src_content = source_node.get('content', '')[:100]
        tgt_content = target_node.get('content', '')[:100]
        
        rel_str = f"- Relation '{rel_type}'"
        
        if rel_type == 'semantically_similar':
             rel_str += f" (similarity: {sim_score:.2f})"
        elif rel_type == 'same_section':
             section = source_node.get('metadata', {}).get('section_type', 'unknown')
             rel_str += f" (section: {section})"
             
        rel_str += f":\n  From: {src_content}...\n  To: {tgt_content}..."
        relationships_list.append(rel_str)
        
    except Exception as e:
        logger.debug(f"Failed to format relationship: {e}")