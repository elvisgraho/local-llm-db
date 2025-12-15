import logging
from typing import Any, Optional, List, Dict, Tuple

# Modern LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Local Imports
from query.data_service import data_service
# Ensure query_helpers has the NEW logic-aware _apply_metadata_filter
from query.query_helpers import _apply_metadata_filter

logger = logging.getLogger(__name__)

def _retrieve_semantic(
    query_text: str, 
    db: VectorStore, 
    k: int, 
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Tuple[Document, float]]:
    """
    Performs semantic search with Post-Retrieval Filtering.
    """
    try:
        # Fetch 4x candidates to account for Python-side filtering attrition
        # e.g., If k=10, we fetch 40, filter down to ~15, then return top 10.
        fetch_k = k * 4
        
        # Hard cap the internal fetch to prevent retrieving entire DB
        # If MAX_INITIAL=100, k=100 -> fetch_k=400. This is upper bound of safety.
        if fetch_k > 400: 
            fetch_k = 400

        logger.debug(f"Semantic Search: Fetching {fetch_k} candidates for post-filtering.")
        
        # 1. Retrieve Raw (No DB Filter)
        raw_results = db.similarity_search_with_score(
            query_text, 
            k=fetch_k
        )

        # 2. Apply Logic Filter in Python
        filtered_results = _apply_metadata_filter(raw_results, metadata_filter)

        # 3. Return top k
        return filtered_results[:k]

    except Exception as e:
        error_msg = str(e)
        if "dimension" in error_msg:
             logger.error("Configuration Error: Embedding Dimension Mismatch.")
        else:
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
    bm25_data = data_service._bm25_cache.get((rag_type, db_name))
    
    if not bm25_data or bm25_data[0] is None:
        data_service.get_bm25_index(rag_type, db_name)
        bm25_data = data_service._bm25_cache.get((rag_type, db_name))
        
    if not bm25_data or bm25_data[0] is None:
        logger.warning(f"BM25 index missing for {rag_type}/{db_name}.")
        return []

    bm25, bm25_corpus, bm25_doc_ids = bm25_data

    try:
        tokenized_query = query_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top indices (k * 4 for filtering buffer)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k*4]
        
        # Filter zero scores
        top_candidates_indices = [i for i in top_indices if bm25_scores[i] > 0]

        if not top_candidates_indices:
            return []

        # Map to IDs
        target_ids = [bm25_doc_ids[i] for i in top_candidates_indices]
        id_to_score = {bm25_doc_ids[i]: bm25_scores[i] for i in top_candidates_indices}

        # Retrieve docs
        chroma_data = db.get(ids=target_ids, include=["metadatas", "documents"])
        
        results = []
        if chroma_data and chroma_data.get("ids"):
            ids = chroma_data["ids"]
            docs = chroma_data.get("documents") or [None] * len(ids)
            metas = chroma_data.get("metadatas") or [{}] * len(ids)

            for doc_id, text, meta in zip(ids, docs, metas):
                final_text = text or meta.get("page_content") or ""
                if final_text.strip():
                    doc = Document(page_content=final_text, metadata=meta)
                    score = id_to_score.get(doc_id, 0.0)
                    results.append((doc, score))

        return _apply_metadata_filter(results, metadata_filter)[:k]
        
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}", exc_info=True)
        return []