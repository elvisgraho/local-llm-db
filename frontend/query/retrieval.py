import logging
from typing import Any, Optional, List, Dict, Tuple

# Modern LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Local Imports
from query.data_service import data_service
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
        
        if chroma_bm25_docs_data and chroma_bm25_docs_data.get("ids"):
            # Handle case where 'documents' might be None or a list containing Nones
            retrieved_contents = chroma_bm25_docs_data.get("documents")
            if retrieved_contents is None:
                retrieved_contents = [None] * len(chroma_bm25_docs_data["ids"])

            for doc_id, content, metadata in zip(
                chroma_bm25_docs_data["ids"], 
                retrieved_contents, 
                chroma_bm25_docs_data["metadatas"]
            ):
                final_content = content
                if not final_content and metadata:
                    # Check common keys for content stored in metadata
                    final_content = metadata.get("page_content") or metadata.get("text") or metadata.get("content") or ""
                
                # If still empty, skip it immediately
                if not final_content or not final_content.strip():
                    continue
                    
                doc = Document(page_content=final_content, metadata=metadata)
                score = id_to_score.get(doc_id, 0.0)
                
                # Only add if we actually have content to search against
                if final_content.strip():
                    bm25_docs_with_scores.append((doc, score))

        return _apply_metadata_filter(bm25_docs_with_scores, metadata_filter)
        
    except Exception as e:
        logger.error(f"Error during BM25 search: {e}", exc_info=True)
        return []