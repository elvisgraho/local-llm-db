import argparse
import logging
from typing import Any, Optional, List, Dict, Union, Tuple

# --- Modern LangChain Core Imports ---
from langchain_core.prompts import ChatPromptTemplate

# --- Internal Imports ---
from query.database_paths import DEFAULT_DB_NAME
from query.data_service import data_service
from query.templates import DIRECT_TEMPLATE
from query.llm_service import get_llm_response
from query.global_vars import (
    RAG_SIMILARITY_THRESHOLD
)
from query.query_helpers import calculate_available_context
from query.retrieval import (
    _retrieve_semantic,
    _retrieve_keyword,
    _retrieve_graph,
    MAX_INITIAL_RETRIEVAL_LIMIT
)
from query.processing import (
    _rerank_results,
    select_docs_for_context,
    _generate_response
)

from query.query_refinement import query_processor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def _get_db_or_raise(rag_type: str, db_name: str) -> Any:
    """Helper to retrieve Chroma DB or raise error if missing."""
    db = data_service.get_chroma_db(rag_type, db_name)
    if not db:
        raise ValueError(f"Chroma DB '{rag_type}/{db_name}' not loaded.")
    return db

def _prepare_retrieval_context(
    query_text: str,
    llm_config: Optional[Dict],
    conversation_history: Optional[List[Dict[str, str]]]
) -> Tuple[str, List[Dict[str, str]], int]:
    
    retrieval_query_text = query_processor.process_query(
        query_text, 
        conversation_history
    )
    
    # 2. Context Calculation
    truncated_history, available_tokens = calculate_available_context(
        query_text=query_text,
        conversation_history=conversation_history,
        llm_config=llm_config
    )
        
    return retrieval_query_text, truncated_history, available_tokens

# --- Query Strategy Functions ---

def query_direct(
    query_text: str, 
    llm_config: Optional[Dict] = None,
    verify: Optional[bool] = False,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Union[str, List[str]]]:
    """
    Query the model directly without using RAG.
    """
    try:
        logger.debug("Processing direct query")
        prompt_template = ChatPromptTemplate.from_template(DIRECT_TEMPLATE)
        
        # If DIRECT_TEMPLATE uses a history placeholder, format it here. 
        # For simplicity, assuming standard template.
        prompt_val = prompt_template.invoke({"question": query_text})
        prompt_str = prompt_val.to_string() 

        response_text = get_llm_response(
            prompt_str, 
            llm_config=llm_config, 
            conversation_history=conversation_history,
            verify=verify
        )
        
        return {"text": response_text, "sources": []}
    except Exception as e:
        logger.error(f"Error in direct query: {str(e)}", exc_info=True)
        raise

def query_rag(
    query_text: str,
    top_k: int = 4,
    hybrid: bool = False,
    verify: Optional[bool] = False,
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """
    Query using standard RAG (Semantic + Keyword + Reranking).
    """
    try:
        # 1. Prepare
        retrieval_query, history, tokens = _prepare_retrieval_context(
            query_text, llm_config, conversation_history
        )

        # 2. Get Resources
        db = _get_db_or_raise('rag', db_name)
        
        # 3. Retrieve
        # We fetch 3x more documents initially to give the Reranker enough candidates to sort.
        # But we enforce a sanity limit (e.g., don't fetch 300 docs if user asks for 100).
        k_initial = min(top_k * 3, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"RAG initial retrieval k = {k_initial} (Target Top-K: {top_k})")

        semantic_results = _retrieve_semantic(retrieval_query, db, k_initial, metadata_filter)
        keyword_results = _retrieve_keyword(retrieval_query, db, 'rag', db_name, k_initial, metadata_filter)

        # 4. Merge & Deduplicate
        combined_dict = {doc.page_content: (doc, score) for doc, score in semantic_results}
        for doc, score in keyword_results:
            if doc.page_content not in combined_dict or score > combined_dict[doc.page_content][1]:
                combined_dict[doc.page_content] = (doc, score)
        
        initial_docs_scores = sorted(list(combined_dict.values()), key=lambda x: x[1], reverse=True)

        # 5. Rerank & Select
        # Rerank and keep only 'top_k' best matches
        reranked_docs_scores = _rerank_results(retrieval_query, initial_docs_scores, top_k)
        
        # Select final docs (cutting off if context limit exceeded)
        final_docs, sources, estimated_tokens = select_docs_for_context(reranked_docs_scores, tokens)

        # 6. Generate
        return _generate_response(
            query_text=query_text,
            final_docs=final_docs,
            sources=sources,
            rag_type='rag',
            llm_config=llm_config,
            truncated_history=history,
            estimated_context_tokens=estimated_tokens,
            hybrid=hybrid,
            verify=verify
        )
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise

def query_lightrag(
    query_text: str,
    top_k: int = 4,
    hybrid: bool = False,
    verify: Optional[bool] = False,
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """
    Query using LightRAG (Semantic + Similarity Threshold).
    """
    try:
        # 1. Prepare
        retrieval_query, history, tokens = _prepare_retrieval_context(
            query_text, llm_config, conversation_history
        )

        # 2. Get Resources
        db = _get_db_or_raise('lightrag', db_name)

        # 3. Retrieve
        k_initial = min(top_k * 3, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"LightRAG initial retrieval k = {k_initial}")
        
        semantic_results = _retrieve_semantic(retrieval_query, db, k_initial, metadata_filter)

        # 4. Filter
        threshold_filtered_docs = [
            (doc, score) for doc, score in semantic_results 
            if score >= RAG_SIMILARITY_THRESHOLD
        ]
        threshold_filtered_docs.sort(key=lambda x: x[1], reverse=True)

        if not threshold_filtered_docs:
            logger.warning("No documents met the similarity threshold for LightRAG.")
            return {
                "text": "No relevant information found matching the similarity criteria.", 
                "sources": [], 
                "estimated_context_tokens": 0
            }

        # 5. Rerank & Select
        # Use top_k here
        reranked_docs_scores = _rerank_results(retrieval_query, threshold_filtered_docs, top_k)
        final_docs, sources, estimated_tokens = select_docs_for_context(reranked_docs_scores, tokens)

        # 6. Generate
        return _generate_response(
            query_text=query_text,
            final_docs=final_docs,
            sources=sources,
            rag_type='lightrag',
            llm_config=llm_config,
            truncated_history=history,
            estimated_context_tokens=estimated_tokens,
            hybrid=hybrid,
            verify=verify
        )
    except Exception as e:
        logger.error(f"Error in LightRAG query: {str(e)}", exc_info=True)
        raise

def query_kag(
    query_text: str,
    top_k: int = 4,
    hybrid: bool = False,
    verify: Optional[bool] = False,
    db_name: str = DEFAULT_DB_NAME,
    llm_config: Optional[Dict] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """
    Query using KAG (Graph RAG).
    """
    try:
        # 1. Prepare
        retrieval_query, history, tokens = _prepare_retrieval_context(
            query_text, llm_config, conversation_history
        )

        # 2. Get Resources
        if not data_service.get_kag_graph('kag', db_name):
             raise ValueError(f"Knowledge Graph 'kag/{db_name}' not loaded.")
        
        _ = _get_db_or_raise('kag', db_name)

        # 3. Retrieve (Graph Search)
        # Graph traversal can explode, so we limit initial node visits carefully
        k_initial = min(top_k * 3, MAX_INITIAL_RETRIEVAL_LIMIT)
        logger.info(f"KAG initial retrieval k = {k_initial}")
        
        initial_docs_scores, formatted_relationships = _retrieve_graph(
            retrieval_query, 'kag', db_name, k_initial, metadata_filter
        )

        if not initial_docs_scores:
            logger.warning("Graph retrieval yielded no documents for KAG.")
            return _generate_response(
                query_text=query_text,
                final_docs=[],
                sources=[],
                rag_type='kag',
                llm_config=llm_config,
                truncated_history=history,
                estimated_context_tokens=0,
                hybrid=hybrid,
                verify=verify,
                formatted_relationships=[]
            )

        # 4. Rerank & Select
        # Use top_k here
        reranked_docs_scores = _rerank_results(retrieval_query, initial_docs_scores, top_k)
        final_docs, sources, estimated_tokens = select_docs_for_context(reranked_docs_scores, tokens)

        # 5. Generate
        return _generate_response(
            query_text=query_text,
            final_docs=final_docs,
            sources=sources,
            rag_type='kag',
            llm_config=llm_config,
            truncated_history=history,
            estimated_context_tokens=estimated_tokens,
            hybrid=hybrid,
            formatted_relationships=formatted_relationships
        )
    except Exception as e:
        logger.error(f"Error in KAG query: {str(e)}", exc_info=True)
        raise

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Run RAG queries using Modern LCEL stack.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--rag_type", type=str, choices=['rag', 'direct', 'lightrag', 'kag'], default='rag')
    parser.add_argument("--db_name", type=str, default=DEFAULT_DB_NAME)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--hybrid", action="store_true")
    
    # New CLI arg for top_k
    parser.add_argument("--top_k", type=int, default=5, help="Number of documents to retrieve.")

    args = parser.parse_args()

    try:
        default_llm_config = {'provider': 'local', 'modelName': ''}
        
        logger.info(f"CLI Start: Type={args.rag_type.upper()} | DB={args.db_name} | TopK={args.top_k}")

        query_func_map = {
            'direct': query_direct,
            'rag': query_rag,
            'lightrag': query_lightrag,
            'kag': query_kag
        }

        query_func = query_func_map[args.rag_type]

        func_args = {
            "query_text": args.query_text,
            "llm_config": default_llm_config,
            "conversation_history": None 
        }
        
        if args.rag_type != 'direct':
            func_args.update({
                "optimize": args.optimize,
                "hybrid": args.hybrid,
                "rag_type": args.rag_type,
                "db_name": args.db_name,
                "top_k": args.top_k  # Pass CLI arg
            })

        result = query_func(**func_args)

        print("\n--- Response ---")
        print(result["text"])
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()