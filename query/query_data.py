import argparse
import logging
from typing import Any, Optional, List, Dict, Union, Tuple, Set

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_core.vectorstores import VectorStore
# Removed networkx and cosine_similarity imports

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
from query.llm_service import get_llm_response, optimize_query # Removed get_model_context_length, truncate_history
from query.global_vars import (
    RAG_SIMILARITY_THRESHOLD, # Used for lightrag filtering
    RAG_MAX_DOCUMENTS, # Target for final context document count
    GRAPH_MAX_NODES, # Used for KAG node selection/context limit
)
from query.query_helpers import ( # Added import
    _calculate_available_context
)
from query.retrieval import ( # Added import
    _retrieve_semantic,
    _retrieve_keyword,
    _retrieve_graph,
    INITIAL_RETRIEVAL_MULTIPLIER, # Moved constant
    MAX_INITIAL_RETRIEVAL_LIMIT   # Moved constant
)
from query.processing import ( # Added import
    _rerank_results,
    _select_docs_for_context,
    _generate_response
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
