import logging
from typing import Any, Optional, List, Dict, Union, Tuple, Set

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from query.data_service import data_service
from query.llm_service import get_llm_response
from query.query_helpers import _estimate_tokens, _reorder_documents_for_context
from query.templates import (
    RAG_ONLY_TEMPLATE,
    KAG_TEMPLATE,
    HYBRID_TEMPLATE,
    LIGHTRAG_HYBRID_TEMPLATE,
    KAG_HYBRID_TEMPLATE
)

logger = logging.getLogger(__name__)

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