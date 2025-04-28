import logging
from typing import Any, Optional, List, Dict, Tuple

from langchain.schema import Document

from query.llm_service import get_model_context_length, truncate_history

logger = logging.getLogger(__name__)

# --- Constants for Token Estimation ---
TOKEN_ESTIMATION_FACTOR = 4 # Simple approximation: 1 token ~= 4 characters
CONTEXT_SAFETY_MARGIN = 200 # Reserve tokens for safety margin, prompt variations
RESERVED_FOR_RETRIEVAL_DEFAULT = 3000 # Default tokens reserved for context (can be overridden)

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