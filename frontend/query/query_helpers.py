import logging
from typing import Any, Optional, List, Dict, Tuple

try:
    import tiktoken
except ImportError:
    tiktoken = None

# Modern LangChain Core Import
from langchain_core.documents import Document

# Local Imports
from query.llm_service import get_model_context_length, truncate_history
from query.templates import (
    RAG_USER_TEMPLATE,
    STRICT_CONTEXT_INSTRUCTIONS,
    QUESTION_BLOCK,
    STRICT_CONTEXT_INSTRUCTIONS, 
    QUESTION_BLOCK,
)

# Configure Logger
logger = logging.getLogger(__name__)

# --- Constants for Token Estimation ---
TOKEN_ESTIMATION_FACTOR = 4  # Approximation: 1 token ~= 4 characters
CONTEXT_SAFETY_MARGIN = 200  # Buffer for prompt variations and special tokens
DEFAULT_RESPONSE_RESERVE = 1500  # Tokens reserved for the LLM's generated answer

# --- Helper Functions ---

def _estimate_tokens(text: str) -> int:
    """
    Estimates tokens using tiktoken if available, else heuristic.
    Essential for Code/Log heavy contexts.
    """
    if not text:
        return 0
        
    if tiktoken:
        try:
            # cl100k_base is used by GPT-4/3.5 and many modern embeddings
            enc = tiktoken.get_encoding("cl100k_base") 
            return len(enc.encode(text))
        except Exception:
            pass
            
    # Heuristic fallback: Code is denser than prose
    return len(text) // 3.5 

def _reorder_documents_for_context(docs: List[Document]) -> List[Document]:
    """
    Reorders documents to mitigate the 'Lost in the Middle' phenomenon.
    Places the most relevant document first, and the second most relevant last.
    """
    if len(docs) >= 3:
        logger.debug(f"Reordering {len(docs)} documents for context optimization.")
        # Assumes docs are currently sorted by score (Descending)
        most_relevant = docs[0]
        second_most_relevant = docs[1]
        middle_docs = docs[2:]
        return [most_relevant] + middle_docs + [second_most_relevant]
    return docs

def _apply_metadata_filter(
    docs_with_scores: List[Tuple[Document, float]], 
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Tuple[Document, float]]:
    """
    Applies exact-match metadata filtering to a list of (Document, score) tuples.
    """
    if not metadata_filter:
        return docs_with_scores

    filtered = []
    for doc, score in docs_with_scores:
        # Check if all filter criteria match document metadata
        match = all(
            doc.metadata.get(key) == value 
            for key, value in metadata_filter.items()
        )
        if match:
            filtered.append((doc, score))
            
    logger.debug(f"Metadata filter: {len(docs_with_scores)} -> {len(filtered)} docs.")
    return filtered

def _calculate_available_context(
    query_text: str,
    conversation_history: Optional[List[Dict[str, str]]],
    llm_config: Optional[Dict[str, Any]],
    reserved_for_response: int = 1500
) -> Tuple[List[Dict[str, str]], int]:
    
    # 1. Get Model Limits
    total_context_length = get_model_context_length(llm_config)
    fixed_tokens = 0

    # 2. Get User's System Prompt Size
    user_system_prompt = llm_config.get('system_prompt', "")
    fixed_tokens += _estimate_tokens(user_system_prompt)

    # 3. Add Overhead for RAG Instructions (Estimate)
    # We use the larger of the two instruction sets to be safe
    instruction_overhead = _estimate_tokens(STRICT_CONTEXT_INSTRUCTIONS) + 50 
    fixed_tokens += instruction_overhead

    # 4. Add Overhead for User Template Wrapper
    # (Context placeholders, sources placeholders, etc)
    user_wrapper_overhead = _estimate_tokens(RAG_USER_TEMPLATE)
    fixed_tokens += user_wrapper_overhead

    # 5. Add Query and Block Overheads
    fixed_tokens += _estimate_tokens(query_text)
    fixed_tokens += _estimate_tokens(QUESTION_BLOCK)
    # Add a buffer for the context headers ("Context:", "Sources:")
    fixed_tokens += 50 

    # 6. Calculate Remaining Budget
    remaining_tokens = total_context_length - fixed_tokens - reserved_for_response - 200 # Safety margin
    
    if remaining_tokens < 100:
        remaining_tokens = 100 
    
    # 7. Allocate History
    max_history_tokens = max(0, int(remaining_tokens * 0.4)) 
    truncated_history, history_tokens_used = truncate_history(conversation_history, max_history_tokens)

    # 8. Result
    available_tokens_for_context = remaining_tokens - history_tokens_used
    available_tokens_for_context = max(1000, available_tokens_for_context)
    
    return truncated_history, available_tokens_for_context