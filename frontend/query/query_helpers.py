import logging
import tiktoken
import math
from typing import Any, Optional, List, Dict, Tuple
# Modern LangChain Core Import
from langchain_core.documents import Document

# Local Imports
from query.llm_service import get_model_context_length, truncate_history
from query.templates import (
    RAG_USER_TEMPLATE,
    STRICT_CONTEXT_INSTRUCTIONS,
    QUESTION_BLOCK
)

# Configure Logger
logger = logging.getLogger(__name__)

# --- Constants for Token Estimation ---
TOKEN_ESTIMATION_FACTOR = 4  # Approximation: 1 token ~= 4 characters
CONTEXT_SAFETY_MARGIN = 200  # Buffer for prompt variations and special tokens
DEFAULT_RESPONSE_RESERVE = 1500  # Tokens reserved for the LLM's generated answer

# --- Helper Functions ---

def estimate_tokens(text: str, model_encoding: str = "cl100k_base") -> int:
    """
    Estimates tokens. Prioritizes safety (overestimation) in heuristic fallback.
    """
    if not text:
        return 0
        
    # Check if tiktoken is actually imported in local/global scope
    if 'tiktoken' in globals():
        try:
            enc = tiktoken.get_encoding(model_encoding)
            # encode_ordinary is faster as it ignores special tokens
            return len(enc.encode_ordinary(text))
        except Exception:
            pass
            
    # Fallback: Use 3.0 chars/token for safety buffer. 
    # Use math.ceil to ensure we don't return 0 for non-empty short strings.
    return math.ceil(len(text) / 3.0)

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

def apply_metadata_filter(
    docs_with_scores: List[Tuple[Document, float]], 
    metadata_filter: Optional[Dict[str, Any]]
) -> List[Tuple[Document, float]]:
    """
    Applies logic-aware filtering. 
    Handles flattened strings (e.g. "recon, poc") and Entity formats "Name (Category)".
    """
    if not metadata_filter:
        return docs_with_scores

    filtered = []
    for doc, score in docs_with_scores:
        match = True
        for key, target_val in metadata_filter.items():
            # 1. Get actual value (default to empty string)
            actual_val = doc.metadata.get(key, "")
            
            # 2. Normalize to strings
            actual_str = str(actual_val).lower()
            target_str = str(target_val).lower()

            # 3. Check Membership
            # We flattened lists into strings: "item1, item2 (CAT), item3"
            if "," in actual_str:
                actual_list = [item.strip() for item in actual_str.split(",")]
                
                # Check if target is in the list (Relaxed for Entities)
                found = False
                for item in actual_list:
                    # Case A: Exact Match (e.g. "T1059")
                    if target_str == item:
                        found = True
                        break
                    # Case B: Entity Match "Mimikatz" matches "mimikatz (tool)"
                    # We check if target is the "prefix" of the item
                    if item.startswith(target_str) and "(" in item:
                        found = True
                        break
                
                if not found:
                    match = False
                    break
            else:
                # 4. Fallback for single values (Exact or Substring)
                # For safety with loose tags, we prefer exact match if it's not a list
                if actual_str != target_str:
                    # Optional: Allow substring if you want looser filtering
                    # if target_str not in actual_str: 
                    match = False
                    break
        
        if match:
            filtered.append((doc, score))
            
    return filtered

def calculate_available_context(
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
    fixed_tokens += estimate_tokens(user_system_prompt)

    # 3. Add Overhead for RAG Instructions (Estimate)
    # We use the larger of the two instruction sets to be safe
    instruction_overhead = estimate_tokens(STRICT_CONTEXT_INSTRUCTIONS) + 50 
    fixed_tokens += instruction_overhead

    # 4. Add Overhead for User Template Wrapper
    # (Context placeholders, sources placeholders, etc)
    user_wrapper_overhead = estimate_tokens(RAG_USER_TEMPLATE)
    fixed_tokens += user_wrapper_overhead

    # 5. Add Query and Block Overheads
    fixed_tokens += estimate_tokens(query_text)
    fixed_tokens += estimate_tokens(QUESTION_BLOCK)
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