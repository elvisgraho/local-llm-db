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
    BASE_RESPONSE_STRUCTURE,
    STRICT_CONTEXT_INSTRUCTIONS, 
    HYBRID_INSTRUCTIONS,
    CONTEXT_BLOCK, 
    SOURCES_BLOCK, 
    RELATIONSHIPS_BLOCK, 
    QUESTION_BLOCK,
    KAG_CONTEXT_TYPE, 
    STANDARD_CONTEXT_TYPE,
    KAG_RELATIONSHIP_QUALIFIER, 
    KAG_RELATIONSHIP_QUALIFIER_CITE,
    KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT, 
    KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID,
    EMPTY_STRING
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
    hybrid: bool = False,
    rag_type: str = 'rag',
    reserved_for_response: int = DEFAULT_RESPONSE_RESERVE
) -> Tuple[List[Dict[str, str]], int]:
    """
    Calculates the token budget available for retrieved context.
    
    Logic:
    Total_Context_Window
    - Fixed_Prompt_Overhead (Instructions + Query Wrapper)
    - Reserved_For_Output (The Answer)
    - Safety_Margin
    - History (Truncated to fit remaining space)
    = Available_For_Documents

    Returns:
        Tuple[List[Dict], int]: (truncated_history, available_tokens_for_context)
    """
    # 1. Get Model Limits
    total_context_length = get_model_context_length(llm_config)
    fixed_tokens = 0

    # 2. Estimate Fixed Template Components
    # We estimate the overhead of the structural blocks by stripping placeholders
    base_structure_overhead = BASE_RESPONSE_STRUCTURE \
        .replace("{assistant_persona}", "") \
        .replace("{persona_description}", "") \
        .replace("{initial_answer_block_placeholder}", "") \
        .replace("{context_block_placeholder}", "") \
        .replace("{relationships_block_placeholder}", "") \
        .replace("{sources_block_placeholder}", "") \
        .replace("{instructions}", "") \
        .replace("{question_block_placeholder}", "") \
        .replace("Answer:", "")
    
    fixed_tokens += _estimate_tokens(base_structure_overhead)

    # 3. Handle KAG Specifics
    is_kag = (rag_type == 'kag')
    kag_specific_instruction = EMPTY_STRING
    relationship_qualifier = EMPTY_STRING
    relationship_qualifier_cite = EMPTY_STRING
    
    if is_kag:
        relationship_qualifier = KAG_RELATIONSHIP_QUALIFIER
        relationship_qualifier_cite = KAG_RELATIONSHIP_QUALIFIER_CITE
        # Add overhead for the Relationships block
        fixed_tokens += _estimate_tokens(RELATIONSHIPS_BLOCK.replace("{relationships}", ""))

    # 4. Construct Instruction Block
    context_type_label = KAG_CONTEXT_TYPE if is_kag else STANDARD_CONTEXT_TYPE
    
    # but the prompt structure relies on the 'hybrid' flag.
    if hybrid:
        if is_kag:
            kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID
        instruction_text = HYBRID_INSTRUCTIONS.format(
            context_type=context_type_label, 
            relationship_qualifier=relationship_qualifier, 
            relationship_qualifier_cite=relationship_qualifier_cite, 
            kag_specific_instruction_placeholder=kag_specific_instruction
        )
    else:
        if is_kag:
            kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT
        instruction_text = STRICT_CONTEXT_INSTRUCTIONS.format(
            context_type=context_type_label, 
            context_type_lower=context_type_label.lower(), 
            relationship_qualifier=relationship_qualifier, 
            relationship_qualifier_cite=relationship_qualifier_cite, 
            kag_specific_instruction_placeholder=kag_specific_instruction
        )
    
    fixed_tokens += _estimate_tokens(instruction_text)

    # 5. Add Query and Block Overheads
    fixed_tokens += _estimate_tokens(query_text)
    fixed_tokens += _estimate_tokens(QUESTION_BLOCK.replace("{question}", ""))
    fixed_tokens += _estimate_tokens(CONTEXT_BLOCK.replace("{context_type}", "").replace("{context}", ""))
    fixed_tokens += _estimate_tokens(SOURCES_BLOCK.replace("{sources}", ""))

    # 6. Calculate Remaining Budget for History
    # Budget = Total - Fixed - Output_Reserve - Safety
    remaining_tokens = total_context_length - fixed_tokens - reserved_for_response - CONTEXT_SAFETY_MARGIN
    
    # FIX: Prevent negative calculation
    if remaining_tokens < 100:
        logger.warning(f"Extreme low context warning. Remaining: {remaining_tokens}. Forcing minimum buffer.")
        remaining_tokens = 100 
    
    # We allow history to take up to 40% of the remaining space
    max_history_tokens = max(0, int(remaining_tokens * 0.4)) 
    
    truncated_history, history_tokens_used = truncate_history(conversation_history, max_history_tokens)

    # 7. Final Calculation for Context Documents
    available_tokens_for_context = remaining_tokens - history_tokens_used
    available_tokens_for_context = max(1000, available_tokens_for_context)

    logger.info(
        f"Context Budget [{rag_type.upper()}|Hybrid:{hybrid}]: "
        f"Total={total_context_length} | Fixed={fixed_tokens} | ResponseReserve={reserved_for_response} | "
        f"History={history_tokens_used} | AvailableForDocs={available_tokens_for_context}"
    )
    
    return truncated_history, available_tokens_for_context