import logging
from typing import Any, Optional, List, Dict, Tuple

from langchain.schema import Document

from query.llm_service import get_model_context_length, truncate_history
# --- Modular Template Imports ---
from query.templates import (
    BASE_RESPONSE_STRUCTURE,
    STRICT_CONTEXT_INSTRUCTIONS, HYBRID_INSTRUCTIONS, OPTIMIZED_INSTRUCTIONS,
    CONTEXT_BLOCK, SOURCES_BLOCK, RELATIONSHIPS_BLOCK, INITIAL_ANSWER_BLOCK, QUESTION_BLOCK,
    KAG_CONTEXT_TYPE, STANDARD_CONTEXT_TYPE,
    KAG_RELATIONSHIP_QUALIFIER, KAG_RELATIONSHIP_QUALIFIER_CITE,
    KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT, KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID, KAG_SPECIFIC_DETAIL_INSTRUCTION_OPTIMIZED,
    EMPTY_STRING
)
# --- End Modular Template Imports ---


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
    # --- Flags to determine prompt structure ---
    optimize: bool = False,
    hybrid: bool = False,
    rag_type: str = 'rag', # Need rag_type to determine KAG specifics
    # --- Data specific to optimized flow ---
    draft_answer: Optional[str] = None,
    # --- Common parameters ---
    reserved_for_context: int = RESERVED_FOR_RETRIEVAL_DEFAULT
) -> Tuple[Optional[List[Dict[str, str]]], int]:
    """Calculates available tokens for retrieval context based on dynamic prompt assembly."""
    context_length = get_model_context_length(llm_config)
    fixed_tokens = 0 # Accumulator for non-history, non-retrieved-context tokens

    # --- Estimate tokens for static parts of the base structure ---
    # Remove all placeholders to estimate the fixed text
    base_structure_text = BASE_RESPONSE_STRUCTURE.replace("{assistant_persona}", "") \
                                                 .replace("{persona_description}", "") \
                                                 .replace("{initial_answer_block_placeholder}", "") \
                                                 .replace("{context_block_placeholder}", "") \
                                                 .replace("{relationships_block_placeholder}", "") \
                                                 .replace("{sources_block_placeholder}", "") \
                                                 .replace("{instructions}", "") \
                                                 .replace("{question_block_placeholder}", "") \
                                                 .replace("Answer:", "")
    fixed_tokens += _estimate_tokens(base_structure_text)

    # --- Estimate tokens based on flags ---
    # Determine KAG specifics
    is_kag = (rag_type == 'kag')
    kag_specific_instruction = EMPTY_STRING
    relationship_qualifier = EMPTY_STRING
    relationship_qualifier_cite = EMPTY_STRING
    if is_kag:
        relationship_qualifier = KAG_RELATIONSHIP_QUALIFIER
        relationship_qualifier_cite = KAG_RELATIONSHIP_QUALIFIER_CITE
        # Estimate relationship block overhead (label + formatting)
        fixed_tokens += _estimate_tokens(RELATIONSHIPS_BLOCK.replace("{relationships}", ""))

    # Estimate instruction set tokens
    context_type_label = KAG_CONTEXT_TYPE if is_kag else STANDARD_CONTEXT_TYPE
    context_type_label_lower = context_type_label.lower()
    if optimize:
        if not draft_answer:
            raise ValueError("Draft answer must be provided for optimized context calculation.")
        fixed_tokens += _estimate_tokens(query_text) # Original query
        draft_answer_tokens = _estimate_tokens(draft_answer)
        fixed_tokens += draft_answer_tokens
        fixed_tokens += _estimate_tokens(INITIAL_ANSWER_BLOCK.replace("{draft_answer}", "")) # Block overhead
        fixed_tokens += _estimate_tokens(QUESTION_BLOCK.replace("{question}", "").replace("Question:", "Original Query:")) # Block overhead + label change
        if is_kag: kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_OPTIMIZED
        instruction_text = OPTIMIZED_INSTRUCTIONS.format(kag_specific_instruction_placeholder=kag_specific_instruction)
        fixed_tokens += _estimate_tokens(instruction_text)
    elif hybrid:
        fixed_tokens += _estimate_tokens(query_text) # Query
        fixed_tokens += _estimate_tokens(QUESTION_BLOCK.replace("{question}", "")) # Block overhead
        if is_kag: kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID
        instruction_text = HYBRID_INSTRUCTIONS.format(context_type=context_type_label, relationship_qualifier=relationship_qualifier, relationship_qualifier_cite=relationship_qualifier_cite, kag_specific_instruction_placeholder=kag_specific_instruction)
        fixed_tokens += _estimate_tokens(instruction_text)
    else: # Strict Context
        fixed_tokens += _estimate_tokens(query_text) # Query
        fixed_tokens += _estimate_tokens(QUESTION_BLOCK.replace("{question}", "")) # Block overhead
        if is_kag: kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT
        instruction_text = STRICT_CONTEXT_INSTRUCTIONS.format(context_type=context_type_label, context_type_lower=context_type_label_lower, relationship_qualifier=relationship_qualifier, relationship_qualifier_cite=relationship_qualifier_cite, kag_specific_instruction_placeholder=kag_specific_instruction)
        fixed_tokens += _estimate_tokens(instruction_text)

    # Estimate overhead for context and sources blocks (labels + formatting)
    fixed_tokens += _estimate_tokens(CONTEXT_BLOCK.replace("{context_type}", "").replace("{context}", ""))
    fixed_tokens += _estimate_tokens(SOURCES_BLOCK.replace("{sources}", ""))

    # Calculate max tokens for history based on remaining space AFTER fixed elements and reserved context
    max_history_tokens = context_length - fixed_tokens - reserved_for_context - CONTEXT_SAFETY_MARGIN
    max_history_tokens = max(0, max_history_tokens) # Ensure non-negative
    truncated_history, history_tokens = truncate_history(conversation_history, max_history_tokens)

    # Calculate final available tokens for the retrieved context itself (after accounting for history)
    available_tokens_for_context = context_length - fixed_tokens - history_tokens - CONTEXT_SAFETY_MARGIN
    available_tokens_for_context = max(0, available_tokens_for_context) # Ensure non-negative

    logger.info(f"Context Calc (Optimize={optimize}, Hybrid={hybrid}, Type={rag_type}): Total={context_length}, Fixed={fixed_tokens}, History={history_tokens} (Max={max_history_tokens}), SafetyMargin={CONTEXT_SAFETY_MARGIN} -> Available for Context={available_tokens_for_context}")
    return truncated_history, available_tokens_for_context