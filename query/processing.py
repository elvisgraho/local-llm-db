import logging
from typing import Any, Optional, List, Dict, Union, Tuple, Set

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from query.data_service import data_service
from query.llm_service import get_llm_response
from query.query_helpers import _estimate_tokens, _reorder_documents_for_context
# --- Modular Template Imports ---
from query.templates import (
    BASE_RESPONSE_STRUCTURE,
    # Personas
    PERSONA_PRECISE_ACCURATE, PERSONA_KNOWLEDGE_AWARE, PERSONA_COMPREHENSIVE, PERSONA_LIGHTWEIGHT, # noqa
    DESC_PRECISE_ACCURATE, DESC_KNOWLEDGE_AWARE, DESC_COMPREHENSIVE, DESC_LIGHTWEIGHT, # noqa
    # Instruction Sets
    STRICT_CONTEXT_INSTRUCTIONS, HYBRID_INSTRUCTIONS, # OPTIMIZED_INSTRUCTIONS removed
    # Optional Blocks
    CONTEXT_BLOCK, SOURCES_BLOCK, RELATIONSHIPS_BLOCK, QUESTION_BLOCK, # INITIAL_ANSWER_BLOCK removed
    # Helper Strings
    KAG_CONTEXT_TYPE, STANDARD_CONTEXT_TYPE,
    KAG_RELATIONSHIP_QUALIFIER, KAG_RELATIONSHIP_QUALIFIER_CITE,
    KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT, KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID, # KAG_SPECIFIC_DETAIL_INSTRUCTION_OPTIMIZED removed
    EMPTY_STRING
)
# --- End Modular Template Imports ---

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
    # --- Parameters for standard and optimized flow ---
    final_docs: List[Document], # Docs selected based on original or refined query
    sources: List[str], # Sources from selected docs
    rag_type: str,
    llm_config: Optional[Dict],
    truncated_history: Optional[List[Dict[str, str]]],
    estimated_context_tokens: int,
    hybrid: bool = False, # Flag to allow general knowledge fallback
    formatted_relationships: Optional[List[str]] = None # KAG relationships
) -> Dict[str, Union[str, List[str], int]]:
    """Dynamically builds the prompt using modular components and calls the LLM."""
    # --- Handle case where no documents are found ---
    if not final_docs:
        # No draft answer to return anymore.
        # If KAG and relationships exist, still proceed to format them. Otherwise, return no info.
        if not (rag_type == 'kag' and formatted_relationships):
            logger.warning(f"No documents provided to _generate_response for {rag_type} mode and no KAG relationships.")
            no_info_msg = "No relevant information found in the knowledge graph." if rag_type == 'kag' else "No relevant information found in the database."
            return {"text": no_info_msg, "sources": [], "estimated_context_tokens": 0}


    # --- Prepare core data ---
    reordered_docs = _reorder_documents_for_context(final_docs)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in reordered_docs])
    sources_text = "\n".join(f"- {s}" for s in sources if s and s != 'unknown')
    if not sources_text: sources_text = "No specific sources identified for this context."

    # --- Dynamically Assemble Prompt Components ---
    assistant_persona = PERSONA_PRECISE_ACCURATE # Default
    persona_description = DESC_PRECISE_ACCURATE # Default
    instructions = ""
    initial_answer_block_placeholder = EMPTY_STRING # No longer used
    context_block_placeholder = EMPTY_STRING
    relationships_block_placeholder = EMPTY_STRING
    sources_block_placeholder = EMPTY_STRING
    question_block_placeholder = EMPTY_STRING

    # Determine Context Type Label
    context_type_label = KAG_CONTEXT_TYPE if rag_type == 'kag' else STANDARD_CONTEXT_TYPE
    context_type_label_lower = context_type_label.lower()

    # Determine KAG specific instruction details
    kag_specific_instruction = EMPTY_STRING
    relationship_qualifier = EMPTY_STRING
    relationship_qualifier_cite = EMPTY_STRING
    if rag_type == 'kag':
        relationship_qualifier = KAG_RELATIONSHIP_QUALIFIER
        relationship_qualifier_cite = KAG_RELATIONSHIP_QUALIFIER_CITE

    # Select Instructions and Persona based on flags
    # The 'optimize' flag now only affects retrieval query, not response generation instructions.
    if hybrid:
        logger.debug(f"Assembling Hybrid prompt for {rag_type}")
        instructions_base = HYBRID_INSTRUCTIONS
        if rag_type == 'kag':
            assistant_persona = PERSONA_KNOWLEDGE_AWARE # KAG Hybrid
            persona_description = DESC_KNOWLEDGE_AWARE
            kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID
        elif rag_type == 'lightrag':
             assistant_persona = PERSONA_LIGHTWEIGHT # LightRAG Hybrid
             persona_description = DESC_LIGHTWEIGHT
        else: # Default RAG Hybrid
             assistant_persona = PERSONA_COMPREHENSIVE
             persona_description = DESC_COMPREHENSIVE
        question_block_placeholder = QUESTION_BLOCK.format(question=query_text)
    else: # Strict Context (hybrid=False)
        logger.debug(f"Assembling Strict Context prompt for {rag_type}")
        instructions_base = STRICT_CONTEXT_INSTRUCTIONS
        if rag_type == 'kag':
            assistant_persona = PERSONA_KNOWLEDGE_AWARE # KAG Strict
            persona_description = DESC_KNOWLEDGE_AWARE
            kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT
        else: # RAG/LightRAG Strict
            assistant_persona = PERSONA_PRECISE_ACCURATE
            persona_description = DESC_PRECISE_ACCURATE
        question_block_placeholder = QUESTION_BLOCK.format(question=query_text)

    # Format instructions with KAG specifics if applicable
    instructions = instructions_base.format(
        context_type=context_type_label,
        context_type_lower=context_type_label_lower,
        relationship_qualifier=relationship_qualifier,
        relationship_qualifier_cite=relationship_qualifier_cite,
        kag_specific_instruction_placeholder=kag_specific_instruction
    )

    # Format optional blocks
    context_block_placeholder = CONTEXT_BLOCK.format(context_type=context_type_label, context=context_text)
    sources_block_placeholder = SOURCES_BLOCK.format(sources=sources_text)

    # Format relationships block only if KAG and relationships exist
    relationships_text = ""
    if rag_type == 'kag' and formatted_relationships:
        relationships_text = "\n\n".join(formatted_relationships)
        if not relationships_text: # Handle case where list is empty
             relationships_text = "No specific relationships were identified for this context."
        relationships_block_placeholder = RELATIONSHIPS_BLOCK.format(relationships=relationships_text)

    # Assemble the final prompt using the base structure
    final_prompt_str = BASE_RESPONSE_STRUCTURE.format(
        assistant_persona=assistant_persona,
        persona_description=persona_description,
        initial_answer_block_placeholder=initial_answer_block_placeholder,
        context_block_placeholder=context_block_placeholder,
        relationships_block_placeholder=relationships_block_placeholder,
        sources_block_placeholder=sources_block_placeholder,
        instructions=instructions,
        question_block_placeholder=question_block_placeholder
    )

    # Get response from LLM, passing truncated history
    response_text = get_llm_response(final_prompt_str, llm_config=llm_config, conversation_history=truncated_history)

    return {"text": response_text, "sources": sources, "estimated_context_tokens": estimated_context_tokens}