import logging
from typing import Any, Optional, List, Dict, Union, Tuple, Set

# Modern LangChain Core Imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Local Imports
from query.data_service import data_service
from query.llm_service import get_llm_response
from query.query_helpers import _estimate_tokens, _reorder_documents_for_context

# --- Modular Template Imports ---
from query.templates import (
    BASE_RESPONSE_STRUCTURE,
    # Personas
    PERSONA_PRECISE_ACCURATE, PERSONA_KNOWLEDGE_AWARE, 
    PERSONA_COMPREHENSIVE, PERSONA_LIGHTWEIGHT,
    DESC_PRECISE_ACCURATE, DESC_KNOWLEDGE_AWARE, 
    DESC_COMPREHENSIVE, DESC_LIGHTWEIGHT,
    # Instruction Sets
    STRICT_CONTEXT_INSTRUCTIONS, HYBRID_INSTRUCTIONS,
    # Optional Blocks
    CONTEXT_BLOCK, SOURCES_BLOCK, RELATIONSHIPS_BLOCK, QUESTION_BLOCK,
    # Helper Strings
    KAG_CONTEXT_TYPE, STANDARD_CONTEXT_TYPE,
    KAG_RELATIONSHIP_QUALIFIER, KAG_RELATIONSHIP_QUALIFIER_CITE,
    KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT, KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID,
    EMPTY_STRING
)

logger = logging.getLogger(__name__)

# Constants
DOC_SEPARATOR = "\n\n---\n\n"

def _rerank_results(
    query_text: str, 
    results: List[Tuple[Document, float]], 
    k: int
) -> List[Tuple[Document, float]]:
    """
    Reranks retrieval results using the CrossEncoder model from DataService.
    Falls back to original scores if reranker is unavailable or fails.
    """
    reranker = data_service.reranker
    
    if not reranker or not results:
        # Sort by original score just in case
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    try:
        logger.info(f"Reranking {len(results)} results.")
        
        # Prepare pairs for CrossEncoder: [Query, Document Content]
        pairs = [[query_text, doc.page_content] for doc, _ in results]
        
        # Predict scores
        rerank_scores = reranker.predict(pairs)
        
        # Update scores preserving Document objects
        reranked_results = [
            (results[i][0], float(rerank_scores[i])) 
            for i in range(len(results))
        ]
        
        # Sort by new rerank scores
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Selected top {k} results after reranking.")
        return reranked_results[:k]

    except Exception as e:
        logger.error(f"Error during reranking: {e}. Falling back to original scores.", exc_info=True)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

def _select_docs_for_context(
    docs_with_scores: List[Tuple[Document, float]], 
    available_tokens: int
) -> Tuple[List[Document], List[str], int]:
    """
    Selects documents from a sorted list to fit within the estimated token limit.
    Returns: (selected_docs, unique_sources, tokens_used)
    """
    selected_docs = []
    selected_sources = set()
    current_tokens = 0
    separator_tokens = _estimate_tokens(DOC_SEPARATOR)

    logger.info(f"Selecting documents to fit within {available_tokens} estimated tokens.")

    for doc, score in docs_with_scores:
        doc_content = doc.page_content
        doc_tokens = _estimate_tokens(doc_content)
        
        # This check prevents Context Overflow (e.g. 8k limit)
        if current_tokens + doc_tokens + separator_tokens <= available_tokens:
            selected_docs.append(doc)
            current_tokens += doc_tokens + separator_tokens
            
            # Extract and clean source
            source = doc.metadata.get("source")
            if source and isinstance(source, str) and source.strip():
                selected_sources.add(source.strip())
        else:
            # Limit reached
            logger.warning(f"Stopping retrieval: Context limit reached ({current_tokens}/{available_tokens})")
            break 

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
    llm_config: Optional[Dict[str, Any]],
    truncated_history: Optional[List[Dict[str, str]]],
    estimated_context_tokens: int,
    hybrid: bool = False,
    formatted_relationships: Optional[List[str]] = None
) -> Dict[str, Union[str, List[str], int]]:
    """
    Dynamically builds the prompt using modular components and calls the LLM.
    """
    # 1. Validation: If no docs and no graph data, return early (unless hybrid might allow generic answer, 
    #    but typically RAG expects *some* context or a "no info" response).
    if not final_docs and not (rag_type == 'kag' and formatted_relationships):
        logger.warning(f"No documents/graph data for {rag_type}. Returning 'No Info'.")
        no_info_msg = (
            "No relevant information found in the knowledge graph." 
            if rag_type == 'kag' else 
            "No relevant information found in the database."
        )
        return {
            "text": no_info_msg, 
            "sources": [], 
            "estimated_context_tokens": 0
        }

    # 2. Prepare Context Text
    reordered_docs = _reorder_documents_for_context(final_docs)
    context_parts = []
    for doc in reordered_docs:
        # Extract filename for better context awareness
        src = doc.metadata.get("source", "unknown")
        # Clean path to be relative if possible
        if "/" in src or "\\" in src:
            src = src.split("/")[-1].split("\\")[-1]
            
        # XML wrapping allows the LLM to understand file boundaries
        context_parts.append(f"<document source='{src}'>\n{doc.page_content}\n</document>")
    
    context_text = "\n\n".join(context_parts)

    # 3. Determine Context Labels & KAG Specifics
    is_kag = (rag_type == 'kag')
    context_type_label = KAG_CONTEXT_TYPE if is_kag else STANDARD_CONTEXT_TYPE
    
    relationship_qualifier = KAG_RELATIONSHIP_QUALIFIER if is_kag else EMPTY_STRING
    relationship_qualifier_cite = KAG_RELATIONSHIP_QUALIFIER_CITE if is_kag else EMPTY_STRING

    # 4. Select Persona and Description
    # Logic: KAG > LightRAG > Default, modified by Hybrid flag
    if is_kag:
        assistant_persona = PERSONA_KNOWLEDGE_AWARE
        persona_description = DESC_KNOWLEDGE_AWARE
    elif rag_type == 'lightrag' and hybrid:
        assistant_persona = PERSONA_LIGHTWEIGHT
        persona_description = DESC_LIGHTWEIGHT
    elif hybrid:
        assistant_persona = PERSONA_COMPREHENSIVE
        persona_description = DESC_COMPREHENSIVE
    else:
        # Standard Strict RAG
        assistant_persona = PERSONA_PRECISE_ACCURATE
        persona_description = DESC_PRECISE_ACCURATE

    # 5. Select Instructions and Question Block
    if hybrid:
        instructions_base = HYBRID_INSTRUCTIONS
        kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID if is_kag else EMPTY_STRING
    else:
        instructions_base = STRICT_CONTEXT_INSTRUCTIONS
        kag_specific_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT if is_kag else EMPTY_STRING
    
    instructions = instructions_base.format(
        context_type=context_type_label,
        context_type_lower=context_type_label.lower(),
        relationship_qualifier=relationship_qualifier,
        relationship_qualifier_cite=relationship_qualifier_cite,
        kag_specific_instruction_placeholder=kag_specific_instruction
    )

    # 6. Assemble Prompt Blocks
    question_block = QUESTION_BLOCK.format(question=query_text)
    context_block = CONTEXT_BLOCK.format(context_type=context_type_label, context=context_text)
    sources_block = SOURCES_BLOCK.format(sources=sources)
    
    relationships_block = EMPTY_STRING
    if is_kag and formatted_relationships:
        rel_text = "\n\n".join(formatted_relationships)
        relationships_block = RELATIONSHIPS_BLOCK.format(relationships=rel_text)

    # 7. Final Prompt Assembly
    # Note: initial_answer_block_placeholder is explicitly empty as per modern design
    final_prompt_str = BASE_RESPONSE_STRUCTURE.format(
        assistant_persona=assistant_persona,
        persona_description=persona_description,
        initial_answer_block_placeholder=EMPTY_STRING, 
        context_block_placeholder=context_block,
        relationships_block_placeholder=relationships_block,
        sources_block_placeholder=sources_block,
        instructions=instructions,
        question_block_placeholder=question_block
    )

    # 8. Execute LLM Call
    response_text = get_llm_response(
        final_prompt_str, 
        llm_config=llm_config, 
        conversation_history=truncated_history
    )

    return {
        "text": response_text, 
        "sources": sources, 
        "estimated_context_tokens": estimated_context_tokens
    }