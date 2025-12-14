import logging
from typing import Any, Optional, List, Dict, Tuple, Union
from langchain_core.documents import Document
from query.data_service import data_service
from query.llm_service import get_llm_response
from query.query_helpers import _estimate_tokens, _reorder_documents_for_context

# Updated Imports
from query.templates import (
    RAG_SYSTEM_CONSTRUCTION, RAG_USER_TEMPLATE,
    STRICT_CONTEXT_INSTRUCTIONS, HYBRID_INSTRUCTIONS,
    CONTEXT_BLOCK, SOURCES_BLOCK, RELATIONSHIPS_BLOCK, QUESTION_BLOCK,
    KAG_CONTEXT_TYPE, STANDARD_CONTEXT_TYPE,
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

def select_docs_for_context(
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
    verify: Optional[bool] = False,
    formatted_relationships: Optional[List[str]] = None
) -> Dict[str, Union[str, List[str], int]]:
    
    # 1. Validation
    if not final_docs and not (rag_type == 'kag' and formatted_relationships):
        no_info_msg = "No relevant information found in the database."
        return {"text": no_info_msg, "sources": [], "estimated_context_tokens": 0}

    # 2. Prepare Context Text
    reordered_docs = _reorder_documents_for_context(final_docs)
    context_parts = []
    for doc in reordered_docs:
        src = doc.metadata.get("source", "unknown")
        if "/" in src or "\\" in src: src = src.split("/")[-1].split("\\")[-1]
        context_parts.append(f"<document source='{src}'>\n{doc.page_content}\n</document>")
    
    context_text = "\n\n".join(context_parts)

    # 3. Determine Context Labels
    is_kag = (rag_type == 'kag')
    context_type_label = KAG_CONTEXT_TYPE if is_kag else STANDARD_CONTEXT_TYPE
    
    # 4. Select Instructions based on Mode
    # We no longer select a 'Persona'. We only select 'Rules'.
    if hybrid:
        instructions_base = HYBRID_INSTRUCTIONS
        kag_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_HYBRID if is_kag else EMPTY_STRING
    else:
        instructions_base = STRICT_CONTEXT_INSTRUCTIONS
        kag_instruction = KAG_SPECIFIC_DETAIL_INSTRUCTION_STRICT if is_kag else EMPTY_STRING
    
    rag_rules = instructions_base.format(
        context_type=context_type_label,
        kag_specific_instruction_placeholder=kag_instruction
    )

    # 5. CONSTRUCT SYSTEM PROMPT (User Persona + RAG Rules)
    # Extract the user's custom prompt from the config
    user_persona = llm_config.get('system_prompt', "You are a helpful AI assistant.")
    
    final_system_prompt = RAG_SYSTEM_CONSTRUCTION.format(
        user_defined_persona=user_persona,
        rag_instructions=rag_rules
    )

    # 6. CONSTRUCT USER PROMPT (Data + Question)
    question_block = QUESTION_BLOCK.format(question=query_text)
    context_block = CONTEXT_BLOCK.format(context_type=context_type_label, context=context_text)
    sources_block = SOURCES_BLOCK.format(sources=sources)
    
    relationships_block = EMPTY_STRING
    if is_kag and formatted_relationships:
        rel_text = "\n\n".join(formatted_relationships)
        relationships_block = RELATIONSHIPS_BLOCK.format(relationships=rel_text)

    final_user_prompt = RAG_USER_TEMPLATE.format(
        context_block_placeholder=context_block,
        relationships_block_placeholder=relationships_block,
        sources_block_placeholder=sources_block,
        question_block_placeholder=question_block
    )

    # 7. Execute
    # We update the config with the COMBINED system prompt
    run_config = llm_config.copy() if llm_config else {}
    run_config['system_prompt'] = final_system_prompt

    response_text = get_llm_response(
        prompt=final_user_prompt, 
        llm_config=run_config, 
        conversation_history=truncated_history,
        verify=verify
    )

    return {
        "text": response_text, 
        "sources": sources, 
        "estimated_context_tokens": estimated_context_tokens
    }