import logging
import os
import hashlib
from typing import Any, Optional, List, Dict, Tuple, Union, Set

# Modern LangChain Core Imports
from langchain_core.documents import Document

# Internal Imports
from query.data_service import data_service
from query.llm_service import get_llm_response
from query.query_helpers import estimate_tokens, _reorder_documents_for_context
from query.templates import (
    RAG_SYSTEM_CONSTRUCTION, RAG_USER_TEMPLATE,
    STRICT_CONTEXT_INSTRUCTIONS, HYBRID_INSTRUCTIONS,
    CONTEXT_BLOCK, SOURCES_BLOCK, QUESTION_BLOCK,
    STANDARD_CONTEXT_TYPE
)

logger = logging.getLogger(__name__)

# --- Constants ---
DOC_SEPARATOR = "\n\n---\n\n"
MIN_DOC_LENGTH = 50           # Minimum characters for a chunk to be considered useful
HEADER_ONLY_THRESHOLD = 150   # Max length for a chunk starting with '#' to be considered a "header orphan"

def _rerank_results(
    query_text: str, 
    results: List[Tuple[Document, float]], 
    k: int
) -> List[Tuple[Document, float]]:
    """
    Reranks retrieval results using the CrossEncoder model.
    Includes robust error handling and input validation.
    """
    reranker = data_service.reranker
    
    # Fast exit conditions
    if not results:
        return []
    if not reranker:
        # Fallback to original vector/bm25 scores
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    try:
        logger.info(f"Reranking {len(results)} results.")
        
        # Prepare pairs: [Query, Document Content]
        # Validate content exists to prevent model crashes
        pairs = []
        valid_indices = []
        
        for i, (doc, _) in enumerate(results):
            content = doc.page_content
            if content and isinstance(content, str) and content.strip():
                pairs.append([query_text, content])
                valid_indices.append(i)
        
        if not pairs:
            logger.warning("No valid text content found for reranking.")
            return results[:k]

        # Predict scores
        rerank_scores = reranker.predict(pairs)
        
        # Reconstruct list with new scores
        reranked_results = []
        for idx, new_score in zip(valid_indices, rerank_scores):
            reranked_results.append((results[idx][0], float(new_score)))
        
        # Sort descending
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
    Selects documents to fit within token limits.
    
    OPTIMIZATIONS:
    1. Filters empty documents.
    2. Filters "Header-Only" chunks (common splitting artifact).
    3. Deduplicates identical content.
    4. Fallback mechanism to ensure at least one doc is returned if possible.
    """
    selected_docs = []
    selected_sources = set()
    current_tokens = 0
    separator_tokens = estimate_tokens(DOC_SEPARATOR)
    seen_content_hashes: Set[str] = set()

    logger.info(f"Selecting documents from {len(docs_with_scores)} candidates (Limit: {available_tokens} tokens).")

    for i, (doc, score) in enumerate(docs_with_scores):
        doc_content = doc.page_content
        
        # --- FILTER 1: Empty/None ---
        if not doc_content or not doc_content.strip():
            continue

        # --- FILTER 2: Header-Only Artifacts ---
        # Detects chunks that are just "## Title" created by aggressive splitting
        stripped_content = doc_content.strip()
        if stripped_content.startswith('#') and len(stripped_content) < HEADER_ONLY_THRESHOLD:
            # If it has very few newlines, it's likely just a header without body
            if stripped_content.count('\n') < 2:
                logger.debug(f"Skipping Header-Only Chunk: {stripped_content[:50]}...")
                continue
        
        # --- FILTER 3: Noise / Too Short ---
        if len(stripped_content) < MIN_DOC_LENGTH:
            logger.debug(f"Skipping Short Chunk ({len(stripped_content)} chars).")
            continue

        # --- FILTER 4: Content Deduplication ---
        # Hash the content to detect exact duplicates across different chunks/files
        content_hash = hashlib.md5(stripped_content.encode('utf-8')).hexdigest()
        if content_hash in seen_content_hashes:
            logger.debug(f"Skipping duplicate content (Doc {i}).")
            continue
        seen_content_hashes.add(content_hash)

        # --- TOKEN CHECK ---
        est_tokens = estimate_tokens(doc_content)
        
        # Check if adding this doc exceeds the budget
        if current_tokens + est_tokens + separator_tokens <= available_tokens:
            selected_docs.append(doc)
            current_tokens += est_tokens + separator_tokens
            
            # Extract clean source name
            source_raw = doc.metadata.get("source")
            if source_raw and isinstance(source_raw, str) and source_raw.strip():
                # Clean path to just filename
                filename = os.path.basename(source_raw)
                selected_sources.add(filename)
        else:
            logger.info(f"Context limit reached at doc {i}. Used {current_tokens}/{available_tokens} tokens.")
            break 

    # --- FALLBACK MECHANISM ---
    # If filters removed everything but we had candidates, force include the top result
    # to avoid "No relevant information" errors due to strict filtering.
    if not selected_docs and docs_with_scores:
        logger.warning("All documents were filtered out. Applying fallback to include Top-1.")
        top_doc = docs_with_scores[0][0]
        if top_doc.page_content and top_doc.page_content.strip():
            selected_docs.append(top_doc)
            current_tokens = estimate_tokens(top_doc.page_content)
            
            src = top_doc.metadata.get("source")
            if src: selected_sources.add(os.path.basename(src))

    if not selected_docs:
        logger.warning("No documents selected after filtering and fallback.")
        return [], [], 0

    unique_sources_list = sorted(list(selected_sources))
    logger.info(f"Final Selection: {len(selected_docs)} docs | ~{current_tokens} tokens | Sources: {len(unique_sources_list)}")
    
    return selected_docs, unique_sources_list, current_tokens

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
    
    # 2. Prepare Context Text (XML Style)
    reordered_docs = _reorder_documents_for_context(final_docs)
    context_parts = []
    
    for doc in reordered_docs:
        # Clean source for context tag
        raw_source = doc.metadata.get("source", "unknown")
        clean_source = os.path.basename(raw_source) if raw_source else "unknown"
        
        # XML wrapping helps models distinguish separate documents
        context_parts.append(f"<document source='{clean_source}'>\n{doc.page_content}\n</document>")
    
    context_text = "\n\n".join(context_parts)

    context_type_label = STANDARD_CONTEXT_TYPE
    
    # 4. Select Instructions
    if hybrid:
        instructions_base = HYBRID_INSTRUCTIONS
    else:
        instructions_base = STRICT_CONTEXT_INSTRUCTIONS
    
    rag_rules = instructions_base.format(
        context_type=context_type_label
    )

    # 5. CONSTRUCT SYSTEM PROMPT
    user_persona = llm_config.get('system_prompt', "You are a helpful AI assistant.")
    
    final_system_prompt = RAG_SYSTEM_CONSTRUCTION.format(
        user_defined_persona=user_persona,
        rag_instructions=rag_rules
    )

    # 6. CONSTRUCT USER PROMPT
    question_block = QUESTION_BLOCK.format(question=query_text)
    context_block = CONTEXT_BLOCK.format(context_type=context_type_label, context=context_text)
    # Convert list ['a.pdf', 'b.txt'] -> "a.pdf, b.txt"
    # This prevents brackets and quotes from confusing the model's source citation style.
    if isinstance(sources, list):
        sources_str = ", ".join(sources)
    else:
        sources_str = str(sources)

    sources_block = SOURCES_BLOCK.format(sources=sources_str)

    final_user_prompt = RAG_USER_TEMPLATE.format(
        context_block_placeholder=context_block,
        sources_block_placeholder=sources_block,
        question_block_placeholder=question_block
    )

    # 7. Execute LLM Call
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