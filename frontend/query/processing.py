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
from query.metadata_enhancement import (
    rerank_with_metadata_awareness,
    augment_context_with_metadata,
    select_diverse_results
)
from query.config import config

# Lazy import for intent detection
_query_intent_detector = None

def get_query_intent_detector():
    """Lazy import to avoid circular dependencies."""
    global _query_intent_detector
    if _query_intent_detector is None:
        from query.query_intent_detector import detect_query_intent
        _query_intent_detector = detect_query_intent
    return _query_intent_detector

logger = logging.getLogger(__name__)

# --- Constants ---
DOC_SEPARATOR = "\n\n---\n\n"
MIN_DOC_LENGTH = 50           # Minimum characters for a chunk to be considered useful
HEADER_ONLY_THRESHOLD = 150   # Max length for a chunk starting with '#' to be considered a "header orphan"

def expand_chunks_with_context(
    ranked_docs: List[Tuple[Document, float]],
    rag_type: str,
    db_name: str,
    expansion_window: int = 1,
    min_score_threshold: float = 0.5,
    code_only: bool = True
) -> List[Tuple[Document, float]]:
    """
    Expand high-scoring chunks by retrieving adjacent chunks from the same document.

    This provides complete context for code-heavy content by fetching chunks before/after
    the matched chunk from the same source file.

    Args:
        ranked_docs: List of (Document, score) tuples already reranked
        rag_type: 'rag' or 'lightrag' (for database lookup)
        db_name: Database name (for database lookup)
        expansion_window: How many chunks before/after to retrieve (1 = ±1, 2 = ±2)
        min_score_threshold: Only expand chunks scoring above this threshold
        code_only: If True, only expand chunks with code_languages metadata

    Returns:
        Expanded list of (Document, score) with adjacent chunks inserted

    Research basis:
    - Parent Document Retrieval (LangChain pattern)
    - Contextual Chunk Expansion for code understanding
    """
    if not ranked_docs or expansion_window <= 0:
        return ranked_docs

    # Get ChromaDB instance
    db = data_service.get_chroma_db(rag_type, db_name)
    if not db:
        logger.warning(f"ChromaDB not available for chunk expansion ({rag_type}/{db_name})")
        return ranked_docs

    expanded_results = []
    seen_chunk_ids = set()  # Track chunks we've already included
    expansion_count = 0

    for doc, score in ranked_docs:
        # Add the original chunk
        chunk_id = id(doc)  # Use object id as unique identifier
        if chunk_id in seen_chunk_ids:
            continue

        expanded_results.append((doc, score))
        seen_chunk_ids.add(chunk_id)

        # Check if this chunk qualifies for expansion
        metadata = doc.metadata

        # Skip if score too low
        if score < min_score_threshold:
            continue

        # Skip if code_only=True and no code_languages metadata
        if code_only:
            code_langs = metadata.get('code_languages', '')
            if not code_langs or (isinstance(code_langs, str) and not code_langs.strip()):
                continue

        # Extract chunk position info
        chunk_index = metadata.get('chunk_index')
        total_chunks = metadata.get('total_chunks')
        source = metadata.get('source')

        if chunk_index is None or total_chunks is None or not source:
            continue  # Missing required metadata for expansion

        # Calculate adjacent chunk indices to retrieve
        adjacent_indices = []
        for offset in range(-expansion_window, expansion_window + 1):
            if offset == 0:
                continue  # Skip current chunk (already added)

            target_index = chunk_index + offset

            # Boundary checks
            if 0 <= target_index < total_chunks:
                adjacent_indices.append(target_index)

        if not adjacent_indices:
            continue  # No adjacent chunks to retrieve

        # Query ChromaDB for adjacent chunks
        try:
            # Build metadata filter for adjacent chunks from same source
            for target_index in sorted(adjacent_indices):
                # Query with metadata filter
                adjacent_results = db.get(
                    where={
                        "$and": [
                            {"source": {"$eq": source}},
                            {"chunk_index": {"$eq": target_index}}
                        ]
                    },
                    include=["documents", "metadatas"]
                )

                if not adjacent_results or not adjacent_results.get("ids"):
                    continue

                # Extract first matching chunk
                adj_ids = adjacent_results["ids"]
                adj_docs = adjacent_results.get("documents", [])
                adj_metas = adjacent_results.get("metadatas", [])

                if not adj_docs:
                    continue

                # Create Document object
                adj_doc = Document(
                    page_content=adj_docs[0],
                    metadata=adj_metas[0] if adj_metas else {}
                )

                adj_chunk_id = id(adj_doc)
                if adj_chunk_id in seen_chunk_ids:
                    continue

                # Assign inherited score (80% of parent chunk's score)
                inherited_score = score * 0.8

                # Insert adjacent chunk near parent (ordered by chunk_index)
                if target_index < chunk_index:
                    # Insert before current chunk (we'll need to reorder later)
                    expanded_results.insert(-1, (adj_doc, inherited_score))
                else:
                    # Insert after current chunk
                    expanded_results.append((adj_doc, inherited_score))

                seen_chunk_ids.add(adj_chunk_id)
                expansion_count += 1

        except Exception as e:
            logger.debug(f"Error retrieving adjacent chunks for {source} chunk {chunk_index}: {e}")
            continue

    if expansion_count > 0:
        logger.info(f"Chunk expansion: Added {expansion_count} adjacent chunks (window=±{expansion_window})")

    return expanded_results


def _rerank_results(
    query_text: str,
    results: List[Tuple[Document, float]],
    k: int,
    use_metadata_boost: Optional[bool] = None,
    rag_type: str = 'rag',
    db_name: str = 'default'
) -> List[Tuple[Document, float]]:
    """
    Reranks retrieval results using CrossEncoder + metadata awareness + chunk expansion.

    Four-stage process:
    1. CrossEncoder semantic reranking (if available)
    2. Metadata-aware boosting (MITRE tactics, code languages, topics)
    3. Contextual chunk expansion (retrieve adjacent chunks for code-heavy content)
    4. Diversity-based selection (MMR with metadata)

    Includes robust error handling and input validation.
    """
    # Use config value if not explicitly set
    if use_metadata_boost is None:
        use_metadata_boost = config.rag.use_metadata_boost

    reranker = data_service.reranker

    # Fast exit conditions
    if not results:
        return []

    # Stage 1: CrossEncoder Reranking
    crossencoder_results = results
    if reranker:
        try:
            logger.info(f"Stage 1: CrossEncoder reranking {len(results)} results.")

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
                logger.warning("No valid text content found for CrossEncoder reranking.")
            else:
                # Predict scores
                rerank_scores = reranker.predict(pairs)

                # Reconstruct list with new scores
                crossencoder_results = []
                for idx, new_score in zip(valid_indices, rerank_scores):
                    crossencoder_results.append((results[idx][0], float(new_score)))

                # Sort descending
                crossencoder_results.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"CrossEncoder reranking complete. Top score: {crossencoder_results[0][1]:.3f}")

        except Exception as e:
            logger.error(f"Error during CrossEncoder reranking: {e}. Using original scores.", exc_info=True)
            crossencoder_results = results
    else:
        # No CrossEncoder available, sort by original scores
        crossencoder_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Stage 2: Metadata-Aware Boosting (with intent detection)
    if use_metadata_boost and crossencoder_results:
        try:
            # Detect query intent for MITRE-aware boosting
            query_intent = None
            try:
                detect_fn = get_query_intent_detector()
                query_intent = detect_fn(query_text)
            except Exception as e:
                logger.debug(f"Intent detection skipped: {e}")

            logger.info("Stage 2: Metadata-aware boosting" + (" with intent" if query_intent else "") + ".")

            # Apply metadata boosting using config alpha value
            final_results = rerank_with_metadata_awareness(
                query_text,
                crossencoder_results,
                k=k * 2,  # Get 2x for diversity selection + chunk expansion
                alpha=config.rag.metadata_boost_alpha,
                query_intent=query_intent
            )

            # Stage 3: Contextual Chunk Expansion (for code-heavy content)
            if config.rag.enable_chunk_expansion:
                try:
                    logger.info("Stage 3: Contextual chunk expansion.")
                    final_results = expand_chunks_with_context(
                        final_results,
                        rag_type=rag_type,
                        db_name=db_name,
                        expansion_window=config.rag.chunk_expansion_window,
                        min_score_threshold=config.rag.chunk_expansion_min_score,
                        code_only=config.rag.chunk_expansion_for_code_only
                    )
                except Exception as e:
                    logger.error(f"Error during chunk expansion: {e}. Skipping expansion.", exc_info=True)

            # Stage 4: Diversity Selection
            logger.info("Stage 4: Diversity-based selection.")
            final_results = select_diverse_results(
                final_results,
                k=k,
                diversity_weight=config.rag.diversity_weight
            )

            return final_results

        except Exception as e:
            logger.error(f"Error during metadata enhancement: {e}. Using CrossEncoder results.", exc_info=True)
            return crossencoder_results[:k]

    return crossencoder_results[:k]

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
    formatted_relationships: Optional[List[str]] = None,
    use_metadata_augmentation: Optional[bool] = None
) -> Dict[str, Union[str, List[str], int]]:

    # Use config value if not explicitly set
    if use_metadata_augmentation is None:
        use_metadata_augmentation = config.rag.use_metadata_augmentation

    # 2. Prepare Context Text (XML Style with Metadata Augmentation)
    reordered_docs = _reorder_documents_for_context(final_docs)
    context_parts = []

    for doc in reordered_docs:
        # Clean source for context tag
        raw_source = doc.metadata.get("source", "unknown")
        clean_source = os.path.basename(raw_source) if raw_source else "unknown"

        # Optionally augment content with metadata
        if use_metadata_augmentation:
            augmented_content = augment_context_with_metadata(doc)
        else:
            augmented_content = doc.page_content

        # XML wrapping helps models distinguish separate documents
        context_parts.append(f"<document source='{clean_source}'>\n{augmented_content}\n</document>")

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