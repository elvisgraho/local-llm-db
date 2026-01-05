"""
Training module for document processing and LightRAG database population.

This module provides utilities for:
- Document loading and processing
- LLM-based metadata extraction and tagging
- Batch processing operations
- LightRAG database population with graph-enhanced retrieval
- Embedding generation
"""

from .get_embedding_function import get_embedding_function
from .llm_client import get_llm_client, get_llm_response
from .history_manager import ProcessingHistory
from .load_documents import load_documents
from .processing_utils import (
    split_document,
    initialize_chroma_vectorstore,
    validate_metadata,
    get_unique_path,
    calculate_context_ceiling
)

__all__ = [
    'get_embedding_function',
    'get_llm_client',
    'get_llm_response',
    'ProcessingHistory',
    'load_documents',
    'split_document',
    'initialize_chroma_vectorstore',
    'validate_metadata',
    'get_unique_path',
    'calculate_context_ceiling',
]
