"""
Configuration management for the query system.

This module centralizes all configuration settings for the query system,
including model parameters, API endpoints, and system paths.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class LLMConfig:
    """Configuration for LLM-related settings."""
    model_name: Optional[str] = None
    ocr_model_name: Optional[str] = None
    api_url: Optional[str] = None
    temperature: float = 0.7 # Keep other defaults for now
    max_retries: int = 3
    retry_wait_min: int = 1
    retry_wait_max: int = 10

@dataclass
class RAGConfig:
    """Configuration for RAG-related settings."""
    similarity_threshold: float = 0.5
    max_documents: int = 15
    chunk_size: int = 512
    chunk_overlap: int = 100
    # Metadata-enhanced RAG features (automatically enabled)
    use_metadata_boost: bool = True  # Enable metadata-aware reranking
    use_metadata_augmentation: bool = True  # Prepend metadata to context
    metadata_boost_alpha: float = 0.7  # 70% original score, 30% metadata boost
    diversity_weight: float = 0.25  # Weight for diversity in result selection


@dataclass
class SystemConfig:
    """Main system configuration."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create a configuration from environment variables.
        
        Returns:
            SystemConfig: The configuration object.
        """
        # Define fallback defaults here if environment variables are not set
        default_model_name = "gpt-oss-20b-derestricted"
        default_ocr_model_name = "gliese-ocr-7b-post2.0-final-i1"
        default_api_url = "http://10.2.0.2:1234"

        return cls(
            llm=LLMConfig(
                # Use explicit string defaults in getenv if env var is missing
                model_name=os.getenv('LLM_MODEL_NAME', default_model_name),
                ocr_model_name=os.getenv('LLM_OCR_MODEL_NAME', default_ocr_model_name),
                api_url=os.getenv('LLM_API_URL', default_api_url),
                temperature=float(os.getenv('LLM_TEMPERATURE', LLMConfig.temperature)),
                max_retries=int(os.getenv('LLM_MAX_RETRIES', LLMConfig.max_retries)),
                retry_wait_min=int(os.getenv('LLM_RETRY_WAIT_MIN', LLMConfig.retry_wait_min)),
                retry_wait_max=int(os.getenv('LLM_RETRY_WAIT_MAX', LLMConfig.retry_wait_max))
            ),
            rag=RAGConfig(
                similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', RAGConfig.similarity_threshold)),
                max_documents=int(os.getenv('RAG_MAX_DOCUMENTS', RAGConfig.max_documents)),
                chunk_size=int(os.getenv('RAG_CHUNK_SIZE', RAGConfig.chunk_size)),
                chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', RAGConfig.chunk_overlap)),
                use_metadata_boost=os.getenv('RAG_USE_METADATA_BOOST', 'true').lower() == 'true',
                use_metadata_augmentation=os.getenv('RAG_USE_METADATA_AUGMENTATION', 'true').lower() == 'true',
                metadata_boost_alpha=float(os.getenv('RAG_METADATA_BOOST_ALPHA', RAGConfig.metadata_boost_alpha)),
                diversity_weight=float(os.getenv('RAG_DIVERSITY_WEIGHT', RAGConfig.diversity_weight))
            )
        )

# Create a global configuration instance
config = SystemConfig.from_env() 