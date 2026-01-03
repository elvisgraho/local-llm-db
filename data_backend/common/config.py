"""
Unified configuration management for data_backend.

This module provides a centralized configuration system that works across
all deployment modes and can be imported by both backend and frontend.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """Configuration for LLM-related settings."""
    model_name: str = "gpt-oss-20b-derestricted"
    ocr_model_name: str = "gliese-ocr-7b-post2.0-final-i1"
    api_url: str = "http://10.2.0.2:1234"
    temperature: float = 0.7
    max_retries: int = 3
    retry_wait_min: int = 1
    retry_wait_max: int = 10

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create LLM configuration from environment variables."""
        return cls(
            model_name=os.getenv('LOCAL_MAIN_MODEL', cls.model_name),
            ocr_model_name=os.getenv('LOCAL_OCR_MODEL', cls.ocr_model_name),
            api_url=os.getenv('LOCAL_LLM_API_URL', cls.api_url),
            temperature=float(os.getenv('LLM_TEMPERATURE', cls.temperature)),
            max_retries=int(os.getenv('LLM_MAX_RETRIES', cls.max_retries)),
            retry_wait_min=int(os.getenv('LLM_RETRY_WAIT_MIN', cls.retry_wait_min)),
            retry_wait_max=int(os.getenv('LLM_RETRY_WAIT_MAX', cls.retry_wait_max))
        )


@dataclass
class RAGConfig:
    """Configuration for RAG-related settings."""
    similarity_threshold: float = 0.5
    max_documents: int = 15
    chunk_size: int = 512
    chunk_overlap: int = 100
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"

    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create RAG configuration from environment variables."""
        return cls(
            similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', cls.similarity_threshold)),
            max_documents=int(os.getenv('RAG_MAX_DOCUMENTS', cls.max_documents)),
            chunk_size=int(os.getenv('RAG_CHUNK_SIZE', cls.chunk_size)),
            chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', cls.chunk_overlap)),
            embedding_model=os.getenv('EMBEDDING_MODEL_NAME', cls.embedding_model)
        )


@dataclass
class Config:
    """Main system configuration that combines all sub-configurations."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)

    @classmethod
    def from_env(cls) -> 'Config':
        """Create complete configuration from environment variables."""
        return cls(
            llm=LLMConfig.from_env(),
            rag=RAGConfig.from_env()
        )

    def to_env_dict(self) -> dict:
        """Convert configuration to environment variable dictionary."""
        return {
            'LOCAL_MAIN_MODEL': self.llm.model_name,
            'LOCAL_OCR_MODEL': self.llm.ocr_model_name,
            'LOCAL_LLM_API_URL': self.llm.api_url,
            'LLM_TEMPERATURE': str(self.llm.temperature),
            'LLM_MAX_RETRIES': str(self.llm.max_retries),
            'LLM_RETRY_WAIT_MIN': str(self.llm.retry_wait_min),
            'LLM_RETRY_WAIT_MAX': str(self.llm.retry_wait_max),
            'RAG_SIMILARITY_THRESHOLD': str(self.rag.similarity_threshold),
            'RAG_MAX_DOCUMENTS': str(self.rag.max_documents),
            'RAG_CHUNK_SIZE': str(self.rag.chunk_size),
            'RAG_CHUNK_OVERLAP': str(self.rag.chunk_overlap),
            'EMBEDDING_MODEL_NAME': self.rag.embedding_model
        }


# Global configuration instance
config = Config.from_env()
