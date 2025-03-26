"""
Configuration management for the query system.

This module centralizes all configuration settings for the query system,
including model parameters, API endpoints, and system paths.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class LLMConfig:
    """Configuration for LLM-related settings."""
    model_name: str = "claude-3.7-sonnet-reasoning-gemma3-12b"
    api_url: str = "http://localhost:1234/v1/chat/completions"
    temperature: float = 0.7
    max_retries: int = 3
    retry_wait_min: int = 1
    retry_wait_max: int = 10

@dataclass
class RAGConfig:
    """Configuration for RAG-related settings."""
    similarity_threshold: float = 0.5
    max_documents: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200

@dataclass
class GraphConfig:
    """Configuration for graph-related settings."""
    max_depth: int = 2
    max_nodes: int = 5
    min_similarity: float = 0.5

@dataclass
class SystemConfig:
    """Main system configuration."""
    llm: LLMConfig = LLMConfig()
    rag: RAGConfig = RAGConfig()
    graph: GraphConfig = GraphConfig()
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Create a configuration from environment variables.
        
        Returns:
            SystemConfig: The configuration object.
        """
        return cls(
            llm=LLMConfig(
                model_name=os.getenv('LLM_MODEL_NAME', LLMConfig.model_name),
                api_url=os.getenv('LLM_API_URL', LLMConfig.api_url),
                temperature=float(os.getenv('LLM_TEMPERATURE', LLMConfig.temperature)),
                max_retries=int(os.getenv('LLM_MAX_RETRIES', LLMConfig.max_retries)),
                retry_wait_min=int(os.getenv('LLM_RETRY_WAIT_MIN', LLMConfig.retry_wait_min)),
                retry_wait_max=int(os.getenv('LLM_RETRY_WAIT_MAX', LLMConfig.retry_wait_max))
            ),
            rag=RAGConfig(
                similarity_threshold=float(os.getenv('RAG_SIMILARITY_THRESHOLD', RAGConfig.similarity_threshold)),
                max_documents=int(os.getenv('RAG_MAX_DOCUMENTS', RAGConfig.max_documents)),
                chunk_size=int(os.getenv('RAG_CHUNK_SIZE', RAGConfig.chunk_size)),
                chunk_overlap=int(os.getenv('RAG_CHUNK_OVERLAP', RAGConfig.chunk_overlap))
            ),
            graph=GraphConfig(
                max_depth=int(os.getenv('GRAPH_MAX_DEPTH', GraphConfig.max_depth)),
                max_nodes=int(os.getenv('GRAPH_MAX_NODES', GraphConfig.max_nodes)),
                min_similarity=float(os.getenv('GRAPH_MIN_SIMILARITY', GraphConfig.min_similarity))
            )
        )

# Create a global configuration instance
config = SystemConfig.from_env() 