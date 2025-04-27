"""
Global variables for the application.
"""

from query.config import config

# Local LLM model configuration
LOCAL_MAIN_MODEL = config.llm.model_name

# API configuration
LOCAL_LLM_API_URL = config.llm.api_url

# RAG configuration
RAG_SIMILARITY_THRESHOLD = config.rag.similarity_threshold
RAG_MAX_DOCUMENTS = config.rag.max_documents

# Graph configuration (Used by KAG)
GRAPH_MAX_DEPTH = config.graph.max_depth
GRAPH_MAX_NODES = config.graph.max_nodes
GRAPH_MIN_SIMILARITY = config.graph.min_similarity