import os
from pathlib import Path

# Get the training directory
TRAINING_DIR = Path(__file__).parent.parent / "training"

# Base directory for all databases
DATABASE_DIR = TRAINING_DIR / "databases"

# RAG database paths
RAG_DB_DIR = DATABASE_DIR / "rag"
CHROMA_PATH = RAG_DB_DIR / "chroma"

# GraphRAG database paths
GRAPHRAG_DB_DIR = DATABASE_DIR / "graphrag"
GRAPHRAG_GRAPH_PATH = GRAPHRAG_DB_DIR / "graph.json"

# KAG database paths
KAG_DB_DIR = DATABASE_DIR / "kag"
KAG_GRAPH_PATH = KAG_DB_DIR / "graph.json"

# Light RAG database paths
LIGHT_RAG_DB_DIR = DATABASE_DIR / "light_rag"
VECTORSTORE_PATH = LIGHT_RAG_DB_DIR / "vectorstore"

# Ensure all database directories exist
for directory in [RAG_DB_DIR, GRAPHRAG_DB_DIR, KAG_DB_DIR, LIGHT_RAG_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True) 