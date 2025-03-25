import os

# Base directory for all databases
DATABASE_DIR = "databases"

# RAG database paths
RAG_DB_DIR = os.path.join(DATABASE_DIR, "rag")
CHROMA_PATH = os.path.join(RAG_DB_DIR, "chroma")

# GraphRAG database paths
GRAPHRAG_DB_DIR = os.path.join(DATABASE_DIR, "graphrag")
GRAPHRAG_GRAPH_PATH = os.path.join(GRAPHRAG_DB_DIR, "graph.json")

# KAG database paths
KAG_DB_DIR = os.path.join(DATABASE_DIR, "kag")
KAG_GRAPH_PATH = os.path.join(KAG_DB_DIR, "graph.json")

# Light RAG database paths
LIGHT_RAG_DB_DIR = os.path.join(DATABASE_DIR, "light_rag")
VECTORSTORE_PATH = os.path.join(LIGHT_RAG_DB_DIR, "vectorstore")

# Create all necessary directories
for path in [RAG_DB_DIR, GRAPHRAG_DB_DIR, KAG_DB_DIR, LIGHT_RAG_DB_DIR]:
    os.makedirs(path, exist_ok=True) 