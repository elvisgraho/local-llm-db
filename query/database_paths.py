import os
from pathlib import Path
from typing import Dict

# Get the training directory
TRAINING_DIR = Path(__file__).parent.parent / "training"

# Base directory for all databases
DATABASE_DIR = TRAINING_DIR / "databases"

# --- Default Paths (kept for potential backward compatibility or reference) ---
DEFAULT_RAG_DB_DIR = DATABASE_DIR / "rag"
DEFAULT_CHROMA_PATH = DEFAULT_RAG_DB_DIR / "chroma"

DEFAULT_KAG_DB_DIR = DATABASE_DIR / "kag"
DEFAULT_KAG_GRAPH_PATH = DEFAULT_KAG_DB_DIR / "graph.json"

DEFAULT_LIGHT_RAG_DB_DIR = DATABASE_DIR / "light_rag"
DEFAULT_VECTORSTORE_PATH = DEFAULT_LIGHT_RAG_DB_DIR / "vectorstore" # Assuming vectorstore is common or primarily for light_rag/graphrag

# --- Dynamic Path Generation ---

def get_db_paths(db_name: str) -> Dict[str, Path]:
    """
    Generates database paths for a given database name.

    Args:
        db_name (str): The name of the database instance (e.g., 'graphrag_dev', 'my_knowledge_base').

    Returns:
        Dict[str, Path]: A dictionary containing relevant paths.
                         Keys might include 'db_dir', 'graph_path', 'vectorstore_path', 'chroma_path'.
    """
    db_dir = DATABASE_DIR / db_name
    paths = {
        "db_dir": db_dir,
        # Specific paths commonly used by different RAG types
        "graph_path": db_dir / "graph.json",        # For graph-based RAGs
        "vectorstore_path": db_dir / "vectorstore", # For vectorstore-based RAGs (FAISS, etc.)
        "chroma_path": db_dir / "chroma"            # For ChromaDB
    }
    # Ensure the base directory for this instance exists
    db_dir.mkdir(parents=True, exist_ok=True)
    return paths

# Ensure default database directories exist (optional, depends on usage)
# for directory in [DEFAULT_RAG_DB_DIR, DEFAULT_GRAPHRAG_DB_DIR, DEFAULT_KAG_DB_DIR, DEFAULT_LIGHT_RAG_DB_DIR]:
#     directory.mkdir(parents=True, exist_ok=True)