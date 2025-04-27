import os
from pathlib import Path
from typing import Dict

# Get the training directory
TRAINING_DIR = Path(__file__).parent.parent / "training"

# Base directory for all databases
DATABASE_DIR = TRAINING_DIR / "databases"
DEFAULT_DB_NAME = "default" # Default name for a database within a type

# --- Root Directories for RAG Types ---
RAG_ROOT_DIR = DATABASE_DIR / "rag"
KAG_ROOT_DIR = DATABASE_DIR / "kag"
LIGHTRAG_ROOT_DIR = DATABASE_DIR / "lightrag"

# --- Dynamic Path Generation ---

def get_db_paths(rag_type: str, db_name: str) -> Dict[str, Path]:
    """
    Generates database paths for a given RAG type and database name.

    Args:
        rag_type (str): The type of RAG ('rag', 'kag', 'lightrag').
        db_name (str): The specific name of the database instance (e.g., 'my_docs', 'project_data').

    Returns:
        Dict[str, Path]: A dictionary containing relevant paths for the specific DB instance.
                         Keys might include 'db_dir', 'graph_path', 'vectorstore_path', 'chroma_path'.

    Raises:
        ValueError: If the rag_type is invalid.
    """
    if rag_type == "rag":
        base_dir = RAG_ROOT_DIR
    elif rag_type == "kag":
        base_dir = KAG_ROOT_DIR
    elif rag_type == "lightrag":
        base_dir = LIGHTRAG_ROOT_DIR
    else:
        raise ValueError(f"Invalid rag_type: {rag_type}. Must be 'rag', 'kag', or 'lightrag'.")

    db_dir = base_dir / db_name
    paths = {
        "db_dir": db_dir,
        # Specific paths commonly used by different RAG types
        "graph_path": db_dir / "graph.json",        # For KAG
        "vectorstore_path": db_dir / "vectorstore", # For LightRAG (FAISS, etc.)
        "chroma_path": db_dir / "chroma"            # For RAG (ChromaDB)
    }
    # Ensure the base directory for this instance exists
    # db_dir.mkdir(parents=True, exist_ok=True) # Creation should happen during population, not lookup
    return paths

def list_available_dbs(rag_type: str) -> list[str]:
    """
    Lists available database names for a given RAG type.

    Args:
        rag_type (str): The type of RAG ('rag', 'kag', 'lightrag').

    Returns:
        list[str]: A list of database names (subdirectories). Returns an empty list if the
                   root directory for the rag_type doesn't exist or is empty.
    """
    if rag_type == "rag":
        base_dir = RAG_ROOT_DIR
    elif rag_type == "kag":
        base_dir = KAG_ROOT_DIR
    elif rag_type == "lightrag":
        base_dir = LIGHTRAG_ROOT_DIR
    else:
        return [] # Invalid type

    if not base_dir.is_dir():
        return []

    db_names = [d.name for d in base_dir.iterdir() if d.is_dir()]
    return db_names

# Ensure root database type directories exist (optional, depends on usage)
# for directory in [RAG_ROOT_DIR, KAG_ROOT_DIR, LIGHTRAG_ROOT_DIR]:
#     directory.mkdir(parents=True, exist_ok=True)