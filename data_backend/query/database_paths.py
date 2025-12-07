import os
from pathlib import Path
from typing import Dict, List, Tuple

# --- Constants ---

# Valid RAG types used across the system
VALID_RAG_TYPES: Tuple[str, ...] = ("rag", "kag", "lightrag")
DEFAULT_DB_NAME = "default"

# --- Path Configuration ---
_env_db_path = os.getenv("RAG_DATABASE_DIR")
if _env_db_path:
    DATABASE_DIR = Path(_env_db_path)
else:
    # Assuming this file is in <project_root>/query/database_paths.py
    DATABASE_DIR = Path(__file__).resolve().parent.parent / "databases"

# Ensure the directory exists
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---
def _get_type_root_dir(rag_type: str) -> Path:
    """Internal helper to get the root directory for a specific RAG type."""
    if rag_type not in VALID_RAG_TYPES:
        raise ValueError(f"Invalid rag_type: '{rag_type}'. Must be one of: {VALID_RAG_TYPES}")
    return DATABASE_DIR / rag_type

def get_db_paths(rag_type: str, db_name: str) -> Dict[str, Path]:
    """
    Generates standard file paths for a specific RAG type and database instance.

    Args:
        rag_type (str): The type of RAG ('rag', 'kag', 'lightrag').
        db_name (str): The specific name of the database instance.

    Returns:
        Dict[str, Path]: Dictionary containing absolute paths.
        - 'db_dir': Root folder for this instance.
        - 'chroma_path': Path for ChromaDB persistence.
        - 'graph_path': Path for NetworkX JSON graph (KAG).
        - 'vectorstore_path': Legacy path (FAISS).
    """
    # 1. Get Base Directory
    base_dir = _get_type_root_dir(rag_type)
    db_instance_dir = base_dir / db_name

    # 2. Define Standard Paths
    # Note: We now prioritize 'chroma_path' for all types as the system converges on ChromaDB.
    paths = {
        "db_dir": db_instance_dir,
        
        # Standard ChromaDB path (used by RAG, LightRAG, and KAG's vector component)
        "chroma_path": db_instance_dir / "chroma",
        
        # Knowledge Graph file (Specific to KAG)
        "graph_path": db_instance_dir / "graph.json",
        
        # Legacy FAISS support (Older LightRAG implementations)
        "vectorstore_path": db_instance_dir / "vectorstore"
    }

    return paths

def list_available_dbs(rag_type: str) -> List[str]:
    """
    Lists available database names for a given RAG type.

    Args:
        rag_type (str): The type of RAG ('rag', 'kag', 'lightrag').

    Returns:
        List[str]: A sorted list of database names found on disk.
    """
    try:
        base_dir = _get_type_root_dir(rag_type)
    except ValueError:
        return []

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    # List subdirectories that are not hidden
    db_names = [
        d.name for d in base_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    
    return sorted(db_names)

def db_exists(rag_type: str, db_name: str) -> bool:
    """
    Checks if a specific database instance exists and contains data.
    
    Args:
        rag_type (str): The type of RAG.
        db_name (str): The database name.
        
    Returns:
        bool: True if key files/directories exist.
    """
    try:
        paths = get_db_paths(rag_type, db_name)
        db_dir = paths["db_dir"]
        
        if not db_dir.exists():
            return False

        # Check specific indicators based on type
        if rag_type == "kag":
            return paths["graph_path"].exists()
        
        # For RAG/LightRAG, check if Chroma exists
        # Chroma usually creates a 'chroma.sqlite3' file inside the directory
        chroma_db_file = paths["chroma_path"] / "chroma.sqlite3"
        return chroma_db_file.exists() or paths["vectorstore_path"].exists()

    except Exception:
        return False