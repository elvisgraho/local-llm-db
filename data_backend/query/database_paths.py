import os
from pathlib import Path
from typing import Dict, List, Tuple

# --- Constants ---

VALID_RAG_TYPES: Tuple[str, ...] = ("rag", "kag", "lightrag")
DEFAULT_DB_NAME = "default"

# --- Root Path Determination ---

# This assumes the file structure: [ROOT]/data_backend/query/database_paths.py
# .parent = query
# .parent.parent = data_backend
# .parent.parent.parent = PROJECT_ROOT
# Note: Adjust .parent chain length if your file is nested differently.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Directory Configuration ---

# 1. Volumes Base Directory
# Default: [ROOT]/volumes
# Docker/Env Override: env("VOLUMES_DIR")
_env_volumes = os.getenv("VOLUMES_DIR")
if _env_volumes:
    VOLUMES_DIR = Path(_env_volumes)
else:
    VOLUMES_DIR = PROJECT_ROOT / "volumes"

# 2. Raw Files Directory
# Default: [ROOT]/volumes/raw_files
# Env Override: env("RAW_FILES_DIR")
_env_raw = os.getenv("RAW_FILES_DIR")
if _env_raw:
    RAW_FILES_DIR = Path(_env_raw)
else:
    RAW_FILES_DIR = VOLUMES_DIR / "raw_files"

# 3. Databases Directory
# Default: [ROOT]/volumes/databases
_env_db = os.getenv("RAG_DATABASE_DIR")
if _env_db:
    DATABASE_DIR = Path(_env_db)
else:
    DATABASE_DIR = VOLUMES_DIR / "databases"

# --- Initialization ---

def initialize_directories():
    """Ensures that the necessary directories exist."""
    for path in [DATABASE_DIR, RAW_FILES_DIR]:
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {path}: {e}")

# Run initialization immediately on import
initialize_directories()

# --- Helper Functions ---

def _get_type_root_dir(rag_type: str) -> Path:
    """Internal helper to get the root directory for a specific RAG type."""
    if rag_type not in VALID_RAG_TYPES:
        raise ValueError(f"Invalid rag_type: '{rag_type}'. Must be one of: {VALID_RAG_TYPES}")
    return DATABASE_DIR / rag_type

def get_db_paths(rag_type: str, db_name: str) -> Dict[str, Path]:
    """
    Generates standard file paths for a specific RAG type and database instance.
    """
    base_dir = _get_type_root_dir(rag_type)
    db_instance_dir = base_dir / db_name

    paths = {
        "db_dir": db_instance_dir,
        "chroma_path": db_instance_dir / "chroma",
        "graph_path": db_instance_dir / "graph.json",
        "vectorstore_path": db_instance_dir / "vectorstore"
    }
    return paths

def list_available_dbs(rag_type: str) -> List[str]:
    """Lists available database names for a given RAG type."""
    try:
        base_dir = _get_type_root_dir(rag_type)
    except ValueError:
        return []

    if not base_dir.exists() or not base_dir.is_dir():
        return []

    db_names = [
        d.name for d in base_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    return sorted(db_names)

def db_exists(rag_type: str, db_name: str) -> bool:
    """Checks if a specific database instance exists and contains data."""
    try:
        paths = get_db_paths(rag_type, db_name)
        db_dir = paths["db_dir"]
        
        if not db_dir.exists():
            return False

        if rag_type == "kag":
            return paths["graph_path"].exists()
        
        # Check for ChromaDB file or legacy vectorstore
        chroma_file = paths["chroma_path"] / "chroma.sqlite3"
        return chroma_file.exists() or paths["vectorstore_path"].exists()

    except Exception:
        return False