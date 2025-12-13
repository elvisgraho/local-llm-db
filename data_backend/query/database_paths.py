import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- Constants ---

VALID_RAG_TYPES: Tuple[str, ...] = ("rag", "kag", "lightrag")
DEFAULT_DB_NAME = "default"

# --- Adaptive Root Determination ---
_current_dir = Path(__file__).resolve().parent # [root]/query/
_backend_root = _current_dir.parent        # [root]/ or [root]/data_backend

# Logic: Find where 'volumes' lives.
# 1. Docker/Production: Look inside the backend root (e.g., /app/volumes)
if (_backend_root / "volumes").exists():
    PROJECT_ROOT = _backend_root
# 2. Local Dev: Look one level up (e.g., repo_root/volumes vs repo_root/data_backend)
elif (_backend_root.parent / "volumes").exists():
    PROJECT_ROOT = _backend_root.parent
else:
    # 3. Fallback: Default to backend root (will create volumes here)
    PROJECT_ROOT = _backend_root

# --- Directory Configuration ---

# 1. Volumes Base Directory
# Allow override via env var, otherwise default to detected PROJECT_ROOT/volumes
_env_volumes = os.getenv("VOLUMES_DIR")
VOLUMES_DIR = Path(_env_volumes) if _env_volumes else PROJECT_ROOT / "volumes"

# 2. Sub-Directories (Single Source of Truth)
# We define ALL shared paths here to prevent mismatch across modules.
RAW_FILES_DIR = VOLUMES_DIR / "raw_files"
DATABASE_DIR = VOLUMES_DIR / "databases"
SESSIONS_DIR = VOLUMES_DIR / "sessions"
LOGS_DIR = VOLUMES_DIR / "logs"

# --- Initialization ---

def initialize_directories():
    """Ensures that all necessary directories exist with correct permissions."""
    required_dirs = [VOLUMES_DIR, RAW_FILES_DIR, DATABASE_DIR, SESSIONS_DIR, LOGS_DIR]
    
    for path in required_dirs:
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                # Use sys.stderr because logger might not be configured yet
                print(f"Warning: Could not create directory {path}: {e}", file=sys.stderr)

# Run initialization immediately on import
initialize_directories()

# --- Helper Functions ---

def _get_type_root_dir(rag_type: str) -> Path:
    """Internal helper to get the root directory for a specific RAG type."""
    if rag_type not in VALID_RAG_TYPES:
        raise ValueError(f"Invalid rag_type: '{rag_type}'. Must be one of: {VALID_RAG_TYPES}")
    
    # Ensure the type-specific folder exists (e.g., volumes/databases/rag)
    type_dir = DATABASE_DIR / rag_type
    if not type_dir.exists():
        type_dir.mkdir(parents=True, exist_ok=True)
    return type_dir

def get_db_paths(rag_type: str, db_name: str) -> Dict[str, Path]:
    """
    Generates standard file paths for a specific RAG type and database instance.
    Includes config_path for validation logic.
    """
    base_dir = _get_type_root_dir(rag_type)
    db_instance_dir = base_dir / db_name

    return {
        "db_dir": db_instance_dir,
        "chroma_path": db_instance_dir / "chroma",
        "graph_path": db_instance_dir / "graph.json",
        "vectorstore_path": db_instance_dir / "vectorstore",
        "config_path": db_instance_dir / "db_config.json"
    }

def list_available_dbs(rag_type: str) -> List[str]:
    """Lists available database names for a given RAG type."""
    try:
        base_dir = _get_type_root_dir(rag_type)
    except ValueError:
        return []

    if not base_dir.exists():
        return []

    return sorted([
        d.name for d in base_dir.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ])

def db_exists(rag_type: str, db_name: str) -> bool:
    """Checks if a specific database instance exists and appears valid."""
    try:
        paths = get_db_paths(rag_type, db_name)
        db_dir = paths["db_dir"]
        
        if not db_dir.exists():
            return False

        # Specific checks based on architecture
        if rag_type == "kag":
            return paths["graph_path"].exists()
        
        # Check for ChromaDB (Standard & LightRAG) or Legacy
        chroma_db_file = paths["chroma_path"] / "chroma.sqlite3"
        return chroma_db_file.exists() or paths["vectorstore_path"].exists()

    except Exception:
        return False