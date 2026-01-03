"""
Unified path management for data_backend.

This module provides a single source of truth for all file paths,
working seamlessly across standalone, Docker, and docker-compose deployments.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# --- Constants ---

VALID_RAG_TYPES: Tuple[str, ...] = ("rag", "lightrag")
DEFAULT_DB_NAME = "default"

# --- Adaptive Root Determination ---
# This works for both frontend and data_backend regardless of deployment mode

_current_file = Path(__file__).resolve()  # [root]/data_backend/common/paths.py
_backend_root = _current_file.parent.parent  # [root]/data_backend/

# Strategy: Find where 'volumes' directory lives
# 1. Check if we're in data_backend folder with volumes as sibling
if (_backend_root.parent / "volumes").exists():
    PROJECT_ROOT = _backend_root.parent  # Go up to repo root
# 2. Check if volumes is inside backend root (Docker mount point)
elif (_backend_root / "volumes").exists():
    PROJECT_ROOT = _backend_root
# 3. Check if we're running from frontend/
elif _backend_root.name == "frontend" and (_backend_root.parent / "volumes").exists():
    PROJECT_ROOT = _backend_root.parent
else:
    # Fallback: Use backend root and create volumes there
    PROJECT_ROOT = _backend_root

# --- Directory Configuration ---

# Allow environment variable overrides for Docker flexibility
_env_volumes = os.getenv("VOLUMES_DIR")
VOLUMES_DIR = Path(_env_volumes) if _env_volumes else PROJECT_ROOT / "volumes"

# Sub-directories (single source of truth)
RAW_FILES_DIR = Path(os.getenv("RAW_FILES_DIR", VOLUMES_DIR / "raw_files"))
DATABASE_DIR = Path(os.getenv("RAG_DATABASE_DIR", VOLUMES_DIR / "databases"))
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", VOLUMES_DIR / "sessions"))
LOGS_DIR = Path(os.getenv("LOGS_DIR", VOLUMES_DIR / "logs"))

# --- Initialization ---

def initialize_directories():
    """Ensures that all necessary directories exist with correct permissions."""
    required_dirs = [VOLUMES_DIR, RAW_FILES_DIR, DATABASE_DIR, SESSIONS_DIR, LOGS_DIR]

    for path in required_dirs:
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Warning: Could not create directory {path}: {e}", file=sys.stderr)

# Run initialization immediately on import
initialize_directories()

# --- Helper Functions ---

def _get_type_root_dir(rag_type: str) -> Path:
    """Internal helper to get the root directory for a specific RAG type."""
    if rag_type not in VALID_RAG_TYPES:
        raise ValueError(f"Invalid rag_type: '{rag_type}'. Must be one of: {VALID_RAG_TYPES}")

    type_dir = DATABASE_DIR / rag_type
    if not type_dir.exists():
        type_dir.mkdir(parents=True, exist_ok=True)
    return type_dir


def get_db_paths(rag_type: str, db_name: str) -> Dict[str, Path]:
    """
    Generates standard file paths for a specific RAG type and database instance.

    Args:
        rag_type: Type of RAG database ('rag' or 'lightrag')
        db_name: Name of the database instance

    Returns:
        Dictionary containing paths for db_dir, chroma_path, vectorstore_path, config_path
    """
    base_dir = _get_type_root_dir(rag_type)
    db_instance_dir = base_dir / db_name

    return {
        "db_dir": db_instance_dir,
        "chroma_path": db_instance_dir / "chroma",
        "vectorstore_path": db_instance_dir / "vectorstore",
        "config_path": db_instance_dir / "db_config.json"
    }


def list_available_dbs(rag_type: str) -> List[str]:
    """
    Lists available database names for a given RAG type.

    Args:
        rag_type: Type of RAG database ('rag' or 'lightrag')

    Returns:
        Sorted list of database names
    """
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
    """
    Checks if a specific database instance exists and appears valid.

    Args:
        rag_type: Type of RAG database ('rag' or 'lightrag')
        db_name: Name of the database instance

    Returns:
        True if database exists and contains data
    """
    try:
        paths = get_db_paths(rag_type, db_name)
        db_dir = paths["db_dir"]

        if not db_dir.exists():
            return False

        # Check for ChromaDB (Standard & LightRAG) or Legacy vectorstore
        chroma_db_file = paths["chroma_path"] / "chroma.sqlite3"
        return chroma_db_file.exists() or paths["vectorstore_path"].exists()

    except Exception:
        return False
