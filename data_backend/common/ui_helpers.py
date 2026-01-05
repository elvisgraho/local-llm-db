"""
UI-specific helper functions for the Streamlit interface.

This module contains utilities for file management, database scanning,
and data formatting specific to the UI layer.
"""

import fitz
import json
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .paths import DATABASE_DIR, RAW_FILES_DIR
from .system_utils import get_dir_stats


def fetch_local_models(api_url: str) -> List[str]:
    """
    Fetch available models from a local LLM API (LM Studio/Ollama compatible).

    Args:
        api_url: Base URL of the LLM API

    Returns:
        List of model names/IDs
    """
    try:
        clean_url = api_url.rstrip('/')
        if not clean_url.endswith("/v1"):
            clean_url += "/v1"
        response = requests.get(f"{clean_url}/models", timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return [m['id'] for m in data['data']]
            return [str(m) for m in data]
    except Exception:
        return []
    return []


def scan_databases(db_root: Path = DATABASE_DIR) -> List[Dict[str, Any]]:
    """
    Scan database directory and return inventory with metadata.

    Args:
        db_root: Root directory containing RAG databases

    Returns:
        List of database info dictionaries
    """
    inventory = []
    if not db_root.exists():
        return inventory

    # Only scan for LightRAG databases
    type_dir = db_root / "lightrag"
    if type_dir.exists():
        for db_instance in type_dir.iterdir():
            if db_instance.is_dir() and not db_instance.name.startswith('.'):
                size_mb, _ = get_dir_stats(db_instance)

                # Count processed files
                file_count = 0
                reg_path = db_instance / "processed_files.json"
                if reg_path.exists():
                    try:
                        with open(reg_path, 'r', encoding='utf-8') as f:
                            file_count = len(json.load(f))
                    except:
                        pass

                # Read configuration
                config = {}
                config_path = db_instance / "db_config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                    except:
                        pass

                inventory.append({
                    "Type": "lightrag",
                    "Name": db_instance.name,
                    "Size": f"{size_mb:.1f} MB",
                    "Files": file_count,
                    "Path": str(db_instance),
                    "Config": config
                })
    return inventory


def get_file_inventory(
    raw_files_dir: Path = RAW_FILES_DIR,
    limit: int = 50
) -> tuple[List[Dict[str, str]], int]:
    """
    Get inventory of files in the staging area.

    Args:
        raw_files_dir: Directory containing raw files
        limit: Maximum number of files to return details for

    Returns:
        Tuple of (file_list, total_count)
    """
    try:
        if not raw_files_dir.exists():
            return [], 0

        # Fast count
        file_count = sum(
            1 for _ in raw_files_dir.rglob('*')
            if _.is_file() and not _.name.startswith('.')
        )

        if file_count == 0:
            return [], 0

        # Get iterator
        files_iter = (
            f for f in raw_files_dir.rglob('*')
            if f.is_file() and not f.name.startswith('.')
        )

        # Take first 'limit' files (faster than converting thousands to list)
        files_subset = []
        for _, f in zip(range(limit), files_iter):
            files_subset.append(f)

        files_formatted = [{
            "Filename": str(f.relative_to(raw_files_dir)),
            "Size (KB)": f"{f.stat().st_size/1024:.1f}",
            "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime('%H:%M:%S')
        } for f in files_subset]

        return files_formatted, file_count
    except Exception:
        return [], 0


def extract_text_for_preview(file_path: Path, char_limit: int = 10000) -> str:
    """
    Extract text from a file for preview purposes.

    Args:
        file_path: Path to the file
        char_limit: Maximum characters to return

    Returns:
        Extracted text or error message
    """
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            with fitz.open(file_path) as doc:
                text = "".join(page.get_text() for page in doc)
        elif suffix in ['.txt', '.md', '.markdown', '.log', '.json']:
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                text = f.read(char_limit * 2)
        else:
            return f"Preview not supported for {suffix}"
        return text[:char_limit] + ("\n\n[Truncated]" if len(text) > char_limit else "")
    except Exception as e:
        return f"Error reading file: {str(e)}"


def filter_models(models: List[str], filter_type: str = 'chat') -> List[str]:
    """
    Filter models by type (embedding vs chat vs ocr).

    Args:
        models: List of model names
        filter_type: 'chat', 'embed', or 'ocr'

    Returns:
        Filtered list of models
    """
    if not models:
        return []

    embed_keywords = ['embed', 'bert', 'nomic', 'gte', 'bge', 'e5']
    ocr_keywords = ['ocr', 'vision', 'gliese', 'llava', 'bakllava', 'moondream']

    if filter_type == 'embed':
        return [m for m in models if any(k in m.lower() for k in embed_keywords)]
    elif filter_type == 'ocr':
        return [m for m in models if any(k in m.lower() for k in ocr_keywords)]
    else:  # chat
        # Exclude both embed and ocr keywords for chat models
        return [m for m in models if not any(k in m.lower() for k in embed_keywords + ocr_keywords)]
