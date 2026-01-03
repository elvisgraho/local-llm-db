"""
Shared common module for data_backend.

This module provides unified configuration, path management, and utilities
that work across all deployment modes (standalone, Docker, docker-compose).
"""

from .config import config, Config, LLMConfig, RAGConfig
from .paths import (
    PROJECT_ROOT,
    VOLUMES_DIR,
    RAW_FILES_DIR,
    DATABASE_DIR,
    SESSIONS_DIR,
    LOGS_DIR,
    get_db_paths,
    list_available_dbs,
    db_exists,
    initialize_directories
)
from .system_utils import (
    get_system_metrics,
    get_dir_stats,
    delete_database_instance,
    kill_child_processes,
    is_process_alive,
    run_script_generator,
    start_script_background,
    read_log_file,
    JobState
)
from .ui_helpers import (
    fetch_local_models,
    scan_databases,
    get_file_inventory,
    extract_text_for_preview,
    filter_models
)

__all__ = [
    # Config
    'config',
    'Config',
    'LLMConfig',
    'RAGConfig',
    # Paths
    'PROJECT_ROOT',
    'VOLUMES_DIR',
    'RAW_FILES_DIR',
    'DATABASE_DIR',
    'SESSIONS_DIR',
    'LOGS_DIR',
    'get_db_paths',
    'list_available_dbs',
    'db_exists',
    'initialize_directories',
    # System Utils
    'get_system_metrics',
    'get_dir_stats',
    'delete_database_instance',
    'kill_child_processes',
    'is_process_alive',
    'run_script_generator',
    'start_script_background',
    'read_log_file',
    'JobState',
    # UI Helpers
    'fetch_local_models',
    'scan_databases',
    'get_file_inventory',
    'extract_text_for_preview',
    'filter_models',
]
