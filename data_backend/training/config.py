"""
Configuration file for the training and data service components.
This file sets up paths and configurations specific to document processing and training.
"""

import os
import sys
from pathlib import Path

# Get the training directory (where this config.py is located)
TRAINING_DIR = Path(__file__).parent.absolute()

# 1. Define Project Root (One level up from training)
PROJECT_ROOT = TRAINING_DIR.parent

# Add the project root to Python path if not present
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 2. Define paths relative to the PROJECT ROOT, not training dir
DATA_DIR = PROJECT_ROOT / "data"
DATABASES_DIR = PROJECT_ROOT / "databases"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASES_DIR.mkdir(parents=True, exist_ok=True)

# --- Embedding Configuration ---
EMBEDDING_CONTEXT_LENGTH = 1512