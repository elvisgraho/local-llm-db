"""
Configuration file for the training and data service components.
This file sets up paths and configurations specific to document processing and training.
"""

import os
import sys
from pathlib import Path

# Get the training directory (where this config.py is located)
TRAINING_DIR = Path(__file__).parent.absolute()

# Add the parent directory to Python path
sys.path.insert(0, str(TRAINING_DIR.parent))

# Define paths relative to the training directory
DATA_DIR = TRAINING_DIR / "data"
DATABASES_DIR = TRAINING_DIR / "databases"

# Ensure directories exist
for directory in [DATA_DIR, DATABASES_DIR]:
    directory.mkdir(exist_ok=True) 