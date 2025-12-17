import json
import time
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ProcessingHistory:
    """
    Manages incremental processing state using a JSON file.
    Supports atomic writes and mtime-based invalidation.
    """
    def __init__(self, history_file: Path, save_interval: int = 10):
        self.history_file = history_file
        self.save_interval = save_interval
        self.data: Dict[str, Any] = self._load()
        self.unsaved_changes = 0

    def _load(self) -> Dict[str, Any]:
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted history file {self.history_file}. Starting fresh.")
                return {}
        return {}

    def save(self):
        """Atomically save history to disk to prevent corruption."""
        if self.unsaved_changes == 0:
            return

        try:
            temp_file = self.history_file.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            # Atomic move
            shutil.move(str(temp_file), str(self.history_file))
            self.unsaved_changes = 0
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def should_process(self, file_path: Path) -> bool:
        """
        Returns True if file is new or modified since last record.
        """
        if not file_path.exists():
            return False

        key = str(file_path.resolve())
        current_mtime = file_path.stat().st_mtime

        if key not in self.data:
            return True

        last_mtime = self.data[key].get('mtime', 0)
        return current_mtime > last_mtime

    def get_record(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Retrieve the stored data for a file."""
        return self.data.get(str(file_path.resolve()))

    def record_processing(self, file_path: Path, mtime: float = None, **kwargs):
        """
        Update history for a file. 
        Auto-captures current mtime if not provided.
        """
        key = str(file_path.resolve())
        
        # Use provided mtime (useful if file was just deleted/moved) or current stat
        if mtime is None:
            try:
                mtime = file_path.stat().st_mtime
            except FileNotFoundError:
                mtime = time.time() # Fallback

        entry = {
            "mtime": mtime,
            "processed_at": time.time(),
            **kwargs
        }
        
        self.data[key] = entry
        self.unsaved_changes += 1
        
        if self.unsaved_changes >= self.save_interval:
            self.save()