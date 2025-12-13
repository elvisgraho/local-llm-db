import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict
from pathlib import Path

# Import Project Root to ensure local fallback lands in the right place
from query.database_paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 1. Define Paths
DEFAULT_DOCKER_PATH = Path("/app/chat_history")
# CHANGE: Local fallback now goes to 'volumes/sessions' to match other tools
LOCAL_FALLBACK_PATH = PROJECT_ROOT / "volumes" / "sessions"

# 2. Adaptive Logic (The "Smart" Part)
env_path = os.getenv("CHAT_HISTORY_DIR")

if env_path:
    HISTORY_DIR = Path(env_path)
    logger.info(f"Session Manager: Using ENV path -> {HISTORY_DIR}")
elif Path("/app").exists(): # Likely Docker
    HISTORY_DIR = DEFAULT_DOCKER_PATH
    logger.info(f"Session Manager: Detected Docker -> {HISTORY_DIR}")
else:
    HISTORY_DIR = LOCAL_FALLBACK_PATH
    logger.info(f"Session Manager: Using Local Fallback -> {HISTORY_DIR}")

class SessionManager:
    def __init__(self):
        try:
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create history dir at {HISTORY_DIR}: {e}")

    def list_sessions(self) -> List[Dict]:
        """Returns a list of all chat sessions sorted by date."""
        sessions = []
        if not HISTORY_DIR.exists():
            return []

        try:
            # Use glob for cleaner file finding
            for file_path in HISTORY_DIR.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        sessions.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Untitled Chat"),
                            "updated_at": data.get("updated_at", ""),
                            "file_path": str(file_path)
                        })
                except (json.JSONDecodeError, OSError):
                    continue 
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []
        
        return sorted(sessions, key=lambda x: x.get('updated_at', ''), reverse=True)

    def load_session(self, session_id: str) -> Dict:
        path = HISTORY_DIR / f"{session_id}.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
        return self.create_session()

    def create_session(self, title: str = "New Chat") -> Dict:
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "temp_context": ""
        }
        self.save_session(session_data)
        return session_data

    def save_session(self, session_data: Dict):
        session_data["updated_at"] = datetime.now().isoformat()
        path = HISTORY_DIR / f"{session_data['id']}.json"
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def delete_session(self, session_id: str):
        path = HISTORY_DIR / f"{session_id}.json"
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    def update_title(self, session_id: str, new_title: str):
        path = HISTORY_DIR / f"{session_id}.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data["title"] = new_title
                self.save_session(data)
            except Exception:
                pass

session_manager = SessionManager()