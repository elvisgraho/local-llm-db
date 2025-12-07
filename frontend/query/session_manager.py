import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_DOCKER_PATH = "/app/chat_history"
LOCAL_FALLBACK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history")

# Use env var -> Docker Path (if exists/writable) -> Local Fallback
env_path = os.getenv("CHAT_HISTORY_DIR")
if env_path:
    HISTORY_DIR = env_path
elif os.path.exists("/app"): # Check if we are likely in Docker
    HISTORY_DIR = DEFAULT_DOCKER_PATH
else:
    HISTORY_DIR = LOCAL_FALLBACK_PATH

class SessionManager:
    def __init__(self):
        if not os.path.exists(HISTORY_DIR):
            try:
                os.makedirs(HISTORY_DIR, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create history dir: {e}")

    def list_sessions(self) -> List[Dict]:
        """Returns a list of all chat sessions sorted by date."""
        sessions = []
        try:
            files = [f for f in os.listdir(HISTORY_DIR) if f.endswith('.json')]
            for f in files:
                path = os.path.join(HISTORY_DIR, f)
                try:
                    with open(path, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                        sessions.append({
                            "id": data.get("id"),
                            "title": data.get("title", "Untitled Chat"),
                            "updated_at": data.get("updated_at", ""),
                            "file_path": path
                        })
                except json.JSONDecodeError:
                    continue # Skip corrupted files
        except Exception:
            return []
        
        # Sort by newest first
        return sorted(sessions, key=lambda x: x['updated_at'], reverse=True)

    def load_session(self, session_id: str) -> Dict:
        """Loads a specific chat session."""
        path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return self.create_session()

    def create_session(self, title: str = "New Chat") -> Dict:
        """Creates a new empty session."""
        session_id = str(uuid.uuid4())
        session_data = {
            "id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "temp_context": "" # Store session-specific context (uploaded files)
        }
        self.save_session(session_data)
        return session_data

    def save_session(self, session_data: Dict):
        """Saves the current state of the session to disk."""
        session_data["updated_at"] = datetime.now().isoformat()
        path = os.path.join(HISTORY_DIR, f"{session_data['id']}.json")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def delete_session(self, session_id: str):
        """Deletes a session file."""
        path = os.path.join(HISTORY_DIR, f"{session_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def update_title(self, session_id: str, new_title: str):
        """Renames a chat."""
        data = self.load_session(session_id)
        data["title"] = new_title
        self.save_session(data)

session_manager = SessionManager()