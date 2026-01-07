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
            try:
                HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create history directory: {e}")
            return []

        try:
            # Use glob for cleaner file finding
            for file_path in HISTORY_DIR.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        data = json.load(file)

                        # Validate essential fields
                        session_id = data.get("id")
                        if not session_id:
                            logger.warning(f"Session file {file_path} missing ID, skipping")
                            continue

                        # Ensure title has a valid value
                        title = data.get("title", "").strip()
                        if not title:
                            # Use first user message as fallback
                            messages = data.get("messages", [])
                            user_msg = next((m["content"] for m in messages if m.get("role") == "user" and m.get("content")), None)
                            if user_msg:
                                title = user_msg[:35].strip() + ("..." if len(user_msg) > 35 else "")
                            else:
                                title = f"Chat {session_id[:8]}"

                        sessions.append({
                            "id": session_id,
                            "title": title,
                            "updated_at": data.get("updated_at", data.get("created_at", "")),
                            "file_path": str(file_path)
                        })
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(f"Failed to load session file {file_path}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

        return sorted(sessions, key=lambda x: x.get('updated_at', ''), reverse=True)

    def load_session(self, session_id: str) -> Dict:
        """Load a session by ID with validation and error recovery."""
        path = HISTORY_DIR / f"{session_id}.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Validate and sanitize the loaded data
                if not isinstance(data, dict):
                    raise ValueError("Session data is not a dictionary")

                # Ensure required fields exist
                data.setdefault("id", session_id)
                data.setdefault("messages", [])
                data.setdefault("temp_context", "")

                # Ensure title is valid
                if not data.get("title") or not data.get("title").strip():
                    messages = data.get("messages", [])
                    user_msg = next((m["content"] for m in messages if m.get("role") == "user" and m.get("content")), None)
                    if user_msg:
                        data["title"] = user_msg[:35].strip() + ("..." if len(user_msg) > 35 else "")
                    else:
                        data["title"] = "Untitled Chat"

                return data
            except Exception as e:
                logger.error(f"Failed to load session {session_id}: {e}")
                # Try to recover by creating a new session
                logger.info(f"Creating new session as recovery")

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
        """Save session with validation and error handling."""
        if not session_data or not isinstance(session_data, dict):
            logger.error("Invalid session data provided to save_session")
            return

        if "id" not in session_data:
            logger.error("Session data missing ID field")
            return

        # Update timestamp
        session_data["updated_at"] = datetime.now().isoformat()

        # Ensure title exists and is valid
        if not session_data.get("title") or not session_data.get("title").strip():
            messages = session_data.get("messages", [])
            user_msg = next((m["content"] for m in messages if m.get("role") == "user" and m.get("content")), None)
            if user_msg:
                session_data["title"] = user_msg[:35].strip() + ("..." if len(user_msg) > 35 else "")
            else:
                session_data["title"] = "Untitled Chat"

        path = HISTORY_DIR / f"{session_data['id']}.json"
        try:
            # Ensure directory exists
            HISTORY_DIR.mkdir(parents=True, exist_ok=True)

            # Write to temp file first, then rename (atomic operation)
            temp_path = path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            # Atomic rename
            temp_path.replace(path)

        except Exception as e:
            logger.error(f"Failed to save session {session_data.get('id')}: {e}")
            # Clean up temp file if it exists
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass

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