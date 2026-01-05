import streamlit as st
from query.session_manager import session_manager
from query.global_vars import LOCAL_LLM_API_URL

# Default LLM Model fallback
DEFAULT_LLM_MODEL = "local-model"

class StateManager:
    """
    Centralized manager for Streamlit session state.
    Prevents KeyErrors and manages data persistence across reruns.
    """

    def initialize(self):
        """
        Call this at the start of app.py to ensure all state keys exist.
        """
        # --- 1. Session Management ---
        if "session_list" not in st.session_state:
            try:
                st.session_state.session_list = session_manager.list_sessions()
            except Exception:
                st.session_state.session_list = []
        
        if "active_session_id" not in st.session_state:
            self._set_initial_session()

        # --- 2. UI Flags & Metrics ---
        self._init_setting("last_retrieval_count", None)
        self._init_setting("resources_warm", False)

        # --- 3. Configuration Defaults ---
        self._init_setting("llm_url", LOCAL_LLM_API_URL)
        self._init_setting("emb_url", LOCAL_LLM_API_URL)
        
        self._init_setting("rag_top_k", 10)
        self._init_setting("rag_history_limit", 6)

        # --- 2. EAGER MODEL FETCH (The Fix) ---
        # We fetch ONCE when the app starts. 
        if "startup_fetch_done" not in st.session_state:
            import app_utils  # Import here to avoid circular dependency at top level
            
            # A. Fetch Chat Models
            try:
                chat_models = app_utils.fetch_available_models(st.session_state.llm_url)
                # Filter: remove embedding models from chat list
                clean_chat = [m for m in chat_models if "embed" not in m.lower()]
                
                if clean_chat:
                    st.session_state.fetched_models = clean_chat
                    st.session_state.llm_selector = clean_chat[0]
                else:
                    st.session_state.fetched_models = ["local-model"]
            except Exception:
                st.session_state.fetched_models = ["local-model"]

            # B. Fetch Embedding Models
            try:
                emb_models = app_utils.fetch_available_models(st.session_state.emb_url)
                
                if emb_models:
                    st.session_state.fetched_emb_models = emb_models
                    # Smart Select: Auto-pick the one with 'embed' in the name
                    best = next((m for m in emb_models if 'embed' in m.lower()), emb_models[0])
                    st.session_state.emb_model_selector = best
                else:
                    st.session_state.fetched_emb_models = []
            except Exception:
                st.session_state.fetched_emb_models = []

            # Mark complete so we don't re-fetch on every interaction
            st.session_state.startup_fetch_done = True
        
        self._init_setting("rag_strategy", "LightRAG")
        self._init_setting("persisted_db", None)

    def _init_setting(self, key, default_value):
        """Helper to set a default only if key is missing."""
        if key not in st.session_state:
            st.session_state[key] = default_value

    # --- Session Handling ---

    def _set_initial_session(self):
        """Sets the active session to the first available, or creates new."""
        sessions = st.session_state.get("session_list", [])
        if sessions:
            st.session_state.active_session_id = sessions[0]["id"]
        else:
            self.create_new_session()

    def get_active_session_id(self):
        return st.session_state.get("active_session_id")

    def set_active_session(self, session_id):
        """Switches the active session and resets transient UI metrics."""
        st.session_state.active_session_id = session_id
        # Reset to None so the sidebar switches back to "Estimator Mode"
        st.session_state.last_retrieval_count = None

    def get_session_list(self):
        return st.session_state.get("session_list", [])

    def create_new_session(self):
        """Creates a session on disk and updates state immediately."""
        new_sess = session_manager.create_session()
        
        if "session_list" not in st.session_state:
            st.session_state.session_list = []

        st.session_state.session_list.insert(0, {
            "id": new_sess["id"],
            "title": new_sess["title"],
            "updated_at": new_sess["updated_at"]
        })
        
        st.session_state.active_session_id = new_sess["id"]
        # Reset to None so the sidebar switches back to "Estimator Mode"
        st.session_state.last_retrieval_count = None
        return new_sess["id"]

    def delete_session(self, session_id):
        """Deletes from disk and updates state immediately."""
        # 1. Disk Delete
        session_manager.delete_session(session_id)
        
        # 2. State Update
        if "session_list" in st.session_state:
            st.session_state.session_list = [
                s for s in st.session_state.session_list if s["id"] != session_id
            ]
        
        if st.session_state.get("active_session_id") == session_id:
            self._set_initial_session()

    def update_session_title(self, session_id, new_title):
        """Updates title in the local list cache."""
        if "session_list" in st.session_state:
            for s in st.session_state.session_list:
                if s["id"] == session_id:
                    s["title"] = new_title
                    break

    def reset_session(self):
        """Emergency reset if session is corrupt."""
        self._set_initial_session()

    # --- Data Accessors ---

    def get_last_retrieval_count(self):
        # Allow returning None
        return st.session_state.get("last_retrieval_count", None)

    def set_last_retrieval_count(self, count):
        st.session_state.last_retrieval_count = count