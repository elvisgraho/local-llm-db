import streamlit as st
import sys
import os
import logging

#Path Setup
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from interface.state import StateManager
from interface.styles import apply_custom_styles
from interface.sidebar import render_sidebar
from interface.chat import render_chat_area
from interface.processor import process_user_input
from query.session_manager import session_manager

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Config
st.set_page_config(page_title="RAG Architect", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

def main():
    # 1. Init State
    state = StateManager()
    state.initialize()

    # 2. Apply CSS
    apply_custom_styles()

    # 3. Load Session Data FIRST
    # (We need this data to calculate tokens in the sidebar)
    active_id = state.get_active_session_id()
    current_session = None
    
    if active_id:
        try:
            current_session = session_manager.load_session(active_id)
        except Exception as e:
            st.error(f"Error loading session: {e}")
            state.reset_session()
            st.rerun()

    # 4. Render Sidebar
    # Pass the loaded session so the Token Estimator works
    app_config = render_sidebar(state, session_data=current_session)

    # 5. Main Content Check
    if not current_session:
        st.warning("⚠️ No active session. Please create a new chat.")
        st.stop()

    # 6. Render Chat Area
    render_chat_area(current_session, state)

    # 7. Input Loop
    if prompt := st.chat_input("Ask about your data..."):
        current_session["messages"].append({"role": "user", "content": prompt})
        session_manager.save_session(current_session)
        
        st.session_state.pending_processing = True 
        st.rerun()

    # 8. Processing Hook
    if st.session_state.get("pending_processing", False):
        
        # Safety check: Ensure the last message is actually from the user
        if current_session["messages"] and current_session["messages"][-1]["role"] == "user":
            process_user_input(
                session_data=current_session,
                config=app_config,
                state_manager=state
            )
        
        # Reset flag so refreshing the page doesn't re-trigger the LLM
        st.session_state.pending_processing = False
        # Optional: One last rerun to clean the state logic, though process_user_input usually handles it
        # st.rerun() 

if __name__ == "__main__":
    main()