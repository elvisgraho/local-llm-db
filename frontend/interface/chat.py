import streamlit as st
import os
import app_utils
from query.session_manager import session_manager

def render_chat_area(session_data, state_manager):
    """
    Renders the main chat interface, including:
    1. Title / Rename Logic
    2. Context File Uploader
    3. Message History
    """
    
    # --- 1. Header & Renaming ---
    _render_header(session_data, state_manager)
    
    # --- 2. Context Injection (File Upload) ---
    _render_file_uploader(session_data)
    
    if "resources_warm" not in st.session_state:
        if not app_utils.warm_up_resources():
            st.stop()
        st.session_state.resources_warm = True

    # --- 4. Message Loop ---
    messages = session_data.get("messages", [])
    
    # Render container to keep layout stable
    chat_container = st.container()
    
    with chat_container:
        for msg in messages:
            _render_single_message(msg)
            
            
    return chat_container

def _render_header(session_data, state_manager):
    """Renders the chat title and rename input with debounce safety."""
    col_h1, col_h2 = st.columns([3, 1])
    
    current_title = session_data.get('title', 'Untitled')
    col_h1.subheader(f"💬 {current_title}")
    
    # CALLBACK FUNCTION
    def _update_title_callback():
        # Get value from session state using the unique key
        key = f"rename_{session_data['id']}"
        new_val = st.session_state[key]
        if new_val and new_val != session_data["title"]:
            session_manager.update_title(session_data["id"], new_val)
            state_manager.update_session_title(session_data["id"], new_val)
            # Toast provides feedback without full rerun, 
            # though the sidebar will update on next action.
            st.toast(f"Renamed to: {new_val}")

    # Rename Input with ON_CHANGE
    # This ensures logic runs immediately when user hits Enter or clicks away
    col_h2.text_input(
        "Rename", 
        value=current_title, 
        label_visibility="collapsed",
        key=f"rename_{session_data['id']}", 
        on_change=_update_title_callback 
    )

def _render_file_uploader(session_data):
    """Handles uploading files into the session context."""
    with st.expander("📎 Add Session Context (Upload File)"):
        uploaded_file = st.file_uploader(
            "Upload PDF/TXT/Code", 
            type=['pdf', 'txt', 'py', 'md', 'json', 'js', 'html', 'css']
        )
        
        if uploaded_file:
            # Check if already processed to avoid duplicates
            current_context = session_data.get("temp_context", "")
            
            if uploaded_file.name not in current_context:
                with st.spinner("Parsing file..."):
                    text_content = app_utils.parse_uploaded_file(uploaded_file)
                    
                    # Update Session Data
                    session_data["temp_context"] = current_context + text_content
                    session_manager.save_session(session_data)
                    
                    st.success(f"Added {uploaded_file.name} to context!")

def _render_single_message(msg):
    """Renders a single message bubble (User or Assistant)."""
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        # 1. Reasoning Expander (DeepSeek Style)
        if msg.get("reasoning"):
            with st.expander("💭 Reasoning Process", expanded=False): 
                st.markdown(msg["reasoning"])

        # 2. Main Content
        # Apply citations formatting ( [Source: x] -> Badge )
        formatted_content = app_utils.format_citations(content)
        # Apply HTML/Table safety
        formatted_content = app_utils.sanitize_markdown(formatted_content)
        
        st.markdown(formatted_content, unsafe_allow_html=True)

        # 3. Sources Footer
        if msg.get("sources"):
            _render_sources(msg["sources"])

def _render_sources(sources):
    """Renders the list of source documents used for a response."""
    with st.expander(f"📚 Cited Sources ({len(sources)})"):
        # Deduplicate sources
        unique_sources = list(set(sources))
        for src in unique_sources:
            col_ico, col_txt = st.columns([0.05, 0.95])
            col_ico.text("📄")
            col_txt.caption(f"{os.path.basename(src)} — `{src}`")