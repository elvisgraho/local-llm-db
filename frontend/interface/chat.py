import streamlit as st
import os
import app_utils
from query.session_manager import session_manager

# --- OPTIMIZATION: Cache the expensive text processing ---
# This prevents re-running Regex on the entire chat history on every frame.
@st.cache_data(show_spinner=False, max_entries=1000)
def _process_message_content(content: str):
    """
    Pure function to handle expensive string manipulation.
    Cached so it runs ONCE per unique message content.
    """
    if not content: return ""
    # 1. Apply citations formatting ( [Source: x] -> Badge )
    formatted = app_utils.format_citations(content)
    # 2. Apply HTML/Table safety
    formatted = app_utils.sanitize_markdown(formatted)
    return formatted

def render_chat_area(session_data, state_manager):
    """
    Renders the main chat interface with high-performance optimizations
    and duplicate protection.
    """
    
    # --- 1. Header & Renaming ---
    _render_header(session_data, state_manager)
    
    # --- 2. Context Injection (File Upload) ---
    _render_file_uploader(session_data)
    
    # --- 3. Resource Warmup ---
    # Moved to a session_state flag to check boolean (fast) instead of func call
    if not st.session_state.get("resources_warm", False):
        if not app_utils.warm_up_resources():
            st.stop()
        st.session_state.resources_warm = True

    # --- 4. Message Loop with Deduplication ---
    messages = session_data.get("messages", [])
    
    # Container for layout stability
    chat_container = st.container()
    
    with chat_container:
        # TRACKING: Keep track of previous message to detect "Double Paste" bugs
        prev_role = None
        prev_content_hash = None

        for i, msg in enumerate(messages):
            curr_role = msg["role"]
            curr_content = msg["content"]
            # Simple hash for content comparison
            curr_hash = hash(curr_content)

            # BUG FIX: CONSECUTIVE DEDUPLICATION
            # If role matches AND content hash matches previous, it's a duplicate/double-paste.
            # We skip rendering it to clean up the UI.
            if curr_role == prev_role and curr_hash == prev_content_hash:
                continue

            _render_single_message(msg)
            
            # Update trackers
            prev_role = curr_role
            prev_content_hash = curr_hash
            
    return chat_container

def _render_header(session_data, state_manager):
    """Renders the chat title and rename input."""
    col_h1, col_h2 = st.columns([3, 1])
    
    current_title = session_data.get('title', 'Untitled')
    col_h1.subheader(f"💬 {current_title}")
    
    def _update_title_callback():
        key = f"rename_{session_data['id']}"
        if key in st.session_state:
            new_val = st.session_state[key]
            if new_val and new_val != session_data["title"]:
                session_manager.update_title(session_data["id"], new_val)
                state_manager.update_session_title(session_data["id"], new_val)
                st.toast(f"Renamed to: {new_val}")

    col_h2.text_input(
        "Rename", 
        value=current_title, 
        label_visibility="collapsed",
        key=f"rename_{session_data['id']}", 
        on_change=_update_title_callback 
    )

def _render_file_uploader(session_data):
    """Handles uploading files with optimized checking."""
    with st.expander("📎 Add Session Context (Upload File)"):
        uploaded_file = st.file_uploader(
            "Upload PDF/TXT/Code", 
            type=['pdf', 'txt', 'py', 'md', 'json', 'js', 'html', 'css']
        )
        
        if uploaded_file:
            # OPTIMIZATION: Use a separate metadata list for filenames 
            # instead of searching the massive content string.
            processed_files = session_data.get("processed_files", [])
            
            # Fallback for legacy sessions: Check string length (still faster than full search)
            current_context = session_data.get("temp_context", "")
            is_duplicate = (uploaded_file.name in processed_files)
            
            if not is_duplicate and (uploaded_file.name in current_context):
                is_duplicate = True

            if not is_duplicate:
                with st.spinner("Parsing file..."):
                    text_content = app_utils.parse_uploaded_file(uploaded_file)
                    
                    # Update Session Data
                    session_data["temp_context"] = current_context + text_content
                    
                    # Track filename to prevent future duplicate processing
                    if "processed_files" not in session_data:
                        session_data["processed_files"] = []
                    session_data["processed_files"].append(uploaded_file.name)
                    
                    session_manager.save_session(session_data)
                    st.success(f"Added {uploaded_file.name} to context!")

def _render_single_message(msg):
    """Renders a single message bubble (User or Assistant)."""
    role = msg["role"]
    content = msg["content"]
    
    with st.chat_message(role):
        # 1. Reasoning Expander
        if msg.get("reasoning"):
            with st.expander("💭 Reasoning Process", expanded=False): 
                st.markdown(msg["reasoning"])

        # 2. Main Content (Optimized)
        # Use the cached processor to skip expensive regex re-runs
        formatted_content = _process_message_content(content)
        
        st.markdown(formatted_content, unsafe_allow_html=True)

        # 3. Sources Footer
        if msg.get("sources"):
            _render_sources(msg["sources"])

def _render_sources(sources):
    """Renders the list of source documents."""
    with st.expander(f"📚 Cited Sources ({len(sources)})"):
        unique_sources = list(set(sources))
        for src in unique_sources:
            col_ico, col_txt = st.columns([0.05, 0.95])
            col_ico.text("📄")
            # Simply basename to avoid long paths in UI
            col_txt.caption(f"{os.path.basename(src)}")