import streamlit as st
import os
import app_utils
from query.templates import QUERY_REWRITE_PAYLOAD, QUERY_REWRITE_SYSTEM_PROMPT
from query.session_manager import session_manager
from query.query_data import query_direct 
from interface.styles import inject_sticky_css

# --- Cache text processing ---
@st.cache_data(show_spinner=False, max_entries=1000)
def _process_message_content(content: str):
    if not content: return ""
    formatted = app_utils.format_citations(content)
    formatted = app_utils.sanitize_markdown(formatted)
    return formatted

def render_chat_area(session_data, state_manager, config):
    """
    Renders history + Sticky Custom Input Area.
    """
    inject_sticky_css()
    _render_header(session_data, state_manager)
    _render_file_uploader(session_data)
    
    if not st.session_state.get("resources_warm", False):
        if not app_utils.warm_up_resources(): st.stop()
        st.session_state.resources_warm = True

    # Render History
    messages = session_data.get("messages", [])
    
    # Use a specific container for history
    history_container = st.container()
    
    with history_container:
        prev_role = None
        prev_content_hash = None
        for msg in messages:
            curr_hash = hash(msg["content"])
            if msg["role"] == prev_role and curr_hash == prev_content_hash:
                continue
            _render_single_message(msg)
            prev_role = msg["role"]
            prev_content_hash = curr_hash

    # Render Sticky Input (Passed session_data for history access)
    render_custom_chat_input(config, session_data) 

    return history_container

# --- CALLBACKS (Logic before Render) ---

def _cb_rewrite(config, session_data):
    """Enhances the query using LLM + History."""
    current_text = st.session_state.get("custom_chat_input_widget", "")
    
    if not current_text.strip():
        st.toast("⚠️ Type something first!")
        return

    try:
        # --- 1. History Extraction & Cleaning ---
        # Get raw messages, excluding the current input (if it's already in the list)
        # Limit to last 10 items to prevent context overflow.
        raw_msgs = session_data.get("messages", [])[:-1][-10:]
        
        # Filter: Exclude previous 'system' prompts to prevent instruction leakage.
        # We only want the semantic content of the conversation.
        clean_history_lines = []
        for m in raw_msgs:
            if m.get("role") in ["user", "assistant"]:
                clean_history_lines.append(f"{m['role'].upper()}: {m['content']}")
        
        # Join lines or mark as empty
        history_str = "\n".join(clean_history_lines) if clean_history_lines else "No previous context."

        # --- 2. Construct the Dynamic User Prompt ---
        # This is the actual payload we send as the "User Message".
        # It presents the data clearly separated from the instructions.
        
        rewrite_payload = QUERY_REWRITE_PAYLOAD.format(
                history_str=history_str,
                current_text=current_text
            )

        # --- 3. Configure the LLM ---
        llm_cfg = config["llm_config"]
        rw_config = llm_cfg.copy()
        
        # Assign the STATIC instructions
        rw_config["system_prompt"] = QUERY_REWRITE_SYSTEM_PROMPT
        
        # Use low temperature for deterministic, precise rewriting
        rw_config["temperature"] = 0.3 

        # --- 4. Execute Rewrite ---
        # query_text gets the Dynamic Payload (Data)
        # conversation_history is empty [] because we manually formatted the history 
        # into the payload above. We don't want the LLM API to append history automatically.
        res = query_direct(
            query_text=rewrite_payload,
            llm_config=rw_config,
            conversation_history=[] 
        )
        
        new_text = res['text'].strip()

        # Sanity check: if response is empty, fallback to original
        if not new_text:
            new_text = current_text
        

        # 4. Update Widget State
        st.session_state.custom_chat_input_widget = new_text

        if new_text == current_text:
            st.toast("Query can't be improved.")
        else:
            st.toast("Query improved!")
        
    except Exception as e:
        st.error(f"Rewrite failed: {e}")

def _cb_send():
    """Queues the message for processing and clears the input."""
    current_text = st.session_state.get("custom_chat_input_widget", "")
    if not current_text.strip():
        return
    
    # Queue for main.py
    st.session_state.pending_injected_prompt = current_text
    
    # Clear Input
    st.session_state.custom_chat_input_widget = ""

# --- MAIN INPUT RENDERER ---

def render_custom_chat_input(config, session_data):
    """
    Renders the fixed-bottom input area with buttons on top.
    """
    # Start Sticky Container
    st.markdown('<div class="sticky-input-container">', unsafe_allow_html=True)
    
    # 1. ACTION BAR (Buttons on Top)
    # We use columns to position them nicely above the text box
    col_spacer, col_rewrite, col_send = st.columns([0.8, 0.1, 0.1])
    
    with col_rewrite:
        st.button(
            "✨ Enhance", 
            use_container_width=True, 
            help="Rewrite prompt using chat history",
            on_click=_cb_rewrite,
            args=(config, session_data)
        )
            
    with col_send:
        st.button(
            "🚀 Send", 
            type="primary", 
            use_container_width=True,
            on_click=_cb_send
        )

    # 2. TEXT INPUT AREA
    # This sits below the buttons, inside the same sticky container
    st.text_area(
        "Chat Input",
        height=85, # Comfortable height for typing
        placeholder="Type your query here... (Ctrl+Enter to send)",
        label_visibility="collapsed",
        key="custom_chat_input_widget"
    )
    
    # End Sticky Container
    st.markdown('</div>', unsafe_allow_html=True)

# --- Helpers ---

def _render_header(session_data, state_manager):
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
    with st.expander("📎 Add Session Context (Upload File)"):
        uploaded_file = st.file_uploader("Upload PDF/TXT/Code", type=['pdf', 'txt', 'py', 'md', 'json', 'js', 'html'])
        if uploaded_file:
            processed_files = session_data.get("processed_files", [])
            is_duplicate = (uploaded_file.name in processed_files)
            if not is_duplicate:
                with st.spinner("Parsing file..."):
                    text_content = app_utils.parse_uploaded_file(uploaded_file)
                    session_data["temp_context"] = session_data.get("temp_context", "") + text_content
                    if "processed_files" not in session_data: session_data["processed_files"] = []
                    session_data["processed_files"].append(uploaded_file.name)
                    session_manager.save_session(session_data)
                    st.success(f"Added {uploaded_file.name}!")

def _render_single_message(msg):
    role = msg["role"]
    with st.chat_message(role):
        if msg.get("reasoning"):
            with st.expander("💭 Reasoning", expanded=False): st.markdown(msg["reasoning"])
        st.markdown(_process_message_content(msg["content"]), unsafe_allow_html=True)
        if msg.get("sources"):
            _render_sources(msg["sources"])

def _render_sources(sources):
    with st.expander(f"📚 Cited Sources ({len(sources)})"):
        for src in list(set(sources)):
            st.caption(f"📄 {os.path.basename(src)}")