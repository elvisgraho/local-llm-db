import streamlit as st
import os
import sys
import time
import logging

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- Internal Imports ---
from query.query_data import query_direct, query_rag, query_lightrag, query_kag
from query.session_manager import session_manager

# --- New Utils Import ---
import app_utils
import app_settings_ui  # <--- Using your specific file name

# --- Configuration ---
st.set_page_config(
    page_title="RAG Architect",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Styles from Utils
app_utils.apply_custom_css()

# --- Main Application ---
def main():
    # ==========================================
    # SIDEBAR - SESSION MANAGEMENT
    # ==========================================
    if "active_session_id" not in st.session_state:
        sessions = session_manager.list_sessions()
        if sessions:
            st.session_state.active_session_id = sessions[0]["id"]
        else:
            new_sess = session_manager.create_session()
            st.session_state.active_session_id = new_sess["id"]
            
    current_session = session_manager.load_session(st.session_state.active_session_id)
    
    with st.sidebar:
        st.title("🗂️ Chats")
        
        # New Chat Button
        if st.button("➕ New Chat", type="primary", width='stretch'):
            new_sess = session_manager.create_session()
            st.session_state.active_session_id = new_sess["id"]
            st.rerun()

        @st.cache_data(ttl=2, show_spinner=False)
        def get_cached_sessions(_manager):
            return _manager.list_sessions()

        sessions = get_cached_sessions(session_manager)

        # Session List
        active_id = st.session_state.active_session_id
        for s in sessions[:10]: 
            col_name, col_del = st.columns([0.85, 0.15])
            label = s["title"][:22] + "..." if len(s["title"]) > 25 else s["title"]
            
            if s["id"] == active_id:
                label = f"📂 {label}"
            
            if col_name.button(label, key=f"btn_{s['id']}", width='stretch'):
                st.session_state.active_session_id = s["id"]
                st.rerun()
            
            if col_del.button("✕", key=f"del_{s['id']}", help="Delete"):
                session_manager.delete_session(s["id"])
                if s["id"] == active_id:
                    del st.session_state.active_session_id
                st.rerun()
        
        if len(sessions) > 10:
            st.caption(f"...and {len(sessions)-10} more")

        st.divider()

        # ==========================================
        # SIDEBAR - SETTINGS (Imported from UI file)
        # ==========================================
        # This function renders LLM, Embeddings, Personas, DBs, and Params
        # It returns a dictionary `cfg` with all selected values
        cfg = app_settings_ui.render_settings_sidebar()

        # ==========================================
        # SIDEBAR - TOKEN ESTIMATOR
        # ==========================================
        curr_msgs = current_session["messages"] if "active_session_id" in st.session_state else []
        sys_tokens_est = app_utils.count_tokens(cfg["custom_system_prompt"]) if cfg["custom_system_prompt"] else 0
        
        app_utils.render_token_estimator(
            cfg["top_k"] if not cfg["is_direct"] else 0,
            cfg["history_limit"], 
            curr_msgs, 
            cfg["ctx_window"],
            sys_tokens_est
        )

    # ==========================================
    # MAIN CONTENT AREA
    # ==========================================
    
    if "active_session_id" not in st.session_state: st.stop()

    # Header & Title Edit
    col_h1, col_h2 = st.columns([3, 1])
    col_h1.subheader(f"💬 {current_session.get('title', 'Untitled')}")
    new_title = col_h2.text_input("Rename", value=current_session.get("title"), label_visibility="collapsed")
    if new_title != current_session.get("title"):
        session_manager.update_title(current_session["id"], new_title)
        st.rerun()

    # Context Injection (File Upload)
    with st.expander("📎 Add Session Context (Upload File)"):
        uploaded_file = st.file_uploader("Upload PDF/TXT/Code", type=['pdf', 'txt', 'py', 'md', 'json'])
        if uploaded_file:
            if uploaded_file.name not in current_session.get("temp_context", ""):
                with st.spinner("Processing..."):
                    text_content = app_utils.parse_uploaded_file(uploaded_file)
                    current_session["temp_context"] = (current_session.get("temp_context", "") + text_content)
                    session_manager.save_session(current_session)
                    st.success("File added to context!")

    # Warmup Resources
    if not app_utils.warm_up_resources(): st.stop()

    # Render Chat History
    for msg in current_session["messages"]:
        with st.chat_message(msg["role"]):
            if msg.get("reasoning"):
                with st.expander("💭 Reasoning", expanded=False): st.markdown(msg["reasoning"])
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                    for src in msg["sources"]:
                        st.markdown(f"📄 **{os.path.basename(src)}**\n`{src}`")

    # ==========================================
    # INPUT LOOP & LOGIC
    # ==========================================
    if prompt := st.chat_input("Ask about your data..."):
        # 1. Update UI immediately
        current_session["messages"].append({"role": "user", "content": prompt})
        session_manager.save_session(current_session)
        with st.chat_message("user"): st.markdown(prompt)

        # 2. Smart Context Construction
        # Calculate budget using values from `cfg`
        safe_ctx_limit = cfg["ctx_window"] - 1000 - (cfg["top_k"] * 250) 
        
        full_history = [{"role": m["role"], "content": m["content"]} for m in current_session["messages"][:-1]]
        
        # Smart Pruning
        rag_history = app_utils.smart_prune_history(full_history, safe_ctx_limit)
        
        # Enforce the user slider hard cap
        if cfg["history_limit"] < len(rag_history):
            rag_history = rag_history[-cfg["history_limit"]:]

        final_query = prompt
        if current_session.get("temp_context"):
            final_query = f"Session Context:\n{current_session['temp_context']}\n\nQuery: {prompt}"

        # 3. Execution with Granular Status
        with st.chat_message("assistant"):
            start_time = time.time()
            response_container = st.empty()
            
        with st.status("🧠 Orchestrating...", expanded=True) as status:
            try:
                # Base arguments shared by all strategies
                # NOTE: Accessing variables from `cfg` ensures they are always defined
                common_args = {
                    "query_text": final_query, 
                    "llm_config": {
                        "provider": cfg["provider"], 
                        "modelName": cfg["selected_model"], 
                        "apiKey": cfg["api_key"], 
                        "api_url": cfg["local_url"] if cfg["provider"] == "local" else None,
                        "temperature": cfg["temp"], 
                        "system_prompt": cfg["custom_system_prompt"],
                        "context_window": cfg["ctx_window"] 
                    }, 
                    "conversation_history": rag_history
                }
                
                # Routing logic using `cfg` variables
                if cfg["rag_type"] == 'direct':
                    status.write("Direct LLM query (no retrieval)...")
                    response = query_direct(**common_args)
                    
                elif cfg["selected_db"]:
                    status.write(f"🔍 Retrieving from **{cfg['selected_db']}** ({cfg['rag_type']})...")
                    strategies = {'rag': query_rag, 'lightrag': query_lightrag, 'kag': query_kag}
                    
                    response = strategies[cfg["rag_type"]](
                        **common_args, 
                        top_k=cfg["top_k"], 
                        rag_type=cfg["rag_type"], 
                        db_name=cfg["selected_db"], 
                        hybrid=cfg["hybrid"]
                    )
                    src_count = len(response.get("sources", []))
                    status.write(f"✅ Found {src_count} relevant documents.")
                
                status.update(label="Response Generated!", state="complete", expanded=False)

            except Exception as e:
                if type(e).__name__ == "StopException":
                    raise e
                
                status.update(label="❌ Error", state="error")
                st.error(f"Pipeline Error: {str(e)}")
                logging.error(e, exc_info=True)
                st.stop()
                    
            # 4. Parse & Display
            text = response.get("text", "")
            sources = response.get("sources", [])
            tokens_est = response.get("estimated_context_tokens", 0)
            clean_text, reasoning = app_utils.parse_reasoning(text)

            # Display Reasoning (DeepSeek/R1 style)
            if reasoning:
                with st.expander("💭 Internal Thought Process", expanded=False):
                    st.markdown(reasoning)

            response_container.markdown(clean_text)

            # Source Display
            if sources:
                with st.expander(f"📚 Cited Sources ({len(sources)})", expanded=False):
                    unique_sources = list(set(sources))
                    for src in unique_sources:
                        col_ico, col_txt = st.columns([0.05, 0.95])
                        col_ico.text("📄")
                        col_txt.caption(f"{os.path.basename(src)} — `{src}`")

            # Footer Metadata
            col_time, col_tok, col_feed = st.columns([0.2, 0.3, 0.5])
            col_time.caption(f"⏱️ {time.time()-start_time:.2f}s")
            col_tok.caption(f"🪙 {tokens_est} ctx tokens")

            # 5. Save State
            current_session["messages"].append({
                "role": "assistant", "content": clean_text, 
                "reasoning": reasoning, "sources": sources
            })
            
            # Auto-rename
            if len(current_session["messages"]) == 2:
                current_session["title"] = (clean_text[:30] + "...") if len(clean_text) > 30 else clean_text
            
            session_manager.save_session(current_session)
            
if __name__ == "__main__":
    main()