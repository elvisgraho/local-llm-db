# --- START OF FILE app.py ---
import streamlit as st
import os
import sys
import time
import logging

# --- Path Setup ---
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# --- Internal Imports ---
from query.query_data import query_direct, query_rag, query_lightrag, query_kag
from query.database_paths import list_available_dbs, DEFAULT_DB_NAME
from query.global_vars import LOCAL_LLM_API_URL
from query.session_manager import session_manager

# --- New Utils Import ---
import app_utils

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
    # SIDEBAR
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
        
        # --- Session Management ---
        if st.button("➕ New Chat", type="primary", use_container_width=True):
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
            
            if col_name.button(label, key=f"btn_{s['id']}", use_container_width=True):
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

        # --- Settings ---
        with st.expander("🔌 LLM Settings", expanded=True):
            provider = st.radio("Provider", ["local", "gemini"], horizontal=True, label_visibility="collapsed")
            api_key = None
            selected_model = ""
            local_url = LOCAL_LLM_API_URL
            
            if provider == "gemini":
                api_key = st.text_input("Gemini API Key", type="password")
                selected_model = st.text_input("Gemini Model", value="gemini-1.5-flash")
                ctx_window = 32000 # Gemini assumption
            else:
                local_url = st.text_input("API URL", value=LOCAL_LLM_API_URL)
                available_models = app_utils.fetch_available_models(local_url)
                if available_models:
                    selected_model = st.selectbox("Select Model", available_models)
                else:
                    selected_model = st.text_input("Model Name", value="local-model")
                
                # Context Window Setting for Token Estimator
                ctx_window = st.selectbox("Model Context Limit", [4096, 8192, 16384, 32768, 128000], index=1, help="Max tokens your model supports.")


        with st.expander("🎭 System Personas", expanded=False):
            # Load prompts
            saved_prompts = app_utils.load_system_prompts()
            prompt_names = list(saved_prompts.keys())
            
            # State management for selection
            if "selected_persona_name" not in st.session_state:
                st.session_state.selected_persona_name = prompt_names[0]
            
            # Dropdown to select preset
            selected_name = st.selectbox(
                "Load Preset", 
                prompt_names, 
                index=prompt_names.index(st.session_state.selected_persona_name) if st.session_state.selected_persona_name in prompt_names else 0,
                key="persona_selector"
            )

            # Update text area if selection changes
            if selected_name != st.session_state.get("last_loaded_persona"):
                st.session_state.custom_system_prompt = saved_prompts[selected_name]
                st.session_state.last_loaded_persona = selected_name
                st.session_state.selected_persona_name = selected_name

            # The editable text area
            custom_system_prompt = st.text_area(
                "Current Instructions", 
                key="custom_system_prompt", # Binds to st.session_state.custom_system_prompt
                height=150,
                help="Edit the behavior of the AI."
            )

            # Save functionality
            with st.popover("💾 Save New Persona"):
                new_persona_name = st.text_input("Name", placeholder="e.g., Python Auditor")
                if st.button("Save Preset", use_container_width=True):
                    if new_persona_name and custom_system_prompt:
                        app_utils.save_system_prompt(new_persona_name, custom_system_prompt)
                        st.success(f"Saved '{new_persona_name}'")
                        st.rerun()


        with st.expander("📚 Knowledge Base", expanded=True):
            rag_type = st.selectbox(
                "Strategy", ["rag", "lightrag", "kag", "direct"],
                format_func=lambda x: {"rag": "Standard RAG", "lightrag": "LightRAG", "kag": "KAG", "direct": "LLM Only"}.get(x, x)
            )
            is_direct = (rag_type == "direct")
            
            col_db_select, col_db_refresh = st.columns([0.85, 0.15])
            
            # 1. Fetch the DB list (this happens fresh every script run)
            dbs = list_available_dbs(rag_type) if not is_direct else [DEFAULT_DB_NAME]
            
            # 2. Render Dropdown
            with col_db_select:
                selected_db = st.selectbox("Database", dbs if dbs else ["No DB"], disabled=is_direct, label_visibility="collapsed")
            
            # 3. Render Refresh Button
            with col_db_refresh:
                # Clicking this triggers a script rerun, which re-executes step #1 above
                if st.button("🔄", help="Refresh Database List", use_container_width=True):
                    st.rerun()
            
            col_opt, col_hyb = st.columns(2)
            optimize = col_opt.checkbox("Refine Query", value=True, disabled=is_direct)
            hybrid = col_hyb.checkbox("Hybrid", value=True, disabled=is_direct)

        with st.expander("🎛️ Parameters & Context", expanded=False):
            top_k = st.slider("Retrieval Depth (Docs)", 1, 20, 5, disabled=is_direct)
            history_limit = st.slider("Chat Memory (Msgs)", 0, 20, 6)
            
            temp = st.slider("Temperature", 0.0, 1.0, 0.7)
            curr_msgs = current_session["messages"] if "active_session_id" in st.session_state else []
            # Calculate prompt tokens locally for immediate feedback
            sys_tokens_est = app_utils.count_tokens(custom_system_prompt) if custom_system_prompt else 0
            
            app_utils.render_token_estimator(
                top_k if not is_direct else 0,
                history_limit, 
                curr_msgs, 
                ctx_window,
                sys_tokens_est
            )

    # ==========================================
    # MAIN CONTENT
    # ==========================================
    
    if "active_session_id" not in st.session_state: st.stop()

    # Header & Title Edit
    col_h1, col_h2 = st.columns([3, 1])
    col_h1.subheader(f"💬 {current_session.get('title', 'Untitled')}")
    new_title = col_h2.text_input("Rename", value=current_session.get("title"), label_visibility="collapsed")
    if new_title != current_session.get("title"):
        session_manager.update_title(current_session["id"], new_title)
        st.rerun()

    # Context Injection
    with st.expander("📎 Add Session Context (Upload File)"):
        uploaded_file = st.file_uploader("Upload PDF/TXT/Code", type=['pdf', 'txt', 'py', 'md', 'json'])
        if uploaded_file:
            if uploaded_file.name not in current_session.get("temp_context", ""):
                with st.spinner("Processing..."):
                    text_content = app_utils.parse_uploaded_file(uploaded_file)
                    current_session["temp_context"] = (current_session.get("temp_context", "") + text_content)
                    session_manager.save_session(current_session)
                    st.success("File added to context!")

    # Warmup
    if not app_utils.warm_up_resources(): st.stop()

    # Render History
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
        # Calculate budget: Context Window - (Safety Margin + Max Output + Retrieval Estimate)
        safe_ctx_limit = ctx_window - 1000 - (top_k * 250) 
        
        full_history = [{"role": m["role"], "content": m["content"]} for m in current_session["messages"][:-1]]
        
        # Smart Pruning: Use tokens, not just message count
        rag_history = app_utils.smart_prune_history(full_history, safe_ctx_limit)
        
        # Enforce the user slider as a hard cap if set lower
        if history_limit < len(rag_history):
            rag_history = rag_history[-history_limit:]

        final_query = prompt
        if current_session.get("temp_context"):
            final_query = f"Session Context:\n{current_session['temp_context']}\n\nQuery: {prompt}"

        # 3. Execution with Granular Status
        with st.chat_message("assistant"):
            start_time = time.time()
            response_container = st.empty()
            
        # Use st.status for a collapsible "Thinking" state
        with st.status("🧠 Orchestrating...", expanded=True) as status:
            try:
                # Base arguments shared by all strategies
                common_args = {
                    "query_text": final_query, 
                    "llm_config": {
                        "provider": provider, "modelName": selected_model, 
                        "apiKey": api_key, "api_url": local_url if provider == "local" else None,
                        "temperature": temp, "system_prompt": custom_system_prompt,
                        "context_window": ctx_window 
                    }, 
                    "conversation_history": rag_history
                }
                
                if rag_type == 'direct':
                    status.write("Direct LLM query (no retrieval)...")
                    # query_direct only accepts the common args
                    response = query_direct(**common_args)
                elif selected_db:
                    status.write(f"🔍 Retrieving from **{selected_db}** ({rag_type})...")
                    strategies = {'rag': query_rag, 'lightrag': query_lightrag, 'kag': query_kag}
                    # RAG strategies accept common args + specific kwargs
                    response = strategies[rag_type](
                        **common_args, 
                        top_k=top_k, # top_k is only for RAG
                        rag_type=rag_type, 
                        db_name=selected_db, 
                        optimize=optimize, 
                        hybrid=hybrid
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
            except Exception as e:
                if "StopException" in type(e).__name__:
                    status.update(label="⏹️ Cancelled by User", state="error", expanded=False)
                    st.stop()
                
                # Log actual errors
                status.update(label="❌ Error", state="error")
                st.error(f"Pipeline Error: {str(e)}")
                logging.error(e, exc_info=True)
                st.stop()
                    
                    
            # 4. Parse & Display
            text = response.get("text", "")
            sources = response.get("sources", [])
            tokens_est = response.get("estimated_context_tokens", 0)
            clean_text, reasoning = app_utils.parse_reasoning(text)

            # Display Reasoning separately if available (DeepSeek/R1 style)
            if reasoning:
                with st.expander("💭 Internal Thought Process", expanded=False):
                    st.markdown(reasoning)

            response_container.markdown(clean_text)

            # Smart Source Display
            if sources:
                with st.expander(f"📚 Cited Sources ({len(sources)})", expanded=False):
                    # Deduplicate sources for display
                    unique_sources = list(set(sources))
                    for i, src in enumerate(unique_sources):
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
            
            # Auto-rename untitled sessions
            if len(current_session["messages"]) == 2:
                # Generate a better title using a quick slice
                current_session["title"] = (clean_text[:30] + "...") if len(clean_text) > 30 else clean_text
            
            session_manager.save_session(current_session)
            
if __name__ == "__main__":
    main()