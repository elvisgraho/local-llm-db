import streamlit as st
import app_utils
from query.global_vars import LOCAL_LLM_API_URL
from query.database_paths import DATABASE_DIR
from pathlib import Path

# --- Constants ---
STRATEGY_MAP = {
    "Direct Chat": "direct",
    "Standard RAG": "rag",
    "LightRAG": "lightrag"
}

def get_dbs_for_strategy(strategy_key: str) -> list:
    """Scans the specific subdirectory for available databases."""
    if not strategy_key or strategy_key == 'direct':
        return []
    target_path = Path(DATABASE_DIR) / strategy_key
    if not target_path.exists():
        return []
    try:
        return [d.name for d in target_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    except OSError:
        return []

def render_sidebar(state_manager, session_data=None):
    """
    Renders the sidebar controls.
    """
    with st.sidebar:
        st.title("🗂️ Chats")
        
        # --- 1. Session Management ---
        _render_session_list(state_manager)
        
        st.divider()
        
        # --- 2. Configuration ---
        st.header("⚙️ Configuration")
        
        # A. LLM Settings
        llm_config = _render_llm_settings(state_manager)
        
        st.divider()

        # B. Embedding Settings
        emb_config = _render_embedding_settings(state_manager)

        # C. RAG Strategy (Architecture & DB)
        rag_strategy_config = _render_rag_strategy(state_manager)
        
        # D. System Prompt
        system_config = _render_system_prompt()

        # E. RAG Parameters
        is_direct = (rag_strategy_config["rag_type"] == "direct")
        rag_params_config = _render_rag_params(is_direct)
        
        # Merge Configs
        full_rag_config = {**rag_strategy_config, **rag_params_config}
        llm_config["system_prompt"] = system_config
        
        full_config = {
            "llm_config": llm_config,
            "rag_config": full_rag_config,
            "embedding_config": emb_config
        }

        # --- 3. Token Estimator ---
        if session_data:
            st.divider()
            render_usage_stats(session_data, full_config, state_manager)
        
        return full_config

def _render_session_list(state_manager):
    """
    Renders the session list using an 'Inline State Swap' for deletion.
    This is robust, strictly linear, and prevents layout glitches.
    """
    # 1. Ensure we have a tracker for which item is being deleted
    if "confirm_delete_id" not in st.session_state:
        st.session_state.confirm_delete_id = None

    if st.button("➕ New Chat", type="primary", use_container_width=True):
        state_manager.create_new_session()
        st.rerun()

    sessions = state_manager.get_session_list()
    active_id = state_manager.get_active_session_id()
    
    if not sessions:
        st.caption("No chat history.")
        return

    display_sessions = sessions[:15]
    
    for s in display_sessions: 
        # --- MODE A: CONFIRMATION MODE ---
        # If this specific chat is clicked for deletion, swap the row for a confirmation box
        if st.session_state.confirm_delete_id == s["id"]:
            with st.container(border=True):
                st.markdown(f":red[**Delete '{s.get('title', 'Untitled')}'?**]")
                col_yes, col_no = st.columns(2)
                
                # YES: Delete and Reset
                if col_yes.button("Confirm", key=f"yes_{s['id']}", type="primary", use_container_width=True):
                    state_manager.delete_session(s["id"])
                    st.session_state.confirm_delete_id = None
                    st.rerun()
                
                # NO: Just Reset
                if col_no.button("Cancel", key=f"no_{s['id']}", use_container_width=True):
                    st.session_state.confirm_delete_id = None
                    st.rerun()

        # --- MODE B: NORMAL ROW MODE ---
        else:
            col_name, col_del = st.columns([0.85, 0.15])
            
            clean_title = s.get("title", "Untitled")
            label = clean_title[:22] + "..." if len(clean_title) > 25 else clean_title
            
            if s["id"] == active_id:
                label = f"📂 {label}"
            
            # Select Chat
            if col_name.button(label, key=f"btn_{s['id']}", use_container_width=True, help=s.get("title")):
                state_manager.set_active_session(s["id"])
                st.rerun()
            
            # Trigger Delete Mode
            if col_del.button("🗑️", key=f"trig_{s['id']}"):
                st.session_state.confirm_delete_id = s["id"]
                st.rerun()

def _render_llm_settings(state_manager):
    """Renders Chat Model settings."""
    with st.expander("🤖 Chat Model", expanded=True):
        provider = st.selectbox("Provider", ["local", "gemini"], index=0)
        
        if provider == "local":
            st.text_input(
                "Local API URL", 
                value=st.session_state.get("llm_url", LOCAL_LLM_API_URL),
                key="llm_url_input",
                on_change=lambda: setattr(st.session_state, 'llm_url', st.session_state.llm_url_input)
            )
            
            c_mod, c_ref = st.columns([0.85, 0.15])
            with c_ref:
                if st.button("🔄", key="ref_llm", help="Refresh Chat Models"):
                    url = st.session_state.get("llm_url", LOCAL_LLM_API_URL)
                    models = app_utils.fetch_available_models(url)
                    chat_models = [m for m in models if "embed" not in m.lower()]
                    
                    if chat_models:
                        st.session_state.fetched_models = chat_models
                        st.session_state.llm_selector = chat_models[0]
                    else:
                        st.session_state.fetched_models = ["(No models found)"]

            with c_mod:
                opts = st.session_state.get("fetched_models", ["local-model"])
                
                # --- STATE VALIDATION ---
                current_val = st.session_state.get("llm_selector")
                if current_val not in opts:
                    st.session_state.llm_selector = opts[0]

                # --- RENDER ---
                # ERROR FIX: Removed 'index=idx'.
                selected_model = st.selectbox(
                    "Model", 
                    options=opts, 
                    key="llm_selector"
                )

            return {
                "provider": "local",
                "model_name": selected_model,
                "api_key": "not-needed",
                "local_url": st.session_state.llm_url,
                "context_window": st.number_input("Context Window", 2048, 128000, 12288, step=1024),
                "temperature": st.slider("Temperature", 0.0, 1.0, 0.7)
            }
        else:
            return {
                "provider": "gemini",
                "model_name": st.text_input("Model Name", value="gemini-1.5-pro"),
                "api_key": st.text_input("Gemini API Key", type="password"),
                "local_url": None,
                "context_window": 32000,
                "temperature": 0.7
            }

def _render_embedding_settings(state_manager):
    """
    Renders Embedding Model settings.
    Includes Smart Recovery to prioritize 'embed' models if state is lost.
    """
    with st.expander("🧠 Embeddings", expanded=False):
        
        # 1. URL Input
        st.text_input(
            "Emb API URL", 
            value=st.session_state.get("emb_url", LOCAL_LLM_API_URL),
            key="emb_url_input",
            on_change=lambda: setattr(st.session_state, 'emb_url', st.session_state.emb_url_input)
        )

        c_emb, c_ref = st.columns([0.85, 0.15])
        
        # 2. Refresh Button
        with c_ref:
            if st.button("🔄", key="ref_emb", help="Refresh Embed Models"):
                url = st.session_state.get("emb_url", LOCAL_LLM_API_URL)
                models = app_utils.fetch_available_models(url)
                if models:
                    st.session_state.fetched_emb_models = models
                    # SMART SELECT: Force "embed" model on manual refresh
                    best = next((m for m in models if 'embed' in m.lower()), models[0])
                    st.session_state.emb_model_selector = best
                else:
                    st.session_state.fetched_emb_models = []
                    st.toast("No models found.", icon="⚠️")

        # 3. Selectbox with SMART VALIDATION
        with c_emb:
            opts = st.session_state.get("fetched_emb_models", [])
            if not opts:
                opts = ["(Click 🔄 to fetch)"]
            
            # --- FIX STARTS HERE ---
            current_val = st.session_state.get("emb_model_selector")
            
            # If the current value is invalid (not in list) or missing...
            if current_val not in opts:
                # ...Do NOT just pick opts[0]. 
                # Scan the list again for the best "embed" model.
                best_recovery = next((m for m in opts if 'embed' in m.lower()), opts[0])
                st.session_state.emb_model_selector = best_recovery

            # --- RENDER ---
            selected_emb = st.selectbox(
                "Emb Model", 
                options=opts, 
                key="emb_model_selector"
            )
            
    return {
        "provider": "local",
        "url": st.session_state.emb_url,
        "model_name": selected_emb if selected_emb != "(Click 🔄 to fetch)" else None
    }


def _render_rag_strategy(state_manager):
    """Renders only the Architecture and Database selection."""
    st.subheader("🗄️ Knowledge Base")
    
    curr_strat = st.session_state.get("rag_strategy", "Standard RAG")
    # Safety check if strat key is valid
    if curr_strat not in STRATEGY_MAP: 
        curr_strat = "Standard RAG"
        
    idx = list(STRATEGY_MAP.keys()).index(curr_strat)

    selected_strat_label = st.selectbox(
        "Architecture", 
        list(STRATEGY_MAP.keys()),
        index=idx,
        key="rag_type_selector",
        on_change=lambda: setattr(st.session_state, 'rag_strategy', st.session_state.rag_type_selector)
    )
    
    rag_key = STRATEGY_MAP[selected_strat_label]
    is_direct = (rag_key == "direct")
    selected_db = None
    
    if not is_direct:
        available_dbs = get_dbs_for_strategy(rag_key)
        if available_dbs:
            prev_db = st.session_state.get("persisted_db")
            db_idx = available_dbs.index(prev_db) if prev_db in available_dbs else 0
            
            c_db, c_ref = st.columns([0.85, 0.15])
            with c_db:
                selected_db = st.selectbox("Select Database", available_dbs, index=db_idx, key="db_selector",
                    on_change=lambda: setattr(st.session_state, 'persisted_db', st.session_state.db_selector))
            with c_ref:
                if st.button("🔄", key="ref_db"): st.rerun()
        else:
            st.warning(f"No databases found in /{rag_key}")
            
    return {
        "rag_type": rag_key,
        "db_name": selected_db
    }

def _render_system_prompt():
    """Renders System Persona."""
    with st.expander("🎭 System Persona", expanded=False):
        prompts = app_utils.load_system_prompts()
        names = ["Custom"] + list(prompts.keys())
        # Default to Custom if logic fails
        idx = 1 if len(names) > 1 else 0
        sel = st.selectbox("Select Persona", names, index=idx)
        
        if sel == "Custom":
            return st.text_area("Custom Instructions", height=150)
        return st.text_area("Instructions", value=prompts[sel], height=150)

def _render_rag_params(is_direct):
    """Renders Retrieval Parameters, reading defaults from Session State."""
    with st.expander("🎛️ Retrieval Params", expanded=False):
        default_k = st.session_state.get("rag_top_k", 10)
        default_hist = st.session_state.get("rag_history_limit", 6)

        # The 'key' argument automatically binds to st.session_state["rag_top_k"]
        top_k = st.slider("Top-K (Chunks)", 1, 100, 10, disabled=is_direct, key="rag_top_k")
        history_limit = st.slider("Chat History (Msgs)", 0, 30, 6, key="rag_history_limit")

        # --- Chunk Expansion Settings ---
        st.markdown("---")
        st.markdown("**🔗 Contextual Expansion**")

        enable_expansion = st.toggle(
            "Enable Chunk Expansion",
            value=True,
            key="rag_enable_chunk_expansion",
            disabled=is_direct,
            help="Retrieve adjacent chunks for code-heavy content to provide complete context (e.g., vulnerable code + fix together)"
        )

        if enable_expansion:
            expansion_window = st.slider(
                "Context Window",
                min_value=1,
                max_value=5,
                value=1,
                disabled=is_direct or not enable_expansion,
                key="rag_chunk_expansion_window",
                help="How many chunks before/after to retrieve (1 = ±1, 2 = ±2, 3 = ±3)"
            )
            code_only = st.toggle(
                "Code Only",
                value=True,
                disabled=is_direct or not enable_expansion,
                key="rag_chunk_expansion_code_only",
                help="Only expand chunks with code_languages metadata"
            )
        else:
            # Set defaults when disabled
            expansion_window = 1
            code_only = True

        st.markdown("---")

        hybrid = st.toggle("Hybrid Search", value=True, key="rag_hybrid",
            disabled=is_direct, help="Allows LLM to deviate from source material.")
        verify = st.toggle("Verify Answer", value=False, key="rag_verify",
            help="After initial LLM response, LLM will run again to verify and correct the response")
        rewrite = st.toggle("Rewrite Queries", value=False, key="rag_rewrite",
            help="Automatically uses the LLM to refine your prompt.")

    return {
        "top_k": top_k,
        "history_limit": history_limit,
        "hybrid": hybrid,
        "rewrite": rewrite,
        "verify": verify,
        "enable_chunk_expansion": enable_expansion,
        "chunk_expansion_window": expansion_window,
        "chunk_expansion_code_only": code_only
    }

def render_usage_stats(session_data, config, state_manager):
    """
    Renders the visual Token Usage bar.
    
    FIX: Now prioritizes the LIVE CONFIG (Sliders) for the bar visualization
    so the user can see how changes affect the budget for the NEXT query.
    """
    if not session_data: return

    msgs = session_data.get("messages", [])
    
    # 1. Get Live Config Values (The "Plan")
    # We use the config object which is directly tied to the sidebar sliders
    live_top_k = config["rag_config"]["top_k"]
    live_hist_limit = config["rag_config"]["history_limit"]
    
    # If Direct mode, Retrieval is 0
    if config["rag_config"]["rag_type"] == "direct":
        live_top_k = 0

    # 2. Calculate "Planned" Usage (for the visual bar)
    # This ensures the bar moves immediately when you drag the slider
    
    sys_tokens = app_utils.count_tokens(config["llm_config"].get("system_prompt", ""))
    
    # Render the estimator using the LIVE parameters
    app_utils.render_token_estimator(
        top_k=live_top_k,
        history_limit=live_hist_limit,
        current_messages=msgs,
        context_window=config["llm_config"]["context_window"],
        sys_tokens=sys_tokens,
        # Passing None forces the estimator to use the 'live_top_k' estimation logic
        # instead of locking to the previous message's metadata.
        actual_retrieval_tokens=None 
    )

    # 3. Optional: Show "Last Actuals" as text only (for reference)
    last_actual = 0
    if msgs and msgs[-1]["role"] == "assistant":
        usage = msgs[-1].get("usage", {})
        last_actual = usage.get("retrieval_tokens", 0)
        
    if last_actual > 0:
        st.caption(f"📝 *Last query used {last_actual} retrieval tokens*")