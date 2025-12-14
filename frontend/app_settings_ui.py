import streamlit as st
from pathlib import Path
from query.global_vars import LOCAL_LLM_API_URL
from query.database_paths import DATABASE_DIR
from app_utils import fetch_available_models, load_system_prompts

# --- Constants ---
STRATEGY_MAP = {
    "Direct Chat": "direct",
    "Standard RAG": "rag",
    "LightRAG": "lightrag",
    "KAG (Graph)": "kag"
}

# Default LLM only (Embedding defaults to empty/discovered)
DEFAULT_LLM_MODEL = "local-model"

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

# --- State Management Callbacks ---

def cb_sync_llm_url():
    """Forces sync of LLM URL input to backing variable."""
    st.session_state.llm_url = st.session_state.llm_url_input

def cb_sync_emb_url():
    """Forces sync of Emb URL input to backing variable."""
    st.session_state.emb_url = st.session_state.emb_url_input

def cb_refresh_llm_models():
    """
    Refreshes LLM models, FILTERS OUT embedding models, and updates selection.
    """
    # Ensure we use the latest input value
    url = st.session_state.get("llm_url_input", LOCAL_LLM_API_URL)
    raw_models = fetch_available_models(url)
    
    # FILTER: Exclude models that have 'embed' in the name
    models = [m for m in raw_models if "embed" not in m.lower()]
    
    if models:
        st.session_state.fetched_models = models
        # Keep current selection if valid
        current = st.session_state.get("llm_selector", "")
        if current not in models:
            st.session_state.llm_selector = models[0]
        st.toast(f"LLM: Found {len(models)} models (filtered)", icon="✅")
    else:
        st.toast("LLM: No chat models found or API down.", icon="⚠️")

def cb_refresh_emb_models():
    """
    Refreshes Embedding models and applies Auto-Select logic.
    """
    # Ensure we use the latest input value
    url = st.session_state.get("emb_url_input", LOCAL_LLM_API_URL)
    models = fetch_available_models(url)
    
    if models:
        st.session_state.fetched_emb_models = models
        
        # --- Auto-Select Logic ---
        # 1. Look for 'embed' in the name
        # 2. Fallback to first available
        best_match = None
        for m in models:
            if "embed" in m.lower():
                best_match = m
                break
        
        if best_match:
            st.session_state.emb_model_selector = best_match
            st.toast(f"Auto-selected: {best_match}", icon="✨")
        else:
            st.session_state.emb_model_selector = models[0]
            st.toast(f"Embeddings: Found {len(models)} models", icon="✅")
    else:
        st.toast("Embeddings: Connection failed.", icon="⚠️")

def cb_update_rag_strategy():
    st.session_state.rag_strategy = st.session_state.rag_type_selector

def cb_update_db_selection():
    st.session_state.persisted_db = st.session_state.db_selector

# --- Main Render ---
def render_settings_sidebar():
    """
    Renders the Settings Sidebar and returns the configuration dict.
    """
    # 1. Initialize State 
    st.session_state.setdefault("llm_url", LOCAL_LLM_API_URL)
    st.session_state.setdefault("emb_url", LOCAL_LLM_API_URL)
    st.session_state.setdefault("fetched_models", [DEFAULT_LLM_MODEL])
    # Empty default for embeddings (Removed hardcoded Nomic model)
    st.session_state.setdefault("fetched_emb_models", [])
    st.session_state.setdefault("rag_strategy", "Standard RAG")
    st.session_state.setdefault("persisted_db", None)
    
    # Initialize selectors
    st.session_state.setdefault("llm_selector", DEFAULT_LLM_MODEL)
    st.session_state.setdefault("emb_model_selector", None)

    # 2. SYNC BLOCK (Extra Safety)
    # Ensure backing vars match inputs before rendering
    if "llm_url_input" in st.session_state:
        st.session_state.llm_url = st.session_state.llm_url_input
    if "emb_url_input" in st.session_state:
        st.session_state.emb_url = st.session_state.emb_url_input

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ==========================================
        # 2. MODEL SETTINGS
        # ==========================================
        with st.expander("🤖 Model Settings", expanded=True):
            provider = st.selectbox("Provider", ["local", "gemini"], index=0)
            
            if provider == "local":
                # URL Input with explicit on_change sync
                st.text_input(
                    "Local API URL", 
                    value=st.session_state.llm_url,
                    key="llm_url_input",
                    on_change=cb_sync_llm_url 
                )
                
                c_mod, c_ref = st.columns([0.85, 0.15])
                with c_ref:
                    st.button("🔄", help="Refresh LLM Models", on_click=cb_refresh_llm_models, key="btn_ref_llm")
                
                with c_mod:
                    # Logic to handle empty or invalid lists
                    opts = st.session_state.fetched_models
                    if not opts: opts = ["(No models found)"]
                    
                    # Validate selection
                    if st.session_state.llm_selector not in opts:
                        st.session_state.llm_selector = opts[0]

                    st.selectbox("Model", options=opts, key="llm_selector")
            
            elif provider == "gemini":
                st.text_input("Gemini API Key", type="password", key="gemini_key")
                st.text_input("Model Name", value="gemini-1.5-pro", key="gemini_model")

            # Shared Params
            ctx_window = st.number_input("Context Window", 2048, 128000, 8192, step=1024)
            st.divider()
            
            # --- EMBEDDING SETTINGS ---
            st.caption("Embedding Model")
            st.selectbox("Emb Provider", ["local"], index=0, disabled=True)
            
            # URL Input with explicit on_change sync
            st.text_input(
                "Emb API URL", 
                value=st.session_state.emb_url,
                key="emb_url_input",
                on_change=cb_sync_emb_url
            )

            c_emb_mod, c_emb_ref = st.columns([0.85, 0.15])
            with c_emb_ref:
                st.button("🔄", help="Refresh & Auto-Select", on_click=cb_refresh_emb_models, key="btn_ref_emb")

            with c_emb_mod:
                # Handle empty embedding list (Removed hardcoded default)
                emb_opts = st.session_state.fetched_emb_models
                placeholder_opt = False
                
                if not emb_opts:
                    emb_opts = ["(Click 🔄 to fetch)"]
                    placeholder_opt = True
                
                # Logic to determine index or key binding
                # If it's a placeholder, we don't bind it to the selector key permanently if possible,
                # or we just let it be.
                
                if not placeholder_opt and st.session_state.emb_model_selector not in emb_opts:
                    st.session_state.emb_model_selector = emb_opts[0]
                
                st.selectbox(
                    "Emb Model", 
                    options=emb_opts,
                    disabled=placeholder_opt,
                    key="emb_model_selector" 
                )

        # ==========================================
        # 3. RAG STRATEGY & DATABASE
        # ==========================================
        st.subheader("🗄️ Knowledge Base")
        
        curr_strat = st.session_state.rag_strategy
        try:
            idx_strat = list(STRATEGY_MAP.keys()).index(curr_strat)
        except ValueError:
            idx_strat = 0

        st.selectbox(
            "Architecture", 
            list(STRATEGY_MAP.keys()),
            index=idx_strat,
            key="rag_type_selector",
            on_change=cb_update_rag_strategy
        )
        
        rag_key = STRATEGY_MAP[st.session_state.rag_type_selector]
        is_direct = (rag_key == "direct")

        selected_db = None

        if not is_direct:
            available_dbs = get_dbs_for_strategy(rag_key)
            if available_dbs:
                prev_db = st.session_state.persisted_db
                
                if prev_db in available_dbs:
                    db_idx = available_dbs.index(prev_db)
                else:
                    db_idx = 0
                
                c_db, c_ref_db = st.columns([0.85, 0.15])
                with c_db:
                    st.selectbox(
                        "Select Database", 
                        available_dbs, 
                        index=db_idx,
                        key="db_selector",
                        on_change=cb_update_db_selection,
                        label_visibility="collapsed"
                    )
                    selected_db = st.session_state.db_selector
                with c_ref_db:
                    if st.button("🔄", key="ref_db_list"): st.rerun()
            else:
                st.warning(f"No databases found in /{rag_key}")
                if st.button("🔄", key="ref_db_empty"): st.rerun()

        # Options
        col_opt1, col_opt2 = st.columns(2)
        use_hybrid = col_opt1.checkbox("Hybrid Search", value=True, disabled=is_direct)
        use_verify = col_opt2.checkbox("Verify Response", value=False, disabled=is_direct)

        # ==========================================
        # 4. SYSTEM PROMPTS & PARAMS
        # ==========================================
        with st.expander("🎭 System Persona", expanded=False):
            prompts = load_system_prompts()
            prompt_names = ["Custom"] + list(prompts.keys())
            selected_persona = st.selectbox("Select Persona", prompt_names, index=1)
            
            if selected_persona == "Custom":
                system_prompt = st.text_area("Custom Instructions", height=150)
            else:
                system_prompt = st.text_area("Instructions", value=prompts[selected_persona], height=150)

        with st.expander("🎛️ Parameters", expanded=False):
            top_k = st.slider("Retrieval Depth", 1, 50, 10, disabled=is_direct)
            history_limit = st.slider("Chat Memory", 0, 35, 5)
            temp = st.slider("Temperature", 0.0, 1.0, 0.7)

    # 5. Configuration Output
    
    if provider == "local":
        final_api_key = None
        final_model = st.session_state.get("llm_selector", DEFAULT_LLM_MODEL)
        final_url = st.session_state.get("llm_url_input", LOCAL_LLM_API_URL)
    else:
        final_api_key = st.session_state.get("gemini_key", "")
        final_model = st.session_state.get("gemini_model", "gemini-1.5-pro")
        final_url = None

    # Handle embedding model output if not selected yet
    final_emb_model = st.session_state.get("emb_model_selector")
    if not final_emb_model or final_emb_model == "(Click 🔄 to fetch)":
        final_emb_model = None

    return {
        "llm_config": {
            "provider": provider,
            "api_key": final_api_key,
            "model_name": final_model,
            "local_url": final_url,
            "context_window": ctx_window,
            "temperature": temp,
            "system_prompt": system_prompt
        },
        "embedding_config": {
            "provider": "local",
            "url": st.session_state.get("emb_url_input", LOCAL_LLM_API_URL),
            "model_name": final_emb_model
        },
        "rag_config": {
            "rag_type": rag_key,
            "db_name": selected_db,
            "top_k": top_k,
            "hybrid": use_hybrid,
            "verify": use_verify,
            "history_limit": history_limit
        }
    }