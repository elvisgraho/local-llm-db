import streamlit as st
import requests
from query.global_vars import LOCAL_LLM_API_URL
from query.database_paths import DATABASE_DIR
import app_utils

# --- Constants ---
STRATEGY_MAP = {
    "Direct Chat": "direct",
    "Standard RAG": "rag",
    "LightRAG": "lightrag",
    "KAG (Graph)": "kag"
}

# --- Helpers ---
def fetch_available_models(base_url):
    """
    Fetches models from the Local LLM API (LM Studio/Ollama compatible).
    """
    if not base_url: return []
    try:
        clean_url = base_url.rstrip('/')
        if not clean_url.endswith("/v1"): clean_url += "/v1"
        
        response = requests.get(f"{clean_url}/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return [m['id'] for m in data['data']]
            return []
    except Exception:
        return []
    return []

def get_dbs_for_strategy(strategy_key: str) -> list:
    """Scans the specific subdirectory for available databases."""
    if not strategy_key or strategy_key == 'direct':
        return []
    
    target_dir = DATABASE_DIR / strategy_key
    if not target_dir.exists():
        return []
        
    return [d.name for d in target_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

# --- Callbacks ---
def _update_rag_type():
    st.session_state.rag_strategy = st.session_state.rag_type_selector

def _update_selected_db():
    st.session_state.persisted_db = st.session_state.db_selector

def render_settings_sidebar():
    """
    Renders the Settings Sidebar and returns the configuration dict.
    """
    # 1. Initialize State
    if "llm_url" not in st.session_state: 
        st.session_state.llm_url = LOCAL_LLM_API_URL
    if "emb_url" not in st.session_state:
        st.session_state.emb_url = LOCAL_LLM_API_URL
    if "fetched_models" not in st.session_state:
        st.session_state.fetched_models = ["local-model"]
    if "rag_strategy" not in st.session_state:
        st.session_state.rag_strategy = "Standard RAG"
    if "persisted_db" not in st.session_state:
        st.session_state.persisted_db = None

    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ==========================================
        # 2. MODEL SETTINGS (Restored Functionality)
        # ==========================================
        with st.expander("🤖 Model Settings", expanded=True):
            # Provider Selection (NO OpenAI)
            provider = st.selectbox("Provider", ["local", "gemini"], index=0)
            
            api_key = None
            local_url = None
            selected_model = "local-model"

            # --- LOCAL PROVIDER LOGIC ---
            if provider == "local":
                # URL Input
                local_url = st.text_input(
                    "Local API URL", 
                    value=st.session_state.llm_url,
                    key="llm_url_input",
                    on_change=lambda: st.session_state.update({"llm_url": st.session_state.llm_url_input})
                )
                
                # Refresh Models Button
                c_mod, c_ref = st.columns([0.85, 0.15])
                with c_ref:
                    if st.button("🔄", help="Refresh Models"):
                        models = fetch_available_models(local_url)
                        if models:
                            st.session_state.fetched_models = models
                            st.toast(f"Found {len(models)} models")
                        else:
                            st.toast("No models found or API down", icon="⚠️")
                
                # Model Selector (Dropdown)
                with c_mod:
                    model_opts = st.session_state.fetched_models
                    selected_model = st.selectbox("Model", model_opts, index=0)

            # --- GEMINI PROVIDER LOGIC ---
            elif provider == "gemini":
                api_key = st.text_input("Gemini API Key", type="password")
                selected_model = st.text_input("Model Name", value="gemini-1.5-pro")

            ctx_window = st.number_input("Context Window", 2048, 128000, 8192, step=1024)

            st.divider()
            
            # --- Embeddings ---
            st.caption("Embedding Model")
            emb_provider = st.selectbox("Emb Provider", ["local"], index=0, disabled=True)
            emb_url = st.text_input("Emb API URL", value=st.session_state.emb_url)
            emb_model = st.text_input("Emb Model Name", value="text-embedding-nomic-embed-text-v1.5")

        # ==========================================
        # 3. RAG STRATEGY & DATABASE (New Logic)
        # ==========================================
        st.subheader("🗄️ Knowledge Base")
        
        # Sync Strategy
        curr_strat = st.session_state.rag_strategy
        idx_strat = list(STRATEGY_MAP.keys()).index(curr_strat) if curr_strat in STRATEGY_MAP else 1
        
        selected_strategy_label = st.selectbox(
            "Architecture", 
            list(STRATEGY_MAP.keys()),
            index=idx_strat,
            key="rag_type_selector",
            on_change=_update_rag_type
        )
        rag_key = STRATEGY_MAP[selected_strategy_label]
        is_direct = (rag_key == "direct")

        # Dynamic DB List based on Strategy
        available_dbs = get_dbs_for_strategy(rag_key)
        selected_db = None

        if not is_direct:
            if available_dbs:
                prev_db = st.session_state.persisted_db
                db_idx = available_dbs.index(prev_db) if prev_db in available_dbs else 0
                
                c_db, c_ref_db = st.columns([0.85, 0.15])
                with c_db:
                    selected_db = st.selectbox(
                        "Select Database", 
                        available_dbs, 
                        index=db_idx,
                        key="db_selector",
                        on_change=_update_selected_db,
                        label_visibility="collapsed"
                    )
                with c_ref_db:
                    if st.button("🔄", key="ref_db_list"): st.rerun()
            else:
                st.warning(f"No {selected_strategy_label} DBs.")
                if st.button("🔄"): st.rerun()

        col_opt1, col_opt2 = st.columns(2)
        use_hybrid = col_opt1.checkbox("Hybrid Search", value=True, disabled=is_direct)
        use_verify = col_opt2.checkbox("Verify respone", value=False, disabled=is_direct)

        # ==========================================
        # 4. SYSTEM PROMPTS & PARAMS
        # ==========================================
        with st.expander("🎭 System Persona", expanded=False):
            prompts = app_utils.load_system_prompts()
            prompt_names = ["Custom"] + list(prompts.keys())
            selected_persona = st.selectbox("Select Persona", prompt_names, index=1)
            
            if selected_persona == "Custom":
                system_prompt = st.text_area("Custom Instructions", height=150)
            else:
                system_prompt = st.text_area("Instructions", value=prompts[selected_persona], height=150)

        with st.expander("🎛️ Parameters", expanded=False):
            top_k = st.slider("Retrieval Depth", 1, 25, 10, disabled=is_direct)
            history_limit = st.slider("Chat Memory", 0, 35, 5)
            temp = st.slider("Temperature", 0.0, 1.0, 0.7)

    return {
        "llm_config": {
            "provider": provider,
            "api_key": api_key,
            "model_name": selected_model,
            "local_url": local_url,
            "context_window": ctx_window,
            "temperature": temp,
            "system_prompt": system_prompt
        },
        "embedding_config": {
            "provider": emb_provider,
            "url": emb_url,
            "model_name": emb_model
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