import streamlit as st
import time
from query.global_vars import LOCAL_LLM_API_URL
from query.database_paths import list_available_dbs, DEFAULT_DB_NAME
import app_utils

# --- Callbacks for Immediate Persistence ---
def _update_llm_url():
    st.session_state.llm_url = st.session_state.llm_url_input

def _update_emb_url():
    st.session_state.emb_url = st.session_state.emb_url_input

def _update_emb_model():
    st.session_state.emb_model = st.session_state.emb_selector

def _update_rag_type():
    """Persist strategy selection immediately."""
    st.session_state.rag_strategy = st.session_state.rag_type_selector

def _update_selected_db():
    """Persist DB selection immediately."""
    st.session_state.persisted_db = st.session_state.db_selector

def render_settings_sidebar():
    """
    Renders the Settings Sidebar and returns the configuration dict.
    Uses callbacks to ensure UI state survives reruns.
    """
    
    # ==========================================
    # 1. INITIALIZE PERSISTENT STATE
    # ==========================================
    if "llm_url" not in st.session_state: 
        st.session_state.llm_url = LOCAL_LLM_API_URL
    
    if "emb_url" not in st.session_state: 
        st.session_state.emb_url = LOCAL_LLM_API_URL
    if "emb_model" not in st.session_state: 
        st.session_state.emb_model = "text-embedding-nomic-embed-text-v1.5"
    
    # Knowledge Base Persistence
    if "rag_strategy" not in st.session_state:
        st.session_state.rag_strategy = "rag" # Default
    if "persisted_db" not in st.session_state:
        st.session_state.persisted_db = DEFAULT_DB_NAME

    # ==========================================
    # 2. AUTO-SELECT SINGLE KNOWLEDGE BASE
    # ==========================================
    # If the user has exactly ONE db across all types, auto-select it.
    # This runs before widgets are rendered to set the correct defaults.
    
    strategies = ["rag", "lightrag", "kag"]
    # Check what exists
    found_map = {}
    total_dbs = 0
    
    for s in strategies:
        dbs = list_available_dbs(s)
        if dbs:
            found_map[s] = dbs
            total_dbs += len(dbs)
            
    # Logic: If exactly 1 type has DBs, and that type has exactly 1 DB
    if len(found_map) == 1 and total_dbs == 1:
        target_strat = list(found_map.keys())[0]
        target_db = found_map[target_strat][0]
        
        # Only override if we aren't currently on "Direct" or already selected
        if st.session_state.rag_strategy != "direct":
            if st.session_state.rag_strategy != target_strat or st.session_state.persisted_db != target_db:
                st.session_state.rag_strategy = target_strat
                st.session_state.persisted_db = target_db

    config = {}

    with st.expander("🔌 LLM & Embedding Settings", expanded=True):
        
        # ------------------------------------------
        # A. CHAT MODEL SETTINGS
        # ------------------------------------------
        st.caption("🗣️ Chat Model")
        provider = st.radio("Provider", ["local", "gemini"], horizontal=True, label_visibility="collapsed")
        
        # Init outputs
        api_key = None 
        selected_model = ""
        local_url = LOCAL_LLM_API_URL
        ctx_window = 8192

        if provider == "gemini":
            api_key = st.text_input("Gemini API Key", type="password")
            selected_model = st.text_input("Gemini Model", value="gemini-1.5-flash")
            ctx_window = 32000
        elif provider == "local":
            local_url = st.text_input(
                "LLM API URL", 
                value=st.session_state.llm_url, 
                key="llm_url_input",
                on_change=_update_llm_url
            )

            col_mod, col_ref = st.columns([0.85, 0.15])
            with col_ref:
                if st.button("🔄", key="refresh_chat", help="Fetch Chat Models"):
                    st.cache_data.clear()
                    st.rerun()
            
            with col_mod:
                # FILTER FOR CHAT MODELS
                available_models = app_utils.fetch_available_models(local_url, filter_type='chat')
                if st.session_state.get("selected_model") and st.session_state.selected_model not in available_models:
                     available_models.insert(0, st.session_state.selected_model)
                
                if not available_models:
                    available_models = ["local-model"]

                selected_model = st.selectbox("Select Model", available_models, label_visibility="collapsed")
            
            ctx_window = st.selectbox(
                "Context Limit (Max Tokens)",
                [4096, 8192, 16384, 56000, 120000],
                index=1,
                help="Set this to match your loaded model's limit (e.g. Llama3 is 8192)"
            )
                
        st.divider()

        # ------------------------------------------
        # B. EMBEDDING SETTINGS
        # ------------------------------------------
        st.caption("🧠 Embedding Model")

        new_emb_url = st.text_input(
            "Embedding API URL", 
            value=st.session_state.emb_url, 
            key="emb_url_input",
            on_change=_update_emb_url
        )

        col_emb_sel, col_emb_ref = st.columns([0.85, 0.15])
        
        with col_emb_ref:
            if st.button("🔄", key="refresh_emb", help="Fetch Embedding Models"):
                st.cache_data.clear()
                st.rerun()

        with col_emb_sel:
            # FILTER FOR EMBEDDING MODELS
            emb_models = app_utils.fetch_available_models(new_emb_url, filter_type='embed')
            
            # PRESERVE SELECTION: Ensure current model is in list so it doesn't vanish
            current_emb = st.session_state.emb_model
            if current_emb and current_emb not in emb_models:
                emb_models.insert(0, current_emb)
            
            if not emb_models:
                emb_models = [current_emb] if current_emb else ["text-embedding-nomic-embed-text-v1.5"]
            
            # Find index safely
            try:
                current_idx = emb_models.index(current_emb)
            except ValueError:
                current_idx = 0

            selected_emb_model = st.selectbox(
                "Select Embedding", 
                emb_models, 
                index=current_idx,
                key="emb_selector",
                on_change=_update_emb_model,
                label_visibility="collapsed"
            )

        if st.button("💾 Apply & Load Embeddings", type="primary", width="stretch"):
            from query.data_service import data_service
            with st.spinner("Initializing Embeddings..."):
                success = data_service.update_embedding_config(
                    st.session_state.emb_url, 
                    st.session_state.emb_model
                )
                if success:
                    st.success(f"Loaded: {st.session_state.emb_model}")
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to connect to Embedding API.")

    # ==========================================
    # 2. SYSTEM PERSONAS
    # ==========================================
    with st.expander("🎭 System Personas", expanded=False):
        saved_prompts = app_utils.load_system_prompts()
        prompt_names = list(saved_prompts.keys())
        
        if "selected_persona_name" not in st.session_state:
            st.session_state.selected_persona_name = prompt_names[0]
        
        selected_name = st.selectbox(
            "Load Preset", 
            prompt_names, 
            index=prompt_names.index(st.session_state.selected_persona_name) if st.session_state.selected_persona_name in prompt_names else 0,
            key="persona_selector"
        )

        if selected_name != st.session_state.get("last_loaded_persona"):
            st.session_state.custom_system_prompt = saved_prompts[selected_name]
            st.session_state.last_loaded_persona = selected_name
            st.session_state.selected_persona_name = selected_name

        custom_system_prompt = st.text_area(
            "Current Instructions", 
            key="custom_system_prompt", 
            height=150
        )

        with st.popover("💾 Save New Persona"):
            new_persona_name = st.text_input("Name", placeholder="e.g., Python Auditor")
            if st.button("Save Preset", width='stretch'):
                if new_persona_name and custom_system_prompt:
                    app_utils.save_system_prompt(new_persona_name, custom_system_prompt)
                    st.success(f"Saved '{new_persona_name}'")
                    st.rerun()

    # ==========================================
    # 3. KNOWLEDGE BASE (Updated for Persistence)
    # ==========================================
    with st.expander("📚 Knowledge Base", expanded=True):
        
        # 1. Strategy Selector (Persisted)
        rag_options = ["rag", "lightrag", "kag", "direct"]
        try:
            strat_idx = rag_options.index(st.session_state.rag_strategy)
        except ValueError:
            strat_idx = 0
            
        rag_type = st.selectbox(
            "Strategy", 
            rag_options,
            index=strat_idx,
            key="rag_type_selector",
            on_change=_update_rag_type,
            format_func=lambda x: {"rag": "Standard RAG", "lightrag": "LightRAG", "kag": "KAG", "direct": "LLM Only"}.get(x, x)
        )
        is_direct = (rag_type == "direct")
        
        col_db_select, col_db_refresh = st.columns([0.85, 0.15])
        
        # 2. Get Available DBs based on selected strategy
        dbs = list_available_dbs(rag_type) if not is_direct else [DEFAULT_DB_NAME]
        
        # 3. Ensure Persisted DB is valid for current strategy
        # If the persisted DB (e.g., 'default') isn't in the new strategy (e.g., KAG has different DBs),
        # we default to the first one available, but we don't overwrite session state immediately
        # unless the user makes a selection.
        current_selection_valid = st.session_state.persisted_db in dbs
        
        if not current_selection_valid and dbs:
            # Auto-select first if invalid
            current_db_value = dbs[0]
            # Update persistence implicitly so logic downstream works
            st.session_state.persisted_db = current_db_value
        elif not dbs:
            current_db_value = "No DB"
        else:
            current_db_value = st.session_state.persisted_db

        try:
            db_index = dbs.index(current_db_value)
        except ValueError:
            db_index = 0

        with col_db_select:
            selected_db = st.selectbox(
                "Database", 
                dbs if dbs else ["No DB"], 
                index=db_index,
                disabled=is_direct, 
                label_visibility="collapsed",
                key="db_selector",
                on_change=_update_selected_db
            )
        
        with col_db_refresh:
            if st.button("🔄", key="refresh_db", help="Refresh Database List", width='stretch'):
                st.rerun()
        
        hybrid = st.checkbox("Hybrid", value=True, disabled=is_direct)

    # ==========================================
    # 4. PARAMETERS
    # ==========================================
    with st.expander("🎛️ Parameters & Context", expanded=False):
        top_k = st.slider("Retrieval Depth (Docs)", 1, 25, 10, disabled=is_direct)
        history_limit = st.slider("Chat Memory (Msgs)", 0, 35, 5)
        temp = st.slider("Temperature", 0.0, 1.0, 0.7)

    # Return clean configuration
    return {
        "provider": provider,
        "api_key": api_key,
        "selected_model": selected_model,
        "local_url": local_url,
        "ctx_window": ctx_window,
        "custom_system_prompt": custom_system_prompt,
        "rag_type": rag_type,
        "selected_db": selected_db,
        "hybrid": hybrid,
        "top_k": top_k,
        "history_limit": history_limit,
        "temp": temp,
        "is_direct": is_direct
    }