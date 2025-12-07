import streamlit as st
import time
from query.global_vars import LOCAL_LLM_API_URL
from query.database_paths import list_available_dbs, DEFAULT_DB_NAME
import app_utils

def render_settings_sidebar():
    """
    Renders the Settings Sidebar (LLM, Embeddings, Knowledge Base) 
    and returns a dictionary of configuration variables.
    """
    config = {}

    # ==========================================
    # 1. LLM & EMBEDDING SETTINGS
    # ==========================================
    with st.expander("🔌 LLM & Embedding Settings", expanded=True):
        
        # --- Chat Model Settings ---
        st.caption("🗣️ Chat Model")
        provider = st.radio("Provider", ["local", "gemini"], horizontal=True, label_visibility="collapsed")
        
        # Initialize variables to avoid UnboundLocalError
        api_key = None 
        selected_model = ""
        local_url = LOCAL_LLM_API_URL
        ctx_window = 8192

        if provider == "gemini":
            api_key = st.text_input("Gemini API Key", type="password")
            selected_model = st.text_input("Gemini Model", value="gemini-1.5-flash")
            ctx_window = 32000
        else:
            # URL Input (Persisted)
            if "llm_url" not in st.session_state: st.session_state.llm_url = LOCAL_LLM_API_URL
            local_url = st.text_input("LLM API URL", value=st.session_state.llm_url, key="llm_url_input")
            st.session_state.llm_url = local_url

            # Refresh & Select
            col_mod, col_ref = st.columns([0.85, 0.15])
            with col_ref:
                if st.button("🔄", key="refresh_chat", help="Fetch Models"):
                    st.cache_data.clear()
                    st.rerun()
            
            with col_mod:
                available_models = app_utils.fetch_available_models(local_url)
                if not available_models:
                    available_models = ["local-model"]
                
                selected_model = st.selectbox("Select Model", available_models, label_visibility="collapsed")
            
            ctx_window = st.selectbox("Context Limit", [4096, 8192, 16384, 32768, 128000], index=1)

        st.divider()

        # --- Embedding Settings (RAG) ---
        st.caption("🧠 Embedding Model")

        # Initialize Session State for Embeddings
        if "emb_url" not in st.session_state: st.session_state.emb_url = LOCAL_LLM_API_URL
        if "emb_model" not in st.session_state: st.session_state.emb_model = "text-embedding-nomic-embed-text-v1.5"

        # URL Input
        new_emb_url = st.text_input("Embedding API URL", value=st.session_state.emb_url, key="emb_url_input")

        # Refresh & Select
        col_emb_sel, col_emb_ref = st.columns([0.85, 0.15])
        
        with col_emb_ref:
            if st.button("🔄", key="refresh_emb", help="Fetch Embedding Models"):
                st.cache_data.clear()
                st.rerun()

        with col_emb_sel:
            # Fetch models from the Embedding URL
            emb_models = app_utils.fetch_available_models(new_emb_url)
            if not emb_models:
                emb_models = [st.session_state.emb_model, "text-embedding-nomic-embed-text-v1.5"]
            
            # Keep previous selection if valid
            current_idx = 0
            if st.session_state.emb_model in emb_models:
                current_idx = emb_models.index(st.session_state.emb_model)

            selected_emb_model = st.selectbox(
                "Select Embedding", 
                emb_models, 
                index=current_idx,
                key="emb_selector",
                label_visibility="collapsed"
            )

        # Apply Button (Critical for Embeddings)
        if st.button("💾 Apply & Load Embeddings", type="primary", width="stretch"):
            from query.data_service import data_service
            with st.spinner("Initializing Embeddings..."):
                success = data_service.update_embedding_config(new_emb_url, selected_emb_model)
                if success:
                    st.session_state.emb_url = new_emb_url
                    st.session_state.emb_model = selected_emb_model
                    st.success(f"Loaded: {selected_emb_model}")
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
    # 3. KNOWLEDGE BASE
    # ==========================================
    with st.expander("📚 Knowledge Base", expanded=True):
        rag_type = st.selectbox(
            "Strategy", ["rag", "lightrag", "kag", "direct"],
            format_func=lambda x: {"rag": "Standard RAG", "lightrag": "LightRAG", "kag": "KAG", "direct": "LLM Only"}.get(x, x)
        )
        is_direct = (rag_type == "direct")
        
        col_db_select, col_db_refresh = st.columns([0.85, 0.15])
        
        # PERSISTENCE LOGIC for Database
        if "persisted_db" not in st.session_state:
            st.session_state.persisted_db = DEFAULT_DB_NAME
        
        dbs = list_available_dbs(rag_type) if not is_direct else [DEFAULT_DB_NAME]
        
        # Validate persistence
        if st.session_state.persisted_db not in dbs and dbs:
            st.session_state.persisted_db = dbs[0]
        
        # Determine index
        try:
            db_index = dbs.index(st.session_state.persisted_db)
        except ValueError:
            db_index = 0

        with col_db_select:
            selected_db = st.selectbox(
                "Database", 
                dbs if dbs else ["No DB"], 
                index=db_index,
                disabled=is_direct, 
                label_visibility="collapsed",
                key="db_selector"
            )
            st.session_state.persisted_db = selected_db
        
        with col_db_refresh:
            if st.button("🔄", key="refresh_db", help="Refresh Database List", width='stretch'):
                st.rerun()
        
        hybrid = st.checkbox("Hybrid", value=True, disabled=is_direct)

    # ==========================================
    # 4. PARAMETERS
    # ==========================================
    with st.expander("🎛️ Parameters & Context", expanded=False):
        top_k = st.slider("Retrieval Depth (Docs)", 1, 20, 5, disabled=is_direct)
        history_limit = st.slider("Chat Memory (Msgs)", 0, 20, 6)
        temp = st.slider("Temperature", 0.0, 1.0, 0.7)

    # Pack config to return to main
    config = {
        "provider": provider,
        "api_key": api_key, # Guaranteed to be defined now
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
    return config