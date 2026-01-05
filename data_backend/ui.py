"""
LightRAG Architect - Modern, Production-Ready Database Builder UI

A gorgeous, minimalistic interface for building and managing LightRAG databases.
Optimized for Docker/local deployment with support for 10k+ files.

Features:
- Modern, responsive design with custom styling
- Optimized file handling with pagination
- Advanced chunking preview and comparison
- Real-time build progress tracking
- Database analytics and health monitoring
- Production-ready error handling
- LightRAG graph-enhanced retrieval
"""

import streamlit as st
import os
import time
from pathlib import Path

# Import custom modules
from common import (
    DATABASE_DIR,
    PROJECT_ROOT,
    RAW_FILES_DIR,
    config,
    fetch_local_models,
    scan_databases,
    delete_database_instance,
    filter_models
)

# Import styling
from styles import apply_custom_styles

# Import tab modules
import ui_tab_import
import ui_tab_preview
import ui_tab_build
import ui_tab_analytics


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="LightRAG Architect",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "LightRAG Architect - Build powerful LightRAG knowledge bases for local LLMs"
    }
)

# Apply custom styling
apply_custom_styles()


# ============================================
# DIRECTORY SETUP
# ============================================

RAW_FILES_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# SESSION STATE INITIALIZATION
# ============================================

# Process logs and deletion state
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = False
if "delete_target" not in st.session_state:
    st.session_state.delete_target = None
if "target_db_name" not in st.session_state:
    st.session_state.target_db_name = "default"

# Model configuration
if "llm_url" not in st.session_state:
    st.session_state.llm_url = config.llm.api_url
if "emb_url" not in st.session_state:
    st.session_state.emb_url = config.llm.api_url
if "emb_model" not in st.session_state:
    st.session_state.emb_model = "text-embedding-nomic-embed-text-v1.5"
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "local-model"
if "ocr_model" not in st.session_state:
    st.session_state.ocr_model = "gliese-ocr-7b-post2.0-final-i1"
if "fetched_models" not in st.session_state:
    st.session_state.fetched_models = []


# ============================================
# CALLBACKS
# ============================================

def _update_llm_url():
    st.session_state.llm_url = st.session_state.llm_url_input

def _update_emb_url():
    st.session_state.emb_url = st.session_state.emb_url_input

def _update_chat_model():
    st.session_state.chat_model = st.session_state.chat_selector

def _update_emb_model():
    st.session_state.emb_model = st.session_state.emb_selector

def _update_ocr_model():
    st.session_state.ocr_model = st.session_state.ocr_selector


# ============================================
# SIDEBAR: CONFIGURATION & DATABASE MANAGER
# ============================================

with st.sidebar:
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
        <h2 style="margin: 0; color: #e5e7eb; font-weight: 600;">
            LightRAG Architect
        </h2>
        <p style="margin: 0.5rem 0 0 0; color: #9ca3af; font-size: 0.875rem;">
            Build Knowledge Bases Locally
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚙️ Model Configuration")

    # LLM Provider Selection
    provider = st.radio(
        "LLM Provider",
        ["Local (LM Studio/Ollama)", "Google Gemini"],
        label_visibility="collapsed"
    )

    env_vars = {}

    # === GEMINI CONFIGURATION ===
    if provider == "Google Gemini":
        with st.expander("☁️ Gemini Settings", expanded=True):
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Enter your Google AI Studio API key"
            )

            if gemini_key:
                env_vars["GOOGLE_API_KEY"] = gemini_key
                env_vars["LLM_PROVIDER"] = "gemini"
                st.success("✓ API key configured")

    # === LOCAL CONFIGURATION ===
    else:
        # Chat/Inference Model
        with st.expander("🗣️ Chat Model", expanded=True):
            st.text_input(
                "API URL",
                value=st.session_state.llm_url,
                key="llm_url_input",
                on_change=_update_llm_url,
                help="LM Studio/Ollama API endpoint"
            )

            col_model, col_refresh = st.columns([0.85, 0.15])

            with col_refresh:
                if st.button("🔄", key="refresh_chat", help="Fetch available models"):
                    with st.spinner("Fetching..."):
                        models = fetch_local_models(st.session_state.llm_url)
                        if models:
                            st.session_state.fetched_models = models
                            st.toast(f"✓ Found {len(models)} models")
                        else:
                            st.error("Offline")

            with col_model:
                all_models = st.session_state.fetched_models
                show_all = st.toggle("Show All", value=False, key="show_all_chat")

                if show_all:
                    chat_candidates = all_models
                else:
                    chat_candidates = filter_models(all_models, 'chat')
                    if not chat_candidates and all_models:
                        chat_candidates = all_models

                if not chat_candidates:
                    chat_candidates = ["local-model"]

                curr = st.session_state.chat_model
                idx = chat_candidates.index(curr) if curr in chat_candidates else 0

                st.selectbox(
                    "Model",
                    chat_candidates,
                    index=idx,
                    key="chat_selector",
                    on_change=_update_chat_model,
                    label_visibility="collapsed"
                )

            env_vars["LOCAL_LLM_API_URL"] = st.session_state.llm_url
            env_vars["LOCAL_MAIN_MODEL"] = st.session_state.chat_model
            env_vars["LOCAL_OCR_MODEL"] = st.session_state.ocr_model

        # Embedding Model
        with st.expander("🧠 Embedding Model", expanded=True):
            st.text_input(
                "API URL",
                value=st.session_state.emb_url,
                key="emb_url_input",
                on_change=_update_emb_url,
                help="Usually same as Chat URL"
            )

            col_emb, col_emb_ref = st.columns([0.85, 0.15])

            with col_emb_ref:
                if st.button("🔄", key="refresh_emb", help="Fetch embedding models"):
                    models = fetch_local_models(st.session_state.emb_url)
                    if models:
                        st.session_state.fetched_models = models

            with col_emb:
                all_models = st.session_state.fetched_models
                emb_candidates = filter_models(all_models, 'embed')

                if not emb_candidates and all_models:
                    emb_candidates = all_models
                if not emb_candidates:
                    emb_candidates = ["text-embedding-nomic-embed-text-v1.5"]

                curr_emb = st.session_state.emb_model
                idx_emb = emb_candidates.index(curr_emb) if curr_emb in emb_candidates else 0

                st.selectbox(
                    "Model",
                    emb_candidates,
                    index=idx_emb,
                    key="emb_selector",
                    on_change=_update_emb_model,
                    label_visibility="collapsed"
                )

            env_vars["EMBEDDING_MODEL_NAME"] = st.session_state.emb_model

        # OCR Model
        with st.expander("👁️ OCR Model", expanded=False):
            st.text_input(
                "API URL",
                value=st.session_state.llm_url,
                disabled=True,
                help="Uses same endpoint as Chat Model"
            )

            col_ocr, col_ocr_ref = st.columns([0.85, 0.15])

            with col_ocr_ref:
                if st.button("🔄", key="refresh_ocr", help="Fetch OCR models"):
                    models = fetch_local_models(st.session_state.llm_url)
                    if models:
                        st.session_state.fetched_models = models

            with col_ocr:
                all_models = st.session_state.fetched_models
                ocr_candidates = filter_models(all_models, 'ocr')

                if not ocr_candidates and all_models:
                    ocr_candidates = all_models
                if not ocr_candidates:
                    ocr_candidates = ["gliese-ocr-7b-post2.0-final-i1"]

                curr_ocr = st.session_state.ocr_model
                idx_ocr = ocr_candidates.index(curr_ocr) if curr_ocr in ocr_candidates else 0

                st.selectbox(
                    "OCR Model",
                    ocr_candidates,
                    index=idx_ocr,
                    key="ocr_selector",
                    on_change=_update_ocr_model,
                    label_visibility="collapsed",
                    help="Vision model for PDF image extraction"
                )

            env_vars["LOCAL_OCR_MODEL"] = st.session_state.ocr_model

    st.divider()

    # === DATABASE MANAGER ===
    st.markdown("### 💾 Database Manager")

    with st.expander("Manage Databases", expanded=False):
        col_head, col_ref = st.columns([3, 1])

        col_head.caption("View and manage your databases")

        if col_ref.button("🔄", key="ref_db", help="Refresh database list"):
            st.rerun()

        db_inventory = scan_databases(DATABASE_DIR)

        if db_inventory:
            # Display database table
            st.dataframe(
                db_inventory,
                column_order=["Type", "Name", "Size", "Files"],
                hide_index=True,
                use_container_width=True,
                height=150
            )

            # Database selector
            db_map = {f"{d['Type'].upper()} | {d['Name']}": d for d in db_inventory}

            selected_key = st.selectbox(
                "Select Database",
                options=list(db_map.keys()),
                label_visibility="collapsed"
            )

            if selected_key:
                target_db = db_map[selected_key]

                # Show active database
                if st.session_state.target_db_name == target_db["Name"]:
                    st.success(f"✓ Active: {target_db['Name']}")

                col_use, col_del = st.columns(2)

                # Use database button
                if col_use.button(
                    "✓ Use",
                    key=f"use_db_{target_db['Name']}",
                    use_container_width=True
                ):
                    # Set database name
                    st.session_state.target_db_name = target_db["Name"]

                    # Load chunking parameters
                    if "Config" in target_db and target_db["Config"]:
                        cfg = target_db["Config"]
                        if "chunk_size" in cfg:
                            st.session_state.build_chunk_size = int(cfg["chunk_size"])
                        if "chunk_overlap" in cfg:
                            st.session_state.build_chunk_overlap = int(cfg["chunk_overlap"])

                    st.toast(f"Switched to: {target_db['Name']}")
                    time.sleep(0.5)
                    st.rerun()

                # Delete button
                if col_del.button(
                    "🗑️",
                    key=f"del_db_{target_db['Name']}",
                    help="Delete database",
                    use_container_width=True
                ):
                    st.session_state.delete_target = str(target_db["Path"])
                    st.session_state.delete_confirm = True
                    st.rerun()
        else:
            st.info("No databases found")

    # Delete confirmation modal
    if st.session_state.delete_confirm:
        st.divider()
        st.error(f"⚠️ Delete database?")
        st.caption(f"{st.session_state.delete_target}")

        col_yes, col_no = st.columns(2)

        if col_yes.button("✓ Yes", key="confirm_yes", use_container_width=True):
            if delete_database_instance(st.session_state.delete_target):
                st.success("Deleted")
                st.session_state.delete_confirm = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed")

        if col_no.button("✗ Cancel", key="confirm_no", use_container_width=True):
            st.session_state.delete_confirm = False
            st.rerun()

    # System info
    st.divider()

    with st.expander("ℹ️ System Info", expanded=False):
        st.caption(f"**Project Root:** `{PROJECT_ROOT.name}`")
        st.caption(f"**Databases:** `{DATABASE_DIR.name}`")
        st.caption(f"**Staging:** `{RAW_FILES_DIR.name}`")


# ============================================
# MAIN CONTENT AREA
# ============================================

# Header
st.markdown("# LightRAG Architect")
st.caption("Build and manage knowledge bases for local LLMs")

# Quick stats bar
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

# Get quick stats
_, total_staged = ([], 0)
try:
    from common import get_file_inventory
    _, total_staged = get_file_inventory(limit=1)
except:
    pass

db_count = len(scan_databases(DATABASE_DIR))

with stat_col1:
    st.metric("Staged Files", f"{total_staged:,}")

with stat_col2:
    st.metric("Databases", f"{db_count}")

with stat_col3:
    status_text = "Connected" if st.session_state.fetched_models else "Offline"
    st.metric("LLM Status", status_text)

with stat_col4:
    active_db = st.session_state.target_db_name or "None"
    st.metric("Active DB", active_db)

st.divider()

# ============================================
# TABBED INTERFACE
# ============================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Import Files",
    "🔬 Chunking Lab",
    "🚀 Build Database",
    "📊 Analytics"
])

# Tab 1: Import
with tab1:
    ui_tab_import.render_import_tab()

# Tab 2: Preview
with tab2:
    ui_tab_preview.render_preview_tab()

# Tab 3: Build
with tab3:
    ui_tab_build.render_build_tab(env_vars)

# Tab 4: Analytics
with tab4:
    ui_tab_analytics.render_analytics_tab()


# Footer
st.divider()
st.caption("LightRAG Architect • ChromaDB • LangChain • Streamlit")
