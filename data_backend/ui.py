import streamlit as st
import os
import time
import shutil
import ui_tab_build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from common import (
    DATABASE_DIR,
    PROJECT_ROOT,
    RAW_FILES_DIR,
    config,
    fetch_local_models,
    scan_databases,
    get_file_inventory,
    extract_text_for_preview,
    delete_database_instance,
    filter_models
)

# --- Configuration & Setup ---
DB_DIR = DATABASE_DIR
LOCAL_LLM_API_URL = config.llm.api_url

RAW_FILES_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Architect", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if "process_logs" not in st.session_state: st.session_state.process_logs = []
if "delete_confirm" not in st.session_state: st.session_state.delete_confirm = False
if "delete_target" not in st.session_state: st.session_state.delete_target = None
if "target_db_name" not in st.session_state: st.session_state.target_db_name = "default"

# Persistent Configuration
if "llm_url" not in st.session_state: st.session_state.llm_url = LOCAL_LLM_API_URL
if "emb_url" not in st.session_state: st.session_state.emb_url = LOCAL_LLM_API_URL
if "emb_model" not in st.session_state: st.session_state.emb_model = "text-embedding-nomic-embed-text-v1.5"
if "chat_model" not in st.session_state: st.session_state.chat_model = "local-model"
if "ocr_model" not in st.session_state: st.session_state.ocr_model = "gliese-ocr-7b-post2.0-final-i1"
if "fetched_models" not in st.session_state: st.session_state.fetched_models = []

# --- Callbacks ---
def _update_llm_url(): st.session_state.llm_url = st.session_state.llm_url_input
def _update_emb_url(): st.session_state.emb_url = st.session_state.emb_url_input
def _update_chat_model(): st.session_state.chat_model = st.session_state.chat_selector
def _update_emb_model(): st.session_state.emb_model = st.session_state.emb_selector
def _update_ocr_model(): st.session_state.ocr_model = st.session_state.ocr_selector

# Note: filter_models is now imported from common module

# --- Sidebar: Configuration & Manager ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. LLM Provider
    provider = st.radio("LLM Provider", ["Local (LM Studio/Ollama)", "Google Gemini"], label_visibility="collapsed")
    env_vars = {}
    
    # --- GEMINI ---
    if provider == "Google Gemini":
        with st.expander("Gemini Settings", expanded=True):
            gemini_key = st.text_input("Gemini API Key", type="password")
            if gemini_key:
                env_vars["GOOGLE_API_KEY"] = gemini_key
                env_vars["LLM_PROVIDER"] = "gemini"
    
    # --- LOCAL ---
    else:
        # A. Inference / Chat Model
        with st.expander("🗣️ Inference / Chat Model", expanded=True):
            st.text_input(
                "API URL", 
                value=st.session_state.llm_url, 
                key="llm_url_input",
                on_change=_update_llm_url
            )
            
            c_mod, c_ref = st.columns([0.85, 0.15])
            with c_ref:
                if st.button("🔄", key="refresh_chat", help="Fetch Models"):
                    models = fetch_local_models(st.session_state.llm_url)
                    if models:
                        st.session_state.fetched_models = models
                        st.toast(f"Found {len(models)} models")
                    else:
                        st.error("Offline")
            
            with c_mod:
                all_models = st.session_state.fetched_models
                show_all = st.toggle("Show All Models", value=False)

                if show_all:
                    chat_candidates = all_models
                else:
                    chat_candidates = filter_models(all_models, 'chat')
                    # Fallback if filter empties the list
                    if not chat_candidates and all_models: 
                        chat_candidates = all_models
                
                if not chat_candidates: chat_candidates = ["local-model"]
                
                curr = st.session_state.chat_model
                idx = chat_candidates.index(curr) if curr in chat_candidates else 0
                
                st.selectbox(
                    "Select Model", 
                    chat_candidates, 
                    index=idx,
                    key="chat_selector",
                    on_change=_update_chat_model,
                    label_visibility="collapsed"
                )
            
            env_vars["LOCAL_LLM_API_URL"] = st.session_state.llm_url
            env_vars["LOCAL_MAIN_MODEL"] = st.session_state.chat_model
            env_vars["LOCAL_OCR_MODEL"] = st.session_state.ocr_model

        # B. Embedding Model
        with st.expander("🧠 Embedding Model", expanded=True):
            st.text_input(
                "Embedding URL", 
                value=st.session_state.emb_url, 
                key="emb_url_input",
                on_change=_update_emb_url,
                help="Usually same as Chat URL"
            )
            
            c_emb, c_emb_ref = st.columns([0.85, 0.15])
            with c_emb_ref:
                 if st.button("🔄", key="refresh_emb"):
                    models = fetch_local_models(st.session_state.emb_url)
                    if models: st.session_state.fetched_models = models

            with c_emb:
                all_models = st.session_state.fetched_models
                emb_candidates = filter_models(all_models, 'embed')
                
                if not emb_candidates and all_models: emb_candidates = all_models
                if not emb_candidates: emb_candidates = ["text-embedding-nomic-embed-text-v1.5"]
                
                curr_emb = st.session_state.emb_model
                idx_emb = emb_candidates.index(curr_emb) if curr_emb in emb_candidates else 0

                st.selectbox(
                    "Select Model",
                    emb_candidates,
                    index=idx_emb,
                    key="emb_selector",
                    on_change=_update_emb_model,
                    label_visibility="collapsed"
                )

            env_vars["EMBEDDING_MODEL_NAME"] = st.session_state.emb_model

        # C. OCR Model
        with st.expander("👁️ OCR Model", expanded=False):
            st.text_input(
                "OCR URL",
                value=st.session_state.llm_url,
                key="ocr_url_input",
                help="Usually same as Chat URL",
                disabled=True  # Same as chat URL
            )

            c_ocr, c_ocr_ref = st.columns([0.85, 0.15])
            with c_ocr_ref:
                 if st.button("🔄", key="refresh_ocr"):
                    models = fetch_local_models(st.session_state.llm_url)
                    if models: st.session_state.fetched_models = models

            with c_ocr:
                all_models = st.session_state.fetched_models
                # Filter for OCR models (vision/ocr keywords)
                ocr_candidates = filter_models(all_models, 'ocr')

                # Fallback to all models if no OCR models found
                if not ocr_candidates and all_models:
                    ocr_candidates = all_models
                if not ocr_candidates:
                    ocr_candidates = ["gliese-ocr-7b-post2.0-final-i1"]

                curr_ocr = st.session_state.ocr_model
                idx_ocr = ocr_candidates.index(curr_ocr) if curr_ocr in ocr_candidates else 0

                st.selectbox(
                    "Select OCR Model",
                    ocr_candidates,
                    index=idx_ocr,
                    key="ocr_selector",
                    on_change=_update_ocr_model,
                    label_visibility="collapsed",
                    help="Vision model for extracting text from images"
                )

            env_vars["LOCAL_OCR_MODEL"] = st.session_state.ocr_model

    st.divider()
    
    with st.expander("💾 Database Manager", expanded=False):
        c_head, c_ref_db = st.columns([3, 1])
        c_head.caption("Manage Local DBs")
        if c_ref_db.button("🔄", key="ref_db"):
            st.rerun()

        db_inventory = scan_databases(DB_DIR)
        
        if db_inventory:
            # Display Info Table
            st.dataframe(
                db_inventory, 
                column_order=["Type", "Name", "Size", "Files"],
                hide_index=True,
                width='stretch',
                height=150
            )
            
            # Robust Selection (Key Map)
            db_map = {f"{d['Type'].upper()} | {d['Name']}": d for d in db_inventory}
            
            selected_key = st.selectbox(
                "Select DB", 
                options=list(db_map.keys()), 
                label_visibility="collapsed"
            )
            
            if selected_key:
                target_db = db_map[selected_key]
                
                # Visual Confirmation
                if st.session_state.target_db_name == target_db["Name"]:
                     st.caption(f"✅ **Active:** {target_db['Name']} ({target_db['Type']})")
                
                c_use, c_del = st.columns(2)

                if c_use.button("Use Database", key=f"use_db_{target_db['Name']}", use_container_width=True):
                    # 1. Set Name
                    st.session_state.target_db_name = target_db["Name"]

                    # 2. Set Architecture
                    type_map = {
                        "rag": "Standard RAG",
                        "lightrag": "LightRAG"
                    }
                    st.session_state.rag_flavor_selector = type_map.get(target_db["Type"], "Standard RAG")

                    # 3. Set Chunking Parameters
                    if "Config" in target_db and target_db["Config"]:
                        cfg = target_db["Config"]
                        if "chunk_size" in cfg:
                            st.session_state.build_chunk_size = int(cfg["chunk_size"])
                        if "chunk_overlap" in cfg:
                            st.session_state.build_chunk_overlap = int(cfg["chunk_overlap"])

                    st.toast(f"Switched to: {target_db['Name']}")
                    time.sleep(0.5)
                    st.rerun()

                if c_del.button("Delete", key=f"del_db_{target_db['Name']}", type="primary", use_container_width=True):
                    st.session_state.delete_target = str(target_db["Path"])
                    st.session_state.delete_confirm = True
                    st.rerun()
        else:
            st.info("No databases found.")

    # 3. Delete Confirmation Modal
    if st.session_state.delete_confirm:
        st.error(f"⚠️ DELETE: {st.session_state.delete_target}?")
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Yes", key="confirm_delete_yes"):
            if delete_database_instance(st.session_state.delete_target):
                st.success("Deleted.")
                st.session_state.delete_confirm = False
                time.sleep(1)
                st.rerun()
        if col_no.button("❌ Cancel", key="confirm_delete_no"):
            st.session_state.delete_confirm = False
            st.rerun()

# --- Main Layout ---
c1, c2 = st.columns([3, 1])
c1.title("🧬 RAG Architect")

# Tabs
tab_import, tab_preview, tab_build = st.tabs(["📂 Import Files", "🔬 Chunking Lab", "🚀 Build Database"])

# Tab 1: Import
with tab_import:
    c_imp1, c_imp2 = st.columns([1, 1])
    with c_imp1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Drag & Drop Files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'md', 'markdown'],
            help="Supported formats: PDF, TXT, MD"
        )
        if uploaded_files and st.button(f"Save {len(uploaded_files)} Files"):
            try:
                progress = st.progress(0, text="Saving...")
                saved_count = 0
                for i, uf in enumerate(uploaded_files):
                    try:
                        file_path = RAW_FILES_DIR / uf.name
                        with open(file_path, "wb") as f:
                            f.write(uf.getbuffer())
                        saved_count += 1
                    except Exception as e:
                        st.error(f"Failed to save {uf.name}: {str(e)}")
                    progress.progress((i + 1) / len(uploaded_files))
                progress.empty()
                st.success(f"Successfully saved {saved_count}/{len(uploaded_files)} files.")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"Error during file upload: {str(e)}")

    with c_imp2:
        st.subheader("Staging Area")
        # Logic is now fast thanks to the fix above
        file_rows, total_files = get_file_inventory(limit=50)
        
        if total_files > 0:
            st.info(f"Total Staged Files: {total_files}")
            if total_files > 1000:
                st.warning("Large dataset detected. Ensure you have enough RAM for processing.")
            
            # Only show the preview of 50
            st.dataframe(file_rows, height=300, width='stretch')
            st.caption(f"Showing first 50 of {total_files} files.")
            
            if st.button("Clear Staging Area", type="secondary"):
                try:
                    deleted_count = 0
                    for item in RAW_FILES_DIR.iterdir():
                        try:
                            if item.is_file():
                                os.remove(item)
                                deleted_count += 1
                            elif item.is_dir():
                                shutil.rmtree(item)
                                deleted_count += 1
                        except Exception as e:
                            st.error(f"Failed to delete {item.name}: {str(e)}")
                    st.success(f"Deleted {deleted_count} items.")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing staging area: {str(e)}")
        else:
            st.info("No files in staging.")

# Tab 2: Preview
with tab_preview:
    st.header("Chunking Visualizer")
    c_prev1, c_prev2 = st.columns([1, 2])
    with c_prev1:
        chunk_size = st.slider(
            "Chunk Size", 100, 1500, 512, 
            key="chunk_size_preview" 
        )
        chunk_overlap = st.slider(
            "Overlap", 0, 500, 200, 
            key="chunk_overlap_preview"
        )
        
        all_files = [f for f in RAW_FILES_DIR.rglob('*') if f.is_file() and not f.name.startswith('.')]
        test_file = st.selectbox(
            "Select File to Test", 
            all_files,
            format_func=lambda f: str(f.relative_to(RAW_FILES_DIR))
        ) if all_files else None
        
    with c_prev2:
        if test_file:
            content = extract_text_for_preview(test_file)
            if "Error" not in content:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
                    keep_separator=True
                )
                chunks = splitter.split_text(content)
                st.metric("Resulting Chunks", len(chunks))
                
                for i, c in enumerate(chunks[:3]):
                    with st.expander(f"Chunk {i+1} ({len(c)} chars)", expanded=True):
                        st.code(c)
            else:
                st.error(content)
        else:
            st.info("Upload files to test chunking.")

# Tab 3: Build
with tab_build:
    st.header("Population Engine")
    # env_vars populated from sidebar are passed here
    ui_tab_build.render_build_tab(env_vars)