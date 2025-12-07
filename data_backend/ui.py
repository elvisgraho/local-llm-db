import streamlit as st
import os
import time
import shutil
import ui_logic as logic
import ui_tab_build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from query.database_paths import DATABASE_DIR, PROJECT_ROOT

# --- Configuration & Setup ---
# Use Project Root to locate Data (siblings to frontend)
RAW_FILES_DIR = PROJECT_ROOT /  "volumes" / "raw_files"
# Use the centralized DATABASE_DIR (handles Docker/Local paths automatically)
DB_DIR = DATABASE_DIR

RAW_FILES_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Architect", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Management ---
if "process_logs" not in st.session_state:
    st.session_state.process_logs = []
if "available_models" not in st.session_state:
    st.session_state.available_models = []
if "delete_confirm" not in st.session_state:
    st.session_state.delete_confirm = False
if "delete_target" not in st.session_state:
    st.session_state.delete_target = None
# Initialize target_db_name for the text input in the main tab
if "target_db_name" not in st.session_state:
    st.session_state.target_db_name = "default"

# --- Sidebar: Configuration & Manager ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. LLM Provider
    provider = st.radio("LLM Provider", ["Local (LM Studio/Ollama)", "Google Gemini"])
    env_vars = {}
    
    if provider == "Google Gemini":
        gemini_key = st.text_input("Gemini API Key", type="password")
        if gemini_key:
            env_vars["GOOGLE_API_KEY"] = gemini_key
            env_vars["LLM_PROVIDER"] = "gemini"
    else:
        st.subheader("Local API")
        llm_url = st.text_input("API URL", value="http://10.2.0.2:1234")
        env_vars["LOCAL_LLM_API_URL"] = llm_url
        
        if st.button("🔄 Check Connection"):
            models = logic.fetch_local_models(llm_url)
            if models:
                st.session_state.available_models = models
                st.success(f"Online: {len(models)} models")
            else:
                st.error("Offline")
        
        if st.session_state.available_models:
            embed_model = st.selectbox("Embedding Model", st.session_state.available_models, index=0)
            main_model = st.selectbox("Inference Model", st.session_state.available_models, index=0)
        else:
            embed_model = st.text_input("Embedding Model", value="text-embedding-embedder_collection")
            main_model = st.text_input("Inference Model", value="local-model")
            
        env_vars["EMBEDDING_MODEL_NAME"] = embed_model
        env_vars["LOCAL_MAIN_MODEL"] = main_model

    st.divider()
    
    # 2. Database Manager
    c_head, c_ref = st.columns([3, 1])
    c_head.subheader("💾 Database Manager")
    if c_ref.button("🔄", help="Refresh Database List"):
        st.rerun()
    
    # Scan for existing databases
    db_inventory = logic.scan_databases(DB_DIR)
    
    if db_inventory:
        # Display Info Table
        st.dataframe(
            db_inventory, 
            column_order=["Type", "Name", "Size", "Files"],
            hide_index=True,
            width='stretch',
            height=150
        )
        
        # Selection Controls
        db_options = [f"{d['Type'].upper()} | {d['Name']}" for d in db_inventory]
        selected_db_str = st.selectbox("Manage Database", db_options, index=0)
        
        if selected_db_str:
            selected_idx = db_options.index(selected_db_str)
            target_db = db_inventory[selected_idx]
            
            c_use, c_del = st.columns(2)
            
            # Button: Use Name (Updates the input field in Build tab)
            if c_use.button("Use Name"):
                st.session_state.target_db_name = target_db["Name"]
                st.toast(f"Set active DB to: {target_db['Name']}")
                time.sleep(0.5)
                st.rerun()

            # Button: Delete Request
            if c_del.button("Delete"):
                st.session_state.delete_target = target_db["Path"]
                st.session_state.delete_confirm = True

    else:
        st.info("No databases found.")

    # 3. Delete Confirmation Modal
    if st.session_state.delete_confirm:
        st.warning(f"⚠️ Confirm Delete?\nPath: {st.session_state.delete_target}")
        col_yes, col_no = st.columns(2)
        if col_yes.button("✅ Yes, Delete"):
            if logic.delete_database_instance(st.session_state.delete_target):
                st.success("Deleted successfully.")
                st.session_state.delete_confirm = False
                time.sleep(1)
                st.rerun()
            else:
                st.error("Failed to delete.")
        if col_no.button("❌ Cancel"):
            st.session_state.delete_confirm = False
            st.rerun()

# --- Main Layout ---
# Header
c1, c2 = st.columns([3, 1])
c1.title("🧬 RAG Architect")

# Tabs
tab_import, tab_preview, tab_build = st.tabs(["📂 Import Files", "🔬 Chunking Lab", "🚀 Build Database"])

# Tab 1: Import
with tab_import:
    c_imp1, c_imp2 = st.columns([1, 1])
    with c_imp1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader("Drag & Drop Files", accept_multiple_files=True)
        if uploaded_files and st.button(f"Save {len(uploaded_files)} Files"):
            progress = st.progress(0, text="Saving...")
            for i, uf in enumerate(uploaded_files):
                with open(RAW_FILES_DIR / uf.name, "wb") as f:
                    f.write(uf.getbuffer())
                progress.progress((i + 1) / len(uploaded_files))
            progress.empty()
            st.success("Files saved.")
            time.sleep(0.5)
            st.rerun()

    with c_imp2:
        st.subheader("Staging Area")
        file_rows, total_files = logic.get_file_inventory(limit=50)
        if total_files > 0:
            st.info(f"Total Staged Files: {total_files}")
            st.dataframe(file_rows, height=300, width='stretch')
            if st.button("Clear Staging Area"):
                for item in RAW_FILES_DIR.iterdir():
                    if item.is_file():
                        os.remove(item)
                    elif item.is_dir():
                        # Recursively delete the directory and its contents
                        shutil.rmtree(item)
                st.rerun()
        else:
            st.info("No files in staging.")

# Tab 2: Preview
with tab_preview:
    st.header("Chunking Visualizer")
    c_prev1, c_prev2 = st.columns([1, 2])
    with c_prev1:
        chunk_size = st.slider("Chunk Size", 100, 4000, 512)
        chunk_overlap = st.slider("Overlap", 0, 500, 200)
        
        all_files = [f for f in RAW_FILES_DIR.rglob('*') if f.is_file() and not f.name.startswith('.')]
        test_file = st.selectbox(
            "Select File to Test", 
            all_files,
            # Use a lambda function to display the relative path string in the selectbox UI
            format_func=lambda f: str(f.relative_to(RAW_FILES_DIR))
        ) if all_files else None
        
    with c_prev2:
        if test_file:
            content = logic.extract_text_for_preview(RAW_FILES_DIR / test_file)
            if "Error" not in content:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_overlap,
                    length_function=len,
                    is_separator_regex=False,
                    separators=[
                        "\n\n## ", "\n\n### ", "\n\n#### ", "\n\n", 
                        "\n```", "\n", ". ", " ", ""
                    ],
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
    ui_tab_build.render_build_tab(env_vars)
   