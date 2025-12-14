import streamlit as st
import time
import re
import json
import ui_logic as logic
from query.database_paths import PROJECT_ROOT

def render_build_tab(env_vars):
    LOG_FILE = PROJECT_ROOT / "volumes" / "last_run.log"

    # --- 1. State Initialization ---
    if "job_status" not in st.session_state:
        st.session_state.job_status = "idle"
    if "worker_pid" not in st.session_state:
        st.session_state.worker_pid = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = 0
    if "final_duration" not in st.session_state:
        st.session_state.final_duration = ""
    if "target_db_name" not in st.session_state:
        st.session_state.target_db_name = ""

    # Recover active job if browser was refreshed
    if st.session_state.job_status == "idle":
        persisted = logic.load_job_state()
        if persisted:
            if logic.is_process_alive(persisted["pid"]):
                st.session_state.job_status = "running"
                st.session_state.worker_pid = persisted["pid"]
                st.session_state.start_time = persisted["start_time"]
                st.session_state.target_db_name = persisted["db_name"]
                st.toast(f"Recovered active job (PID: {persisted['pid']})")
                time.sleep(0.5)
                st.rerun()
            else:
                logic.clear_job_state()

    is_running = (st.session_state.job_status == "running")
    
    # Check if files exist (Lightweight check)
    _, total_files = logic.get_file_inventory(limit=1)
    if total_files == 0:
        total_files = 1 # Prevent division by zero logic if used later

    # --- 2. Layout & Configuration ---
    c_b1, c_b2 = st.columns([1, 2])
    
    with c_b2:
        # A. Database Name Input
        db_name = st.text_input("Database Name", key="target_db_name", disabled=is_running)
        
        # B. Check for Existing DB
        rag_flavor = st.session_state.get("rag_flavor_selector", "Standard RAG")
        db_path_check = PROJECT_ROOT / "databases" / rag_flavor.lower().split()[0] / db_name
        db_exists = db_path_check.exists() and db_name

        if db_exists and not is_running:
            st.info(f"📁 Database '{db_name}' exists. Settings loaded from config.")

        # C. Smart Configuration Logic
        # 1. Start with defaults from the Chunking Lab (Preview Tab)
        derived_chunk_size = st.session_state.get("chunk_size_preview", 512)
        derived_chunk_overlap = st.session_state.get("chunk_overlap_preview", 200)

        # 2. If DB exists, try to override with its saved config
        if db_exists:
            config_path = db_path_check / "db_config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        saved_cfg = json.load(f)
                        derived_chunk_size = saved_cfg.get("chunk_size", derived_chunk_size)
                        derived_chunk_overlap = saved_cfg.get("chunk_overlap", derived_chunk_overlap)
                except Exception: 
                    pass
        
        # 3. Apply to Session State (if not currently being edited)
        if "build_chunk_size" not in st.session_state:
            st.session_state.build_chunk_size = derived_chunk_size
        if "build_chunk_overlap" not in st.session_state:
            st.session_state.build_chunk_overlap = derived_chunk_overlap

        # D. Action Checkboxes (Reset / Resume / Tags)
        c_opt1, c_opt2, c_opt3 = st.columns(3)
        do_reset = c_opt1.checkbox("Full Reset", help="Delete DB and rebuild from scratch", disabled=is_running)
        do_resume = c_opt2.checkbox("Resume", value=True, help="Process only new files", disabled=is_running)
        do_tags = c_opt3.checkbox("AI Auto-Tagging", help="Use LLM to generate metadata (Slower)", disabled=is_running)

    with c_b1:
        if "rag_flavor_selector" not in st.session_state:
            st.session_state.rag_flavor_selector = "Standard RAG"
            
        st.radio(
            "Architecture", 
            ["Standard RAG", "LightRAG", "KAG (Graph)"], 
            key="rag_flavor_selector",
            disabled=is_running
        )
        
        # Chunking Inputs (Bound to session state calculated above)
        chunk_size = st.number_input(
            "Chunk Size", 100, 4000, 
            key="build_chunk_size",
            disabled=is_running
        )
        chunk_overlap = st.number_input(
            "Overlap", 0, 1000, 
            key="build_chunk_overlap",
            disabled=is_running
        )

    st.divider()

    # --- 3. Action Buttons (Idle State) ---
    if st.session_state.job_status == "idle":
        if st.button("Launch Population", type="primary", disabled=(not db_name)):
            script_map = {
                "Standard RAG": "populate_rag.py",
                "LightRAG": "populate_lightrag.py",
                "KAG (Graph)": "populate_kag.py"
            }
            script = script_map[rag_flavor]
            
            # Construct Arguments
            args = [
                "--db_name", db_name,
                "--chunk_size", str(chunk_size),
                "--chunk_overlap", str(chunk_overlap)
            ]
            
            # Append flags based on checkboxes
            if do_reset: args.append("--reset")
            if do_resume and not do_reset: args.append("--resume")
            if do_tags: args.append("--add-tags")

            # Start Process
            pid = logic.start_script_background(script, args, env_vars, LOG_FILE)
            st.session_state.start_time = time.time()
            logic.save_job_state(pid, st.session_state.start_time, db_name, rag_flavor)
            
            st.session_state.worker_pid = pid
            st.session_state.job_status = "running"
            st.rerun()

    # --- 4. Running State ---
    elif st.session_state.job_status == "running":
        pid = st.session_state.worker_pid
        
        # Read logs without regex parsing (Fix for 8k files lag)
        logs = logic.read_log_file(LOG_FILE, num_lines=1000)
        
        c_status, c_res = st.columns([3, 1])
        with c_status:
            # Simple Spinner instead of calculated progress
            with st.spinner(f"Processing '{st.session_state.target_db_name}'... (Monitor logs below)"):
                time.sleep(0.1)
                
        with c_res:
            if st.button("STOP JOB", type="primary"):
                logic.kill_child_processes(pid)
                logic.clear_job_state()
                st.session_state.job_status = "done"
                st.rerun()

        # Display Logs
        st.code(logs[-2000:], language="bash")

        # Check process health
        if not logic.is_process_alive(pid):
            logic.clear_job_state()
            st.session_state.job_status = "done"
            st.rerun()
        else:
            time.sleep(2) # Refresh rate
            st.rerun()

    # --- 5. Done State ---
    elif st.session_state.job_status == "done":
        logic.clear_job_state()
        logs = logic.read_log_file(LOG_FILE, num_lines=2000)
        
        # Simple Error Detection
        errors = [line for line in logs.splitlines() if "Error" in line or "Traceback" in line]
        
        if errors:
            st.error(f"Process finished with {len(errors)} potential issues.")
            with st.expander("View Failures", expanded=True):
                st.error("\n".join(errors[:20])) # Show first 20 errors
                if len(errors) > 20: st.caption("...and more.")
        else:
            st.success("Processing Complete Successfully")
            
        if st.button("Start New Job"):
            st.session_state.job_status = "idle"
            st.rerun()

        with st.expander("Full Execution Log", expanded=False):
            st.code(logs, language="bash")