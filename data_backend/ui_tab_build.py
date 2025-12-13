import streamlit as st
import time
import re
import ui_logic as logic
from query.database_paths import PROJECT_ROOT

def render_build_tab(env_vars):
    LOG_FILE = PROJECT_ROOT / "volumes" / "last_run.log"

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
    
    _, total_files = logic.get_file_inventory(limit=1)
    if total_files == 0:
        total_files = 1

    c_b1, c_b2 = st.columns([1, 2])
    with c_b1:
        if "rag_flavor_selector" not in st.session_state:
            st.session_state.rag_flavor_selector = "Standard RAG"
            
        rag_flavor = st.radio(
            "Architecture", 
            ["Standard RAG", "LightRAG", "KAG (Graph)"], 
            key="rag_flavor_selector",
            disabled=is_running
        )
        
        # Initialize keys if missing (fallback to preview or defaults)
        if "build_chunk_size" not in st.session_state:
            st.session_state.build_chunk_size = st.session_state.get("chunk_size_preview", 512)
        if "build_chunk_overlap" not in st.session_state:
            st.session_state.build_chunk_overlap = st.session_state.get("chunk_overlap_preview", 200)
        
        # Bind to keys
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

    with c_b2:
        db_path_check = PROJECT_ROOT / "databases" / rag_flavor.lower().split()[0] / st.session_state.target_db_name
        db_exists = db_path_check.exists() and st.session_state.target_db_name
        
        db_name = st.text_input("Database Name", key="target_db_name", disabled=is_running)
        
        if db_exists and not is_running:
            st.warning(f"Warning: '{db_name}' exists. Use 'Resume' or 'Full Reset'.")

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        do_reset = c_opt1.checkbox("Full Reset", disabled=is_running)
        do_resume = c_opt2.checkbox("Resume", value=True, disabled=is_running)
        do_tags = c_opt3.checkbox("AI Auto-Tagging", disabled=is_running)

    st.divider()

    if st.session_state.job_status == "idle":
        if st.button("Launch Population", type="primary", disabled=(not db_name)):
            script_map = {
                "Standard RAG": "populate_rag.py",
                "LightRAG": "populate_lightrag.py",
                "KAG (Graph)": "populate_kag.py"
            }
            script = script_map[rag_flavor]
            
            args = [
                "--db_name", db_name,
                "--chunk_size", str(chunk_size),
                "--chunk_overlap", str(chunk_overlap)
            ]
            if do_reset: args.append("--reset")
            if do_resume and not do_reset: args.append("--resume")
            if do_tags: args.append("--add-tags")

            pid = logic.start_script_background(script, args, env_vars, LOG_FILE)
            st.session_state.start_time = time.time()
            logic.save_job_state(pid, st.session_state.start_time, db_name, rag_flavor)
            
            st.session_state.worker_pid = pid
            st.session_state.job_status = "running"
            st.rerun()

    elif st.session_state.job_status == "running":
        pid = st.session_state.worker_pid
        
        logs = logic.read_log_file(LOG_FILE, num_lines=1000)
        
        processed_count = len(re.findall(r"(Processing|Success|Inserted)", logs)) 
        progress_val = min(processed_count / total_files, 1.0) if total_files else 0
        
        c_prog, c_res = st.columns([3, 1])
        with c_prog:
            st.progress(progress_val, text=f"Progress: ~{int(progress_val*100)}% ({processed_count}/{total_files} files)")
        with c_res:
            if st.button("STOP JOB", type="primary"):
                logic.kill_child_processes(pid)
                logic.clear_job_state()
                st.session_state.final_duration = f"Stopped"
                st.session_state.job_status = "done"
                st.rerun()

        st.code(logs[-2000:], language="bash")

        if not logic.is_process_alive(pid):
            logic.clear_job_state()
            st.session_state.job_status = "done"
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()

    elif st.session_state.job_status == "done":
        logic.clear_job_state()
        logs = logic.read_log_file(LOG_FILE, num_lines=2000)
        
        errors = [line for line in logs.splitlines() if "Error" in line or "Traceback" in line]
        
        if errors:
            st.error(f"Process finished with {len(errors)} issues.")
            with st.expander("View Failures", expanded=True):
                st.error("\n".join(errors))
        else:
            st.success("Processing Complete Successfully")
            
        if st.button("Start New Job"):
            st.session_state.job_status = "idle"
            st.rerun()

        with st.expander("Full Execution Log", expanded=False):
            st.code(logs, language="bash")