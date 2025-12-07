import streamlit as st
import time
import ui_logic as logic
from query.database_paths import PROJECT_ROOT

def render_build_tab(env_vars):
    # --- 1. SETUP & STATE ---
    # Initialize Session State variables if they don't exist
    if "job_status" not in st.session_state:
        st.session_state.job_status = "idle"  # idle, running, done
    if "worker_pid" not in st.session_state:
        st.session_state.worker_pid = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = None
    if "final_duration" not in st.session_state:
        st.session_state.final_duration = ""
    
    # Path to the log file
    LOG_FILE = PROJECT_ROOT / "volumes" / "last_run.log"

    # --- 2. CONFIGURATION UI (Rendered Once) ---
    # We define this HERE, and ONLY HERE.
    # The 'disabled' parameter handles the locking logic.
    is_running = (st.session_state.job_status == "running")
    
    c_b1, c_b2 = st.columns([1, 2])
    with c_b1:
        rag_flavor = st.radio(
            "Architecture", 
            ["Standard RAG", "LightRAG", "KAG (Graph)"], 
            disabled=is_running
        )
    with c_b2:
        # This is the line causing your error if duplicated elsewhere
        db_name = st.text_input(
            "Database Name", 
            key="target_db_name", 
            disabled=is_running
        )
        
        c_opt1, c_opt2, c_opt3 = st.columns(3)
        do_reset = c_opt1.checkbox("Full Reset", help="Delete existing DB.", disabled=is_running)
        do_resume = c_opt2.checkbox("Resume", value=True, disabled=is_running)
        do_tags = c_opt3.checkbox("AI Auto-Tagging", disabled=is_running)

    st.divider()

    # --- 3. EXECUTION LOGIC (State Machine) ---
    
    # STATE: IDLE
    if st.session_state.job_status == "idle":
        if st.button("🚀 Launch Population", type="primary"):
            # Prepare arguments
            script_map = {
                "Standard RAG": "populate_rag.py",
                "LightRAG": "populate_lightrag.py",
                "KAG (Graph)": "populate_kag.py"
            }
            script = script_map[rag_flavor]
            args = ["--db_name", db_name]
            if do_reset: args.append("--reset")
            if do_resume and not do_reset: args.append("--resume")
            if do_tags: args.append("--add-tags")

            # Start Background Process
            pid = logic.start_script_background(script, args, env_vars, LOG_FILE)
            
            # Update State
            st.session_state.worker_pid = pid
            st.session_state.start_time = time.time()
            st.session_state.job_status = "running"
            st.rerun()

    # STATE: RUNNING
    elif st.session_state.job_status == "running":
        pid = st.session_state.worker_pid
        
        # Dashboard
        c_status, c_timer, c_stop = st.columns([2, 1, 1])
        with c_status:
            st.info(f"⚙️ **Processing...** (PID: {pid})")
        with c_timer:
            elapsed = int(time.time() - st.session_state.start_time)
            mins, secs = divmod(elapsed, 60)
            current_duration = f"{mins:02}:{secs:02}"
            st.metric("Duration", current_duration)
        with c_stop:
            if st.button("🛑 STOP JOB", type="primary"):
                logic.kill_child_processes(pid)
                st.session_state.final_duration = current_duration + " (Stopped)"
                st.session_state.job_status = "done"
                st.rerun()

        # Logs
        logs = logic.read_log_file(LOG_FILE)
        with st.container(height=400):
            st.code(logs, language="bash")

        # Auto-Refresh Logic
        if not logic.is_process_alive(pid):
            st.session_state.final_duration = current_duration
            st.session_state.job_status = "done"
            st.rerun()
        else:
            time.sleep(1)
            st.rerun()

    # STATE: DONE
    elif st.session_state.job_status == "done":
        c_res1, c_res2, c_res3 = st.columns([2, 1, 1])
        logs = logic.read_log_file(LOG_FILE)
        is_error = "Error" in logs or "Traceback" in logs or "(Stopped)" in st.session_state.final_duration

        with c_res1:
            if is_error:
                st.error("❌ Process Finished with Issues")
            else:
                st.success("✅ Process Finished Successfully")
        with c_res2:
            st.metric("Total Time", st.session_state.final_duration)
        with c_res3:
            if st.button("🔄 Start New Job"):
                st.session_state.job_status = "idle"
                st.session_state.worker_pid = None
                st.rerun()

        st.subheader("Execution Log")
        with st.container(height=400):
            st.code(logs, language="bash")