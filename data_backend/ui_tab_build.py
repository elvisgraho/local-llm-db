"""
Enhanced Build Tab with improved UX and real-time progress tracking.

Features:
- Visual job progress with live metrics
- Better error handling and recovery
- Build history and logs
- Estimated time remaining
- One-click database testing
- Advanced build options with presets
"""

import streamlit as st
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional
from common import (
    PROJECT_ROOT,
    LOGS_DIR,
    get_file_inventory,
    is_process_alive,
    start_script_background,
    read_log_file,
    kill_child_processes,
    JobState
)


# Local constants
TRAINING_DIR = Path(__file__).resolve().parent / "training"
STATE_FILE = LOGS_DIR / "active_job.json"
HISTORY_FILE = LOGS_DIR / "build_history.json"


# === JOB STATE MANAGEMENT ===

def save_job_state(pid: int, start_time: float, db_name: str, flavor: str, options: Dict) -> None:
    """Save current job state with build options."""
    state = JobState(STATE_FILE)
    data = {
        "pid": pid,
        "start_time": start_time,
        "db_name": db_name,
        "flavor": flavor,
        "options": options
    }
    state.save(**data)


def load_job_state() -> Optional[Dict]:
    """Load active job state."""
    return JobState(STATE_FILE).load()


def clear_job_state() -> None:
    """Clear job state file."""
    JobState(STATE_FILE).clear()


def save_build_history(db_name: str, flavor: str, duration: float, success: bool, options: Dict):
    """Save build to history."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except:
            pass

    history.insert(0, {
        "timestamp": datetime.now().isoformat(),
        "db_name": db_name,
        "flavor": flavor,
        "duration": duration,
        "success": success,
        "options": options
    })

    # Keep only last 20 builds
    history = history[:20]

    HISTORY_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)


def get_build_history() -> list:
    """Load build history."""
    if not HISTORY_FILE.exists():
        return []

    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return []


# === LOG PARSING ===

def parse_log_metrics(log_content: str) -> Dict:
    """
    Extract metrics from log content.

    Returns dict with:
    - processed_files: Number of files processed
    - total_chunks: Number of chunks created
    - errors: Number of errors
    - last_file: Last file being processed
    """
    metrics = {
        "processed_files": 0,
        "total_chunks": 0,
        "errors": 0,
        "last_file": ""
    }

    lines = log_content.split('\n')

    for line in lines:
        # Count processed files
        if "Processing" in line or "Loaded" in line:
            metrics["processed_files"] += 1

        # Extract last file
        match = re.search(r'Processing.*?([^\\/]+\.(pdf|txt|md))', line)
        if match:
            metrics["last_file"] = match.group(1)

        # Count chunks
        match = re.search(r'(\d+)\s+chunks?', line, re.IGNORECASE)
        if match:
            metrics["total_chunks"] = max(metrics["total_chunks"], int(match.group(1)))

        # Count errors
        if "Error" in line or "Failed" in line or "Exception" in line:
            metrics["errors"] += 1

    return metrics


def estimate_time_remaining(start_time: float, processed: int, total: int) -> str:
    """Estimate time remaining based on current progress."""
    if processed == 0 or total == 0:
        return "Calculating..."

    elapsed = time.time() - start_time
    rate = elapsed / processed
    remaining = (total - processed) * rate

    if remaining < 60:
        return f"{int(remaining)}s"
    elif remaining < 3600:
        return f"{int(remaining / 60)}m {int(remaining % 60)}s"
    else:
        hours = int(remaining / 3600)
        minutes = int((remaining % 3600) / 60)
        return f"{hours}h {minutes}m"


# === MAIN RENDER FUNCTION ===

def render_build_tab(env_vars: Dict):
    """Render the enhanced build tab."""

    LOG_FILE = LOGS_DIR / "last_run.log"

    # === STATE INITIALIZATION ===
    if "job_status" not in st.session_state:
        st.session_state.job_status = "idle"
    if "worker_pid" not in st.session_state:
        st.session_state.worker_pid = None
    if "start_time" not in st.session_state:
        st.session_state.start_time = 0
    if "target_db_name" not in st.session_state:
        st.session_state.target_db_name = ""
    if "build_options" not in st.session_state:
        st.session_state.build_options = {}

    # Recover active job
    if st.session_state.job_status == "idle":
        persisted = load_job_state()
        if persisted:
            if is_process_alive(persisted["pid"]):
                st.session_state.job_status = "running"
                st.session_state.worker_pid = persisted["pid"]
                st.session_state.start_time = persisted["start_time"]
                st.session_state.target_db_name = persisted["db_name"]
                st.session_state.build_options = persisted.get("options", {})
                st.toast(f"📊 Recovered active job (PID: {persisted['pid']})")
                time.sleep(0.5)
                st.rerun()
            else:
                clear_job_state()

    is_running = (st.session_state.job_status == "running")

    # Check files
    _, total_files = get_file_inventory(limit=1)

    # === HEADER ===
    st.markdown("### Database Builder")

    if is_running:
        st.caption(f"🔄 Building: **{st.session_state.target_db_name}**")
    else:
        st.caption("Configure and launch database population jobs")

    st.divider()

    # === LAYOUT ===

    if not is_running:
        # === IDLE STATE: CONFIGURATION ===

        # Database name
        db_name = st.text_input(
            "Database Name",
            key="target_db_name",
            placeholder="my-knowledge-base",
            help="Use only letters, numbers, hyphens, and underscores"
        )

        # Check if DB exists
        rag_flavor = "LightRAG"
        db_path_check = PROJECT_ROOT / "databases" / "lightrag" / db_name
        db_exists = db_path_check.exists() and db_name

        if db_exists:
            st.info(f"📁 Database '{db_name}' already exists. You can resume or rebuild.")

        st.divider()

        # Configuration
        st.markdown("#### Chunking Parameters")

        # Load existing config if DB exists
        derived_chunk_size = st.session_state.get("preview_chunk_size", 512)
        derived_chunk_overlap = st.session_state.get("preview_chunk_overlap", 100)

        if db_exists:
            config_path = db_path_check / "db_config.json"
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        saved_cfg = json.load(f)
                        derived_chunk_size = saved_cfg.get("chunk_size", derived_chunk_size)
                        derived_chunk_overlap = saved_cfg.get("chunk_overlap", derived_chunk_overlap)
                except:
                    pass

        # Initialize if not set
        if "build_chunk_size" not in st.session_state:
            st.session_state.build_chunk_size = derived_chunk_size
        if "build_chunk_overlap" not in st.session_state:
            st.session_state.build_chunk_overlap = derived_chunk_overlap

        chunk_size = st.number_input(
            "Chunk Size (characters)",
            min_value=100,
            max_value=4000,
            step=50,
            key="build_chunk_size",
            help="Target size for each document chunk"
        )

        chunk_overlap = st.number_input(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=1000,
            step=25,
            key="build_chunk_overlap",
            help="Overlapping characters between adjacent chunks"
        )

        # Overlap ratio calculation and validation
        overlap_ratio = (chunk_overlap / chunk_size * 100) if chunk_size > 0 else 0

        # Color coding based on overlap ratio ranges
        # 10-20%: Ideal for code and technical documentation (maintains context continuity)
        # 20-30%: Good for dense content with cross-references
        # 30-50%: Acceptable for highly interconnected content
        # >50%: Excessive redundancy
        if 10 <= overlap_ratio <= 20:
            color = '#10b981'  # Green
            status = '✓'
        elif 20 < overlap_ratio <= 30:
            color = '#3b82f6'  # Blue
            status = '✓'
        elif 30 < overlap_ratio <= 50:
            color = '#f59e0b'  # Orange
            status = '⚠'
        else:
            color = '#ef4444'  # Red
            status = '✗'

        st.markdown(f"""
        <div style="background: #1a1d29; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
            <div style="margin-bottom: 0.5rem;">Overlap Ratio: <strong>{overlap_ratio:.1f}%</strong> {status}</div>
            <div style="background: #0f1117; border-radius: 4px; height: 8px; overflow: hidden;">
                <div style="background: {color}; width: {min(overlap_ratio, 100)}%; height: 100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Processing Options")

        opt_col1, opt_col2 = st.columns(2)

        with opt_col1:
            do_reset = st.checkbox(
                "Full Reset",
                value=False,
                help="Delete existing database and rebuild from scratch"
            )

            do_tags = st.checkbox(
                "AI Auto-Tagging",
                value=False,
                help="Use LLM to generate metadata (slower but richer)"
            )

        with opt_col2:
            do_resume = st.checkbox(
                "Resume Build",
                value=True,
                help="Only process new files not in database"
            )

            # OCR disabled by default
            do_ocr = st.checkbox(
                "Enable OCR",
                value=False,
                help="Extract text from images in PDFs"
            )

        if do_ocr:
            st.caption(f"🔍 OCR Model: `{st.session_state.get('ocr_model', 'gliese-ocr-7b-post2.0-final-i1')}`")

        st.divider()

        # Validation and launch
        validation_errors = []

        if not db_name:
            validation_errors.append("Database name is required")
        elif not db_name.replace('_', '').replace('-', '').isalnum():
            validation_errors.append("Database name must contain only letters, numbers, hyphens, and underscores")

        if total_files == 0:
            validation_errors.append("No files found in staging area. Upload documents first.")

        if chunk_overlap >= chunk_size:
            validation_errors.append("Chunk overlap must be less than chunk size")

        # Show warnings
        if validation_errors:
            for error in validation_errors:
                st.warning(f"⚠️ {error}")

        # Launch button
        launch_col1, launch_col2, launch_col3 = st.columns([2, 1, 1])

        with launch_col1:
            if st.button(
                f"🚀 Launch Build ({total_files} files)",
                type="primary",
                disabled=(len(validation_errors) > 0),
                use_container_width=True
            ):
                # Use LightRAG script
                script = "populate_lightrag.py"

                # Build arguments
                args = [
                    "--db_name", db_name,
                    "--chunk_size", str(chunk_size),
                    "--chunk_overlap", str(chunk_overlap)
                ]

                if do_reset:
                    args.append("--reset")
                if do_resume and not do_reset:
                    args.append("--resume")
                if do_tags:
                    args.append("--add-tags")
                if do_ocr:
                    args.append("--ocr")

                # Start job
                script_path = TRAINING_DIR / script
                pid = start_script_background(script_path, args, env_vars, LOG_FILE)

                st.session_state.start_time = time.time()
                st.session_state.worker_pid = pid
                st.session_state.job_status = "running"

                options = {
                    "reset": do_reset,
                    "resume": do_resume,
                    "tags": do_tags,
                    "ocr": do_ocr,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap
                }
                st.session_state.build_options = options

                save_job_state(pid, st.session_state.start_time, db_name, rag_flavor, options)

                st.success("🚀 Build job started!")
                time.sleep(0.5)
                st.rerun()

        with launch_col2:
            if st.button("📜 View History", use_container_width=True):
                st.session_state.show_build_history = True

        # Build history modal
        if st.session_state.get("show_build_history", False):
            st.divider()
            st.markdown("#### Build History")

            history = get_build_history()

            if history:
                for entry in history[:10]:
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    duration_str = str(timedelta(seconds=int(entry["duration"])))

                    status_emoji = "✅" if entry["success"] else "❌"
                    status_color = "#10b981" if entry["success"] else "#ef4444"

                    st.markdown(f"""
                    <div style="background: #1a1d29; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid {status_color};">
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <strong>{status_emoji} {entry['db_name']}</strong>
                                <span style="color: #9ca3af; margin-left: 0.5rem;">({entry['flavor']})</span>
                            </div>
                            <div style="color: #9ca3af; font-size: 0.875rem;">
                                {timestamp.strftime('%Y-%m-%d %H:%M')} • {duration_str}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                if st.button("Close History"):
                    st.session_state.show_build_history = False
                    st.rerun()
            else:
                st.info("No build history available yet")

    elif is_running:
        # === RUNNING STATE ===

        pid = st.session_state.worker_pid
        elapsed = time.time() - st.session_state.start_time

        # Read logs
        logs = read_log_file(LOG_FILE, num_lines=1000)
        metrics = parse_log_metrics(logs)

        # Progress metrics
        progress_percentage = min((metrics["processed_files"] / total_files * 100) if total_files > 0 else 0, 100)
        time_remaining = estimate_time_remaining(st.session_state.start_time, metrics["processed_files"], total_files)

        # Header metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Progress", f"{progress_percentage:.1f}%")

        with metric_col2:
            st.metric("Files Processed", f"{metrics['processed_files']}/{total_files}")

        with metric_col3:
            minutes, seconds = divmod(int(elapsed), 60)
            st.metric("Elapsed Time", f"{minutes}m {seconds}s")

        with metric_col4:
            st.metric("Est. Remaining", time_remaining)

        # Progress bar
        st.progress(progress_percentage / 100, text=f"Processing: {metrics['last_file']}")

        # Additional metrics
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown(f"""
            <div style="background: #1a1d29; padding: 1rem; border-radius: 8px;">
                <div style="color: #9ca3af; margin-bottom: 0.5rem;">Current Status</div>
                <div style="font-size: 1.25rem; font-weight: 600; color: #6366f1;">
                    Processing Documents...
                </div>
                <div style="margin-top: 0.5rem; color: #9ca3af; font-size: 0.875rem;">
                    📄 Last: {metrics['last_file'] or 'Starting...'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with detail_col2:
            st.markdown(f"""
            <div style="background: #1a1d29; padding: 1rem; border-radius: 8px;">
                <div style="color: #9ca3af; margin-bottom: 0.5rem;">Build Statistics</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div>📊 Total Chunks: <strong>{metrics['total_chunks']}</strong></div>
                    <div>⚠️ Errors: <strong style="color: {'#ef4444' if metrics['errors'] > 0 else '#10b981'}">{metrics['errors']}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Control buttons
        control_col1, control_col2 = st.columns([3, 1])

        with control_col2:
            if st.button("🛑 Stop Job", type="primary", use_container_width=True):
                kill_child_processes(pid)
                duration = time.time() - st.session_state.start_time
                save_build_history(
                    st.session_state.target_db_name,
                    "LightRAG",
                    duration,
                    False,
                    st.session_state.build_options
                )
                clear_job_state()
                st.session_state.job_status = "done"
                st.warning("Job stopped by user")
                time.sleep(1)
                st.rerun()

        # Live logs
        with st.expander("📜 Live Logs", expanded=True):
            st.code(logs[-2000:], language="bash")

        # Check if process is still alive
        if not is_process_alive(pid):
            duration = time.time() - st.session_state.start_time

            # Check for errors in logs
            errors = [line for line in logs.split('\n') if "Error" in line or "Traceback" in line]
            success = len(errors) == 0

            save_build_history(
                st.session_state.target_db_name,
                "LightRAG",
                duration,
                success,
                st.session_state.build_options
            )

            clear_job_state()
            st.session_state.job_status = "done"
            st.rerun()
        else:
            # Auto-refresh
            time.sleep(2)
            st.rerun()

    else:
        # === DONE STATE ===

        logs = read_log_file(LOG_FILE, num_lines=2000)
        errors = [line for line in logs.split('\n') if "Error" in line or "Traceback" in line]

        if errors:
            st.error(f"❌ Build completed with {len(errors)} error(s)")

            with st.expander("View Errors", expanded=True):
                for error in errors[:20]:
                    st.text(error)
                if len(errors) > 20:
                    st.caption("...and more. Check full logs below.")
        else:
            st.success("✅ Database build completed successfully!")
            st.balloons()

        if st.button("🔄 Start New Build", type="primary"):
            st.session_state.job_status = "idle"
            st.rerun()

        with st.expander("Full Build Log", expanded=False):
            st.code(logs, language="bash")
