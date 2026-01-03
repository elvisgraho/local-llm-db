"""
System utilities for process management, file operations, and system monitoring.

This module contains OS-level utilities that are independent of the UI layer.
"""

import json
import os
import psutil
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Generator, Optional, Dict, Any

from .paths import PROJECT_ROOT


def get_system_metrics() -> tuple[float, float]:
    """
    Get current system CPU and memory usage.

    Returns:
        Tuple of (cpu_percent, memory_percent)
    """
    return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent


def get_dir_stats(path: Path) -> tuple[float, int]:
    """
    Calculate directory size and file count.

    Args:
        path: Directory path to analyze

    Returns:
        Tuple of (size_in_mb, file_count)
    """
    total_size = 0
    file_count = 0
    if path.exists():
        for p in path.rglob('*'):
            if p.is_file():
                total_size += p.stat().st_size
                file_count += 1
    return total_size / (1024 * 1024), file_count


def delete_database_instance(path_str: str) -> bool:
    """
    Robustly deletes a database directory.

    Args:
        path_str: String path to database directory

    Returns:
        True if deletion succeeded, False otherwise
    """
    try:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
        return False
    except Exception:
        return False


def kill_child_processes(parent_pid: int) -> None:
    """
    Terminate a process and all its children.

    Args:
        parent_pid: PID of the parent process to terminate
    """
    try:
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass


def is_process_alive(pid: int) -> bool:
    """
    Check if a process is still running.

    Args:
        pid: Process ID to check

    Returns:
        True if process is running and not a zombie
    """
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def run_script_generator(
    script_path: Path,
    args: list[str],
    env_vars: Dict[str, str]
) -> Generator[str, None, None]:
    """
    Run a Python script as a subprocess and yield output lines.

    Args:
        script_path: Path to the Python script
        args: Command-line arguments
        env_vars: Environment variables to set

    Yields:
        Output lines from the script (first line is PID:xxx)
    """
    cmd = [sys.executable, str(script_path)] + args

    process_env = os.environ.copy()
    process_env.update(env_vars)
    process_env["ANONYMIZED_TELEMETRY"] = "False"
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{process_env.get('PYTHONPATH', '')}"

    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=process_env,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            cwd=str(PROJECT_ROOT)
        )
        yield f"PID:{proc.pid}"
        for line in proc.stdout:
            yield line
    except GeneratorExit:
        if proc:
            kill_child_processes(proc.pid)
    except Exception as e:
        yield f"CRITICAL ERROR: {str(e)}"
    finally:
        if proc:
            kill_child_processes(proc.pid)


def start_script_background(
    script_path: Path,
    args: list[str],
    env_vars: Dict[str, str],
    log_file: Path
) -> int:
    """
    Start a Python script in background and redirect output to a file.

    Args:
        script_path: Path to the Python script
        args: Command-line arguments
        env_vars: Environment variables to set
        log_file: Path to log file for output

    Returns:
        PID of the started process
    """
    cmd = [sys.executable, str(script_path)] + args

    process_env = os.environ.copy()
    process_env.update(env_vars)
    process_env["ANONYMIZED_TELEMETRY"] = "False"
    process_env["PYTHONIOENCODING"] = "utf-8"
    process_env["PYTHONPATH"] = f"{str(PROJECT_ROOT)}{os.pathsep}{process_env.get('PYTHONPATH', '')}"

    # Open log file to write output
    with open(log_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=process_env,
            cwd=str(PROJECT_ROOT)
        )
    return proc.pid


def read_log_file(log_file: Path, num_lines: int = 500) -> str:
    """
    Read the last N lines from a log file efficiently.

    Args:
        log_file: Path to log file
        num_lines: Number of lines to read from the end

    Returns:
        String containing the last N lines
    """
    if not log_file.exists():
        return ""
    try:
        file_size = log_file.stat().st_size
        # Read last ~N lines (avg 150 bytes/line)
        read_len = min(file_size, num_lines * 150)

        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            if file_size > read_len:
                f.seek(file_size - read_len)
            lines = f.readlines()
            return "".join(lines[-num_lines:])
    except Exception:
        return "Reading logs..."


class JobState:
    """Manager for persistent job state across UI refreshes."""

    def __init__(self, state_file: Path):
        self.state_file = state_file

    def save(self, pid: int, start_time: str, db_name: str, flavor: str) -> None:
        """Save job details to disk."""
        data = {
            "pid": pid,
            "start_time": start_time,
            "db_name": db_name,
            "flavor": flavor
        }
        with open(self.state_file, "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load active job from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def clear(self) -> None:
        """Clear job state from disk."""
        if self.state_file.exists():
            try:
                self.state_file.unlink()
            except Exception:
                pass
