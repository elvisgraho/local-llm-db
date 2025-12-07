import os
import subprocess
import sys
import requests
import psutil
import shutil
import json
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader

# --- Path Configuration ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAINING_DIR = BASE_DIR / "training"

def get_system_metrics():
    return psutil.cpu_percent(interval=None), psutil.virtual_memory().percent

def fetch_local_models(api_url):
    try:
        clean_url = api_url.rstrip('/')
        if not clean_url.endswith("/v1"): clean_url += "/v1"
        response = requests.get(f"{clean_url}/models", timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data: return [m['id'] for m in data['data']]
            return [str(m) for m in data]
    except Exception:
        return []
    return []

def get_dir_stats(path):
    total_size = 0
    file_count = 0
    if path.exists():
        for p in path.rglob('*'):
            if p.is_file():
                total_size += p.stat().st_size
                file_count += 1
    return total_size / (1024 * 1024), file_count

def scan_databases(db_root):
    """Scans the database directory for all instances."""
    inventory = []
    if not db_root.exists(): return inventory
    
    for r_type in ["rag", "lightrag", "kag"]:
        type_dir = db_root / r_type
        if type_dir.exists():
            for db_instance in type_dir.iterdir():
                if db_instance.is_dir() and not db_instance.name.startswith('.'):
                    size_mb, _ = get_dir_stats(db_instance)
                    
                    # Count processed files via registry if it exists
                    file_count = 0
                    reg_path = db_instance / "processed_files.json"
                    if reg_path.exists():
                        try:
                            with open(reg_path, 'r') as f:
                                file_count = len(json.load(f))
                        except: pass
                    
                    inventory.append({
                        "Type": r_type,
                        "Name": db_instance.name,
                        "Size": f"{size_mb:.1f} MB",
                        "Files": file_count,
                        "Path": str(db_instance)
                    })
    return inventory

def delete_database_instance(path_str):
    """Safely deletes a specific database directory."""
    try:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
    except Exception:
        return False


def delete_database_instance(path_str):
    """Robustly deletes a database directory."""
    try:
        path = Path(path_str)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            return True
        return False
    except Exception:
        return False

# --- File & Script Utilities ---

def get_file_inventory(limit=50):
    try:
        if not DATA_DIR.exists(): return [], 0
        all_files = sorted([f for f in DATA_DIR.iterdir() if f.is_file() and not f.name.startswith('.')], 
                           key=os.path.getmtime, reverse=True)
        files = [{
            "Filename": f.name, 
            "Size (KB)": f"{f.stat().st_size/1024:.1f}",
            "Modified": datetime.fromtimestamp(f.stat().st_mtime).strftime('%H:%M:%S')
        } for f in all_files[:limit]]
        return files, len(all_files)
    except Exception:
        return [], 0

def extract_text_for_preview(file_path, char_limit=10000):
    try:
        suffix = file_path.suffix.lower()
        if suffix == '.pdf':
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()[:2]
            text = "\n\n".join([d.page_content for d in docs])
        elif suffix in ['.txt', '.md', '.markdown', '.log', '.json']:
            with open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                text = f.read(char_limit * 2)
        else:
            return f"Preview not supported for {suffix}"
        return text[:char_limit] + ("\n\n[Truncated]" if len(text) > char_limit else "")
    except Exception as e:
        return f"Error reading file: {str(e)}"

def run_script_generator(script_name, args, env_vars):
    script_path = TRAINING_DIR / script_name
    cmd = [sys.executable, str(script_path)] + args
    
    process_env = os.environ.copy()
    process_env.update(env_vars)
    process_env["ANONYMIZED_TELEMETRY"] = "False"
    process_env["PYTHONPATH"] = f"{str(BASE_DIR)}{os.pathsep}{process_env.get('PYTHONPATH', '')}"

    proc = None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
            text=True, env=process_env, bufsize=1, cwd=str(BASE_DIR)
        )
        yield f"PID:{proc.pid}"
        for line in proc.stdout:
            yield line
    except GeneratorExit:
        if proc: kill_child_processes(proc.pid)
    except Exception as e:
        yield f"CRITICAL ERROR: {str(e)}"
    finally:
        if proc: kill_child_processes(proc.pid)

def kill_child_processes(parent_pid):
    try:
        parent = psutil.Process(parent_pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except psutil.NoSuchProcess:
        pass