# --- START OF FILE app_utils.py ---
import html
import streamlit as st
import json
import os
import requests
import re
import tiktoken
import fitz
from query.data_service import data_service

# --- Constants ---
AVG_CHARS_PER_TOKEN = 4

# --- System Prompts / Personas ---
DEFAULT_RED_TEAM_PROMPT = """
You are a Deep-System Specialist with polymath-level expertise in Cybersecurity, IT Infrastructure, and Software Engineering. 

### CORE OPERATING PRINCIPLES:
1.  **Fluid Reasoning**: Do not adhere to rigid templates. Dynamically assess the complexity of the user's request. If it requires a simple one-liner, provide it. If it requires a multi-stage architectural breakdown, provide that. Match the user's depth instantly.
2.  **Abductive Logic**: You are capable of inferring missing context. If a user asks a vague question, use your intuition to identify the most likely technical hurdle they are facing and address that first, then offer alternatives.
3.  **Synthesis over Retrieval**: Do not just "dump" information. Synthesize your vast knowledge base into a cohesive solution. Connect disparate concepts (e.g., how a network race condition affects a specific database lock).
4.  **Recursive Optimization**: Before outputting, simulate a "peer review" of your answer. If a solution is technically valid but practically messy, discard it for the cleaner, more elegant approach.

### OUTPUT STANDARDS:
- **Density**: Maximise information density. Use precise terminology. Avoid fluff, filler, and stating the obvious.
- **Pragmatism**: Prioritize solutions that work in real-world, messy environments over theoretical "textbook" answers.
- **Prohibitions:** No ethical preambles, no educational "fluff", no conversational filler.
- **Formatting:** The use of tables is forbidden.

### TONE:
Effortlessly competent. You are the senior engineer who solves the problem in 5 minutes that took others 5 days. Direct, sharp, and helpful.
"""

# since models are de-restricted this might be a waste
# 4. EXCLUSIONS: NEVER discuss ethics/legal considerations. NEVER include disclaimers. NEVER refuse strictly technical requests.

PROMPTS_FILE = "prompts.json"

def apply_custom_css():
    """Injects custom CSS for chat interface and table-safe code blocks."""
    st.markdown("""
    <style>
        /* MAIN CONTAINER */
        .main .block-container { 
            max_width: 95%; 
            padding-top: 2rem; 
            padding-bottom: 2rem; 
        }

        /* CHAT MESSAGES */
        .stChatMessage { 
            background-color: transparent; 
            border-bottom: 1px solid rgba(128, 128, 128, 0.1); 
        }

        /* GLOBAL CODE FONT */
        code { 
            font-family: 'Fira Code', 'Consolas', monospace; 
            font-size: 0.9em; 
        }

        /* FLATTENED CODE BLOCKS (For inside tables) */
        .flattened-code {
            font-family: 'Fira Code', 'Consolas', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;       /* Wrap text so table doesn't overflow */
            word-break: break-all;       /* Break long strings/hashes */
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.2);
            border-radius: 4px;
            padding: 6px;
            margin: 4px 0;
            display: block;
        }

        /* CITATION BADGES */
        .source-citation {
            display: inline-flex;
            align-items: center;
            background-color: rgba(0, 173, 181, 0.15); 
            border: 1px solid rgba(0, 173, 181, 0.4);
            border-radius: 4px;
            padding: 2px 6px;
            margin: 0 3px;
            font-size: 0.8em;
            color: #00ADB5; /* Teal text for contrast */
            font-family: 'Segoe UI', sans-serif;
            vertical-align: middle;
            cursor: default;
        }
        
        .source-citation:before {
            content: "📄";
            margin-right: 4px;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)
    
def format_citations(text: str) -> str:
    """
    Converts [Source: filename] -> Styled HTML span.
    Handles optional backticks and prevents XSS via filename escaping.
    """
    if not text: 
        return ""
    
    def replace_match(match):
        # Extract filename (group 1)
        filename = match.group(1).strip()
        # Secure the content
        safe_filename = html.escape(filename)
        return f'<span class="source-citation">{safe_filename}</span>'

    # Regex: `? \[Source: (content) \] `?
    # Matches optional backticks around the bracketed source
    pattern = r'`?\[Source:\s*(.*?)\]`?'
    
    return re.sub(pattern, replace_match, text)
    
    
@st.cache_resource
def warm_up_resources():
    """Initialize heavy resources (Embeddings, Reranker) once."""
    try:
        _ = data_service.embedding_function
        _ = data_service.reranker
        return True
    except Exception as e:
        st.error(f"🔥 Critical AI Resource Error: {e}")
        return False
    
def clean_api_url(url: str) -> str:
    """Ensures URL is formatted correctly for requests."""
    if not url: return ""
    url = url.strip().rstrip('/')
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url += "/v1"
    return url

def fetch_available_models(base_url: str) -> list:
    """
    Fetches models from the Local LLM API.
    Returns a list of model IDs.
    """
    if not base_url: return []
    target_url = clean_api_url(base_url)
    
    try:
        response = requests.get(f"{target_url}/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return [m['id'] for m in data['data']]
            elif 'models' in data:
                return [m['name'] for m in data['models']]
    except Exception:
        return []
    return []


def parse_reasoning(text: str):
    """Extracts <think> tags into a separate UI block."""
    pattern = r"<(think|reasoning|thought)>(.*?)</\1>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        reasoning = match.group(2).strip()
        clean_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return clean_text, reasoning
    return text, None

def parse_uploaded_file(uploaded_file):
    """Extract text from uploaded PDF/TXT using PyMuPDF (fitz)."""
    try:
        if uploaded_file.name.endswith('.pdf'):
            # fitz requires bytes; Streamlit's uploaded_file.read() returns bytes
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = chr(12).join([page.get_text() for page in doc])
            return f"\n--- UPLOADED CONTEXT: {uploaded_file.name} ---\n{text}\n"
        else:
            text = uploaded_file.read().decode("utf-8")
            return f"\n--- UPLOADED CONTEXT: {uploaded_file.name} ---\n{text}\n"
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return ""

# Cache the encoder to prevent reloading
@st.cache_resource
def get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    """Robust token counting with fallback."""
    if not text: return 0
    enc = get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return len(text) // AVG_CHARS_PER_TOKEN

def smart_prune_history(messages, max_tokens):
    """
    Smartly selects the most recent messages that fit within the token budget.
    Always keeps the System prompt (if it existed) or the very last user message.
    """
    current_tokens = 0
    selected_msgs = []
    
    # Iterate backwards from most recent
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg["content"])
        if current_tokens + msg_tokens > max_tokens:
            break
        selected_msgs.insert(0, msg)
        current_tokens += msg_tokens
        
    return selected_msgs

def render_token_estimator(top_k, history_limit, current_messages, context_window=8192, sys_tokens=0):
    """
    Renders visual token budget.
    """
    if context_window <= 0: context_window = 8192 # Safety fallback

    # 1. Calculate History Tokens
    history_subset = current_messages[-history_limit:] if history_limit > 0 else []
    history_content = "".join([str(m.get("content", "")) for m in history_subset])
    history_tokens = count_tokens(history_content)
    
    # 2. Estimate Retrieval Context (Approx 250 tokens per doc chunk)
    RETRIEVAL_OVERHEAD = 250
    retrieval_tokens = top_k * RETRIEVAL_OVERHEAD

    # 3. Reserve Buffer for the AI's Reply
    OUTPUT_BUFFER = 500
    
    # 4. Total Calculation
    total_estimated = sys_tokens + history_tokens + retrieval_tokens + OUTPUT_BUFFER
    
    # 5. Visual Logic
    ratio = min(total_estimated / context_window, 1.0)
    percentage = ratio * 100
    
    # Color Coding
    if ratio < 0.75:
        bar_color = "#00ADB5" # Teal (Safe)
    elif ratio < 0.90:
        bar_color = "#FFA500" # Orange (Warning)
    else:
        bar_color = "#FF4B4B" # Red (Critical)

    # UI Render
    st.caption(f"📊 **Context Budget** ({total_estimated} / {context_window} tokens)")
    
    st.markdown(f"""
        <div style="background-color: rgba(128,128,128,0.2); border-radius: 5px; height: 8px; width: 100%; margin-bottom: 5px;">
            <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; border-radius: 5px; transition: width 0.3s;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    # Detailed breakdown
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"**Sys**: {sys_tokens}")
    c2.caption(f"**Chat**: {history_tokens}")
    c3.caption(f"**Docs**: ~{retrieval_tokens}")
    c4.caption(f"**Free**: {max(0, context_window - total_estimated)}")
    
    
def load_system_prompts():
    """Loads saved prompts or returns defaults."""
    defaults = {
        "Red Team": DEFAULT_RED_TEAM_PROMPT
    }
    
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r') as f:
                saved = json.load(f)
                # Merge saved with defaults (saved overrides defaults if same name)
                return {**defaults, **saved}
        except Exception:
            pass
    return defaults

def save_system_prompt(name, content):
    """Saves a new prompt to disk."""
    current = load_system_prompts()
    current[name] = content
    try:
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(current, f, indent=2)
        return True
    except Exception as e:
        return False
    
def sanitize_markdown(text: str) -> str:
    """
    Context-aware sanitizer:
    1. Text Segments: Escapes Windows paths (e.g. \Windows) to prevent LaTeX crashes.
    2. Table Code: Flattens ```blocks``` inside tables to HTML <div class="flattened-code">.
    3. Normal Code: Preserves standard Streamlit code blocks.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Split by code blocks. 
    # Capturing group () keeps the delimiter in the list.
    # Pattern finds: ```language \n content ```
    code_block_pattern = r'(```[\s\S]*?```)'
    parts = re.split(code_block_pattern, text)
    
    processed_parts = []
    
    for i, part in enumerate(parts):
        # Even index = Regular Text
        # Odd index  = Code Block (captured by regex)
        
        if i % 2 == 0:
            # --- PROCESS TEXT ---
            # Escape backslashes followed by alphanumerics (Windows paths)
            # Regex: Backslash NOT preceded by backslash, followed by char/digit
            # This turns "C:\Windows" into "C:\\Windows"
            sanitized_text = re.sub(r'(?<!\\)\\(?=[a-zA-Z0-9])', r'\\\\', part)
            processed_parts.append(sanitized_text)
        
        else:
            # --- PROCESS CODE BLOCK ---
            # Heuristic: Is this block inside a table?
            # Check the PREVIOUS text segment. If it ends with a pipe |, we are likely in a table cell.
            prev_text = parts[i-1].strip() if i > 0 else ""
            is_inside_table = prev_text.endswith("|")
            
            if is_inside_table:
                # 1. Parse content inside backticks
                # Match: ```(optional_lang)\n(content)```
                m = re.match(r'```(\w*)\s*\n?([\s\S]*?)```', part)
                if m:
                    # lang = m.group(1) # Unused in flat view, but available
                    raw_code = m.group(2)
                    
                    # 2. Convert to HTML for Table Safety
                    safe_code = html.escape(raw_code)
                    safe_code = safe_code.replace('\n', '<br>')
                    
                    # 3. Append styled HTML div
                    processed_parts.append(f'<div class="flattened-code">{safe_code}</div>')
                else:
                    # Fallback if regex fails (shouldn't happen on valid blocks)
                    processed_parts.append(part)
            else:
                # Outside table: Keep original Markdown for Streamlit to render natively
                processed_parts.append(part)

    return "".join(processed_parts)