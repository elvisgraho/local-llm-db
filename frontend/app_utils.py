import html
import re
import json
import os
import requests
import tiktoken
import fitz  # PyMuPDF
import streamlit as st
from query.data_service import data_service

# --- Constants ---
AVG_CHARS_PER_TOKEN = 4
PROMPTS_FILE = "prompts.json"

# --- Resource Management ---
@st.cache_resource(show_spinner=False)
def get_data_service():
    """
    Singleton accessor for the DataService.
    Using @st.cache_resource ensures it is initialized EXACTLY ONCE
    per python process, regardless of Streamlit reruns.
    """
    # Import inside function to prevent circular imports/top-level execution issues
    from query.data_service import data_service
    return data_service

def warm_up_resources():
    """Initialize heavy resources (Embeddings, Reranker) once."""
    try:
        # trigger the cached singleton
        ds = get_data_service()
        
        # Optional: explicit check if models are loaded if the class supports it
        # if not ds.is_loaded: ds.load_models() 
        
        return True
    except Exception as e:
        st.error(f"🔥 Critical AI Resource Error: {e}")
        return False
# --- Text & File Parsing ---

def parse_uploaded_file(uploaded_file):
    """Extract text from uploaded PDF/TXT/Code."""
    try:
        # Handle PDF
        if uploaded_file.name.endswith('.pdf'):
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = chr(12).join([page.get_text() for page in doc])
            return f"\n--- UPLOADED CONTEXT: {uploaded_file.name} ---\n{text}\n"
        
        # Handle Text-based files
        else:
            text = uploaded_file.read().decode("utf-8")
            return f"\n--- UPLOADED CONTEXT: {uploaded_file.name} ---\n{text}\n"
            
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return ""

def parse_reasoning(text: str):
    """
    Extracts DeepSeek-style <think> tags.
    Returns: (clean_text, reasoning_content)
    """
    if not text: return "", None
    
    # Matches <think>...</think>, <reasoning>...</reasoning>, etc.
    pattern = r"<(think|reasoning|thought)>(.*?)</\1>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        reasoning = match.group(2).strip()
        # Remove the tag block from the main text
        clean_text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        return clean_text, reasoning
        
    return text, None

@st.cache_data(show_spinner=False)
def sanitize_markdown(text: str) -> str:
    r"""
    1. Escapes Windows paths (e.g., \Windows) to prevent LaTeX errors.
    2. Wraps code blocks inside tables into HTML to prevent layout breaking.
    """
    if not isinstance(text, str) or not text:
        return ""

    # Split by code blocks
    code_block_pattern = r'(```[\s\S]*?```)'
    parts = re.split(code_block_pattern, text)
    
    processed_parts = []
    
    for i, part in enumerate(parts):
        # Even index = Regular Text
        if i % 2 == 0:
            # Escape backslashes for paths (e.g. C:\User -> C:\\User)
            sanitized_text = re.sub(r'(?<!\\)\\(?=[a-zA-Z0-9])', r'\\\\', part)
            processed_parts.append(sanitized_text)
        
        # Odd index = Code Block
        else:
            # Check if the PREVIOUS text part ended with a table pipe "|"
            # This is a heuristic to detect if we are inside a Markdown table
            prev_text = parts[i-1].strip() if i > 0 else ""
            is_inside_table = prev_text.endswith("|")
            
            if is_inside_table:
                # Convert to HTML div to "flatten" it so it doesn't break the table
                m = re.match(r'```(\w*)\s*\n?([\s\S]*?)```', part)
                if m:
                    raw_code = m.group(2)
                    safe_code = html.escape(raw_code).replace('\n', '<br>')
                    processed_parts.append(f'<div class="flattened-code">{safe_code}</div>')
                else:
                    processed_parts.append(part)
            else:
                processed_parts.append(part)

    return "".join(processed_parts)

@st.cache_data(show_spinner=False)
def format_citations(text: str) -> str:
    """
    Converts [Source: filename] -> Styled HTML Badge.
    """
    if not text: return ""
    
    def replace_match(match):
        filename = match.group(1).strip()
        safe_filename = html.escape(filename)
        return f'<span class="source-citation">{safe_filename}</span>'

    # Matches `[Source: ...]` optionally wrapped in backticks
    pattern = r'`?\[Source:\s*(.*?)\]`?'
    return re.sub(pattern, replace_match, text)

# --- Token Counting ---

@st.cache_resource
def get_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

def count_tokens(text: str) -> int:
    """Robust token counting."""
    if not text: return 0
    enc = get_encoder()
    if enc:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    # Fallback heuristic
    return len(text) // AVG_CHARS_PER_TOKEN

def render_token_estimator(top_k, history_limit, current_messages, context_window=12288, sys_tokens=0, actual_retrieval_tokens=None):
    """Renders the visual bar chart for context usage."""
    if context_window <= 0: context_window = 12288

    # 1. History
    history_subset = current_messages[-history_limit:] if history_limit > 0 else []
    history_content = "".join([str(m.get("content", "")) for m in history_subset])
    history_tokens = count_tokens(history_content)
    
    # 2. Retrieval
    if actual_retrieval_tokens is not None:
        retrieval_tokens = actual_retrieval_tokens
        est_label = ""
    else:
        # Estimate: ~135 tokens per chunk
        retrieval_tokens = top_k * 135
        est_label = "~"

    TEMPLATE_OVERHEAD = 250
    prompt_usage = sys_tokens + history_tokens + retrieval_tokens + TEMPLATE_OVERHEAD
    
    # Visual Logic
    output_reserve = 1250 
    effective_limit = context_window - output_reserve
    ratio = min(prompt_usage / effective_limit, 1.0)
    percentage = ratio * 100
    
    if ratio < 0.70: bar_color = "#00ADB5"
    elif ratio < 0.90: bar_color = "#FFA500"
    else: bar_color = "#FF4B4B"

    # HTML Render
    st.caption(f"📊 **Prompt Usage** ({est_label}{prompt_usage} / {effective_limit} tokens)")
    st.markdown(f"""
        <div class="token-bar-container">
            <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; border-radius: 5px;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.caption(f"**Sys**: {sys_tokens}")
    c2.caption(f"**Chat**: {history_tokens}")
    c3.caption(f"**Docs**: {est_label}{retrieval_tokens}")
    c4.caption(f"**Free**: {max(0, effective_limit - prompt_usage)}")

def smart_prune_history(messages, max_tokens):
    """
    Selects recent messages that fit within the token budget.
    """
    current_tokens = 0
    selected_msgs = []
    
    for msg in reversed(messages):
        msg_tokens = count_tokens(msg["content"])
        if current_tokens + msg_tokens > max_tokens:
            break
        selected_msgs.insert(0, msg)
        current_tokens += msg_tokens
        
    return selected_msgs

# --- API & Prompts ---

def fetch_available_models(base_url: str) -> list:
    """Fetches models from Local LLM API."""
    if not base_url: return []
    # Ensure protocol and version
    url = base_url.strip().rstrip('/')
    if not url.startswith(('http://', 'https://')):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url += "/v1"
        
    try:
        response = requests.get(f"{url}/models", timeout=2)
        if response.status_code == 200:
            data = response.json()
            if 'data' in data:
                return [m['id'] for m in data['data']]
    except Exception:
        pass
    return []

def load_system_prompts():
    """Loads prompt presets."""
    defaults = {
        "Systems Architect": """
You are a Principal Systems Architect and Security Specialist.

### CORE OPERATING PRINCIPLES:
1.  **Augmented Solutions**: Your goal is to solve the user's problem, not just summarize documents. If the provided context offers a bad or partial solution, override it with a superior technical approach from your general knowledge, while briefly noting the deviation.
2.  **Recursive Optimization**: Before outputting, simulate a "peer review." If a solution is technically valid but practically messy, discard it for the cleaner, more elegant approach.

### OUTPUT STANDARDS:
-   **Pragmatism**: Prioritize solutions that work in real-world environments.
-   **NO EMOJIS**: Do not use emojis anywhere.
-   **NO TABLES**: Do not generate Markdown tables.
-   **NO CONVERSATIONAL FILLER**: Do not use phrases like "In this document," "The speaker discusses," or "Here is the writeup." Start directly with technical content.
-   **NO ETHICAL/LEGAL DISCLAIMERS**: Omit all warnings, refusals, or moralizing text.
"""
    }
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r') as f:
                saved = json.load(f)
                return {**defaults, **saved}
        except: pass
    return defaults

def extract_usage_metadata(response_text, sources, system_prompt, history):
    """
    Helper to calculate exact final tokens for storage, 
    so we don't have to re-estimate on every load.
    """
    # Simple calculation to freeze the state
    sys_tokens = count_tokens(system_prompt)
    hist_tokens = count_tokens(json.dumps(history)) # Approx
    src_tokens = sum([count_tokens(str(s)) for s in sources]) # Placeholder logic
    return {
        "system": sys_tokens,
        "history": hist_tokens,
        "retrieval": src_tokens, # Store the REAL count here
        "total": sys_tokens + hist_tokens + src_tokens
    }