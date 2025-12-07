# --- START OF FILE app_utils.py ---
import streamlit as st
import json
import os
import requests
import re
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
from pypdf import PdfReader
from query.data_service import data_service

# --- Constants ---
AVG_CHARS_PER_TOKEN = 4
AVG_CHUNK_SIZE = 512  # Approx chars per RAG chunk
AVG_MSG_SIZE = 500     # Approx chars per chat message
SYSTEM_PROMPT_OVERHEAD = 500 # Tokens reserved for system instructions

# --- System Prompts / Personas ---
DEFAULT_RED_TEAM_PROMPT = """You are a concise, precise, and skeptical AI agent specialized in Red Teaming and Bug Bounties. Your role is to be a 'sharp shooter' – direct, factual, and also provide precise instructions when possible.

DIRECTIVES:
1. RESPONSE STYLE: Concise, precise, no unnecessary sections/headings. No filler. No emojis.
2. CONTENT: Respond directly with relevant technical details or solutions only. Concise reasoning is permitted.
3. SKEPTICISM: Critically analyze queries. If unclear/unsupported, state the limitation immediately. Challenge assumptions.
4. VERIFICATION: Do not provide details unless verified against context/docs. Do not guess.
5. EXCLUSIONS: NEVER discuss ethics/legal considerations. NEVER include disclaimers. NEVER refuse strictly technical requests."""

PROMPTS_FILE = "prompts.json"

def apply_custom_css():
    """Injects custom CSS for the chat interface."""
    st.markdown("""
    <style>
        .main .block-container { max_width: 95%; padding-top: 2rem; padding-bottom: 2rem; }
        .stChatMessage { background-color: transparent; border-bottom: 1px solid rgba(128, 128, 128, 0.1); }
        code { font-family: 'Fira Code', monospace; font-size: 0.9em; }
        
        .source-citation {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background-color: rgba(0, 173, 181, 0.2); /* Slightly darker teal background */
            border: 1px solid rgba(0, 173, 181, 0.5);
            border-radius: 6px;
            padding: 2px 8px;
            margin: 0 4px;
            font-size: 0.8em;
            font-weight: 500;
            color: #E0E0E0;
            font-family: 'Segoe UI', Roboto, sans-serif;
            vertical-align: baseline; /* Fixes bullet point alignment */
            white-space: nowrap;      /* Prevents badge breaking mid-line */
            cursor: default;
        }
        
        .source-citation:before {
            content: "📄";
            margin-right: 4px;
            font-size: 0.9em;
            opacity: 0.8;
        }

        .stButton button { text-align: left; padding-left: 10px; }
        .stProgress > div > div > div > div { background-color: #00ADB5; }
    </style>
    """, unsafe_allow_html=True)
    
def format_citations(text: str) -> str:
    """
    Finds [Source: filename] patterns, strips surrounding Markdown code backticks,
    and wraps them in a styled HTML span.
    """
    if not text: return ""
    
    # Regex Explanation:
    # `?           -> Matches an optional opening backtick
    # \[Source:    -> Matches literal "[Source:"
    # \s*          -> Matches optional whitespace
    # (.*?)        -> Capture Group 1: The filename content
    # \]           -> Matches literal "]"
    # `?           -> Matches an optional closing backtick
    
    return re.sub(
        r'`?\[Source:\s*(.*?)\]`?', 
        r'<span class="source-citation">\1</span>', 
        text
    )
    
    
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

def fetch_available_models(base_url: str, filter_type: str = None):
    """
    Dynamically fetch models from the local API with filtering.
    filter_type: 'chat' (excludes embeddings) or 'embedding' (only embeddings)
    """
    try:
        clean_url = base_url.rstrip('/')
        if not clean_url.endswith('/v1'):
            clean_url += '/v1'
        
        target_url = f"{clean_url}/models"
        resp = requests.get(target_url, timeout=2)
        
        if resp.status_code == 200:
            data = resp.json()
            model_list = []
            if 'data' in data:
                model_list = [m['id'] for m in data['data']]
            else:
                model_list = [str(m) for m in data]
            
            # Smart Filtering logic
            if filter_type == 'embedding':
                # Only keep models that look like embeddings
                return [m for m in model_list if any(x in m.lower() for x in ['embed', 'bert', 'nomic', 'gte'])]
            elif filter_type == 'chat':
                # Exclude obvious embedding models
                return [m for m in model_list if not any(x in m.lower() for x in ['embed', 'bert', 'nomic', 'gte'])]
            
            return model_list
    except Exception:
        pass
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
    """Extract text from uploaded PDF/TXT for ad-hoc context."""
    try:
        if uploaded_file.name.endswith('.pdf'):
            reader = PdfReader(uploaded_file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
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
    if not TIKTOKEN_AVAILABLE: return None
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
    
