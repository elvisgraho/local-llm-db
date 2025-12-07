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
AVG_CHUNK_SIZE = 1500  # Approx chars per RAG chunk
AVG_MSG_SIZE = 500     # Approx chars per chat message
SYSTEM_PROMPT_OVERHEAD = 500 # Tokens reserved for system instructions

# --- System Prompts / Personas ---
DEFAULT_RED_TEAM_PROMPT = """You are a concise, precise, and skeptical AI agent specialized in Red Teaming and Bug Bounties. Your role is to be a 'sharp shooter' – direct, factual, and providing detail only when requested.

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
        .source-tag { font-size: 0.75em; background-color: #1E1E1E; padding: 2px 8px; border-radius: 12px; margin-right: 5px; color: #00ADB5; }
        .stButton button { text-align: left; padding-left: 10px; }
        /* Progress bar styling for token meter */
        .stProgress > div > div > div > div { background-color: #00ADB5; }
    </style>
    """, unsafe_allow_html=True)

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

def fetch_available_models(base_url: str):
    """Dynamically fetch models from the local API."""
    try:
        clean_url = base_url.rstrip('/')
        if not clean_url.endswith('/v1'):
            clean_url += '/v1'
        
        target_url = f"{clean_url}/models"
        resp = requests.get(target_url, timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if 'data' in data:
                return [m['id'] for m in data['data']]
            return [str(m) for m in data]
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
    Renders visual token budget with color coding using HTML/CSS.
    """
    # Calculate history tokens
    history_subset = current_messages[-history_limit:] if history_limit > 0 else []
    history_content = "".join([str(m.get("content", "")) for m in history_subset])
    history_tokens = count_tokens(history_content)
    
    # Estimate Retrieval Context (Approximate)
    RETRIEVAL_OVERHEAD = 250
    retrieval_tokens = top_k * RETRIEVAL_OVERHEAD

    # Buffer for output generation
    OUTPUT_BUFFER = 500
    
    total_estimated = sys_tokens + history_tokens + retrieval_tokens + OUTPUT_BUFFER
    
    # Logic for Color and Ratio
    ratio = min(total_estimated / context_window, 1.0)
    percentage = ratio * 100
    
    bar_color = "#00ADB5" # Teal (Safe)
    if ratio > 0.75: bar_color = "#FFA500" # Orange (Warning)
    if ratio > 0.90: bar_color = "#FF4B4B" # Red (Critical)

    st.caption(f"📊 **Context Budget** ({total_estimated} / {context_window} tokens)")
    
    # Custom HTML Progress Bar to support colors
    st.markdown(f"""
        <div style="background-color: rgba(128,128,128,0.2); border-radius: 5px; height: 10px; width: 100%;">
            <div style="background-color: {bar_color}; width: {percentage}%; height: 100%; border-radius: 5px; transition: width 0.5s;"></div>
        </div>
    """, unsafe_allow_html=True)
    
    col_sys, col_hist, col_ret = st.columns(3)
    col_sys.caption(f"**Sys**: {sys_tokens}")
    col_hist.caption(f"**Chat**: {history_tokens}")
    col_ret.caption(f"**RAG**: {retrieval_tokens}")
    
    
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