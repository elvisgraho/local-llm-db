# --- START OF FILE query/llm_service.py ---
import re
import requests
import logging
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

# Local Imports
from query.global_vars import LOCAL_LLM_API_URL, LOCAL_MAIN_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_CONTEXT_LENGTH = 8192

# Regex to strip thinking tags from Refined Query generation
THINKING_TAG_PATTERN = re.compile(
    r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>',
    flags=re.DOTALL | re.IGNORECASE
)

def get_model_context_length(llm_config: Optional[Dict[str, Any]] = None) -> int:
    """Get context length, defaulting to 8k if unknown."""
    if llm_config:
        # Check for context_window in the config (passed from UI slider in future)
        cw = llm_config.get("context_window")
        if cw:
            return int(cw)
            
    # Default for local models
    return 8192

def truncate_history(history: List[Dict], max_tokens: int) -> tuple:
    """Simple history truncation strategy."""
    # Simplified estimation: 1 word ~= 1.3 tokens. 
    # Just take last N messages that fit roughly.
    current_est = 0
    truncated = []
    for msg in reversed(history or []):
        content = msg.get('content', '')
        est_tokens = len(content) / 3
        if current_est + est_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        current_est += est_tokens
    return truncated, int(current_est)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(
    prompt: str,
    llm_config: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7
) -> str:
    """
    Get LLM response with dynamic configuration from Frontend.
    """
    config = llm_config or {}
    
    # 1. Extract Configuration (Frontend Override > Global Defaults)
    provider = config.get('provider', 'local')
    
    # IMPORTANT: Prefer config model, fallback to global
    model_name = config.get('modelName') or LOCAL_MAIN_MODEL
    
    # API URL override
    api_url = config.get('api_url') or LOCAL_LLM_API_URL
    clean_url = api_url.rstrip('/')
    if clean_url.endswith("/chat/completions"):
        api_url = clean_url
    elif clean_url.endswith("/v1"):
        api_url = f"{clean_url}/chat/completions"
    else:
        # Assume base URL provided without /v1
        api_url = f"{clean_url}/v1/chat/completions"

    api_key = config.get('apiKey')
    # Use prompt-specific temp if passed, else config temp, else default
    temp = config.get('temperature', temperature)

    history = conversation_history or []

    # --- LOCAL PROVIDER ---
    if provider == 'local':
        headers = {"Content-Type": "application/json"}
        
        # Construct Messages
        messages = []
        
        # 1. Add System Prompt (If provided in config)
        system_prompt = config.get("system_prompt")
        if system_prompt:
             messages.append({"role": "system", "content": system_prompt})
             
        # 2. Add History
        for entry in history:
            if entry.get('content'):
                messages.append({"role": entry.get('role'), "content": entry.get('content')})
        
        # 3. Add Current Prompt
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "stream": False
        }

        try:
            logger.info(f"Sending request to {api_url} using model {model_name}")
            response = requests.post(api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Local LLM Error: {e}")
            raise

    # --- GEMINI PROVIDER ---
    elif provider == 'gemini':
        if not api_key:
            raise ValueError("Gemini API Key is missing")
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Gemini History Format conversion could happen here
        # For simplicity, we just concatenate for now or use simple generation
        # Real impl would map 'user'/'assistant' to 'user'/'model'
        
        try:
            # Simple fallback: Concatenate prompt (Gemini handles context well)
            full_prompt = prompt
            if history:
                hist_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
                full_prompt = f"{hist_text}\nUser: {prompt}"

            resp = model.generate_content(full_prompt)
            return resp.text
        except Exception as e:
            logger.error(f"Gemini Error: {e}")
            raise

    return "Error: Unknown Provider"

# Used by query_data.py for refinement
def generate_refined_search_query(query_text, history, llm_config):
    # Reuse the get_llm_response but strip thinking tags
    prompt = f"Refine this query for search: {query_text}. Output ONLY the query."
    resp = get_llm_response(prompt, llm_config=llm_config, temperature=0.3)
    return THINKING_TAG_PATTERN.sub('', resp).strip()