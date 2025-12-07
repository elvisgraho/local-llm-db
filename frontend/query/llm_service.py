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

def get_model_context_length(llm_config: Optional[Dict[str, Any]] = None) -> int:
    if llm_config:
        cw = llm_config.get("context_window")
        if cw: return int(cw)
    return 8192

def truncate_history(history: List[Dict], max_tokens: int) -> tuple:
    current_est = 0
    truncated = []
    # Simple heuristic: 1 word ~= 1.3 tokens -> len / 3 is safe under-estimate
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
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    config = llm_config or {}
    
    provider = config.get('provider', 'local')
    model_name = config.get('modelName') or LOCAL_MAIN_MODEL
    
    # URL construction logic
    api_url = config.get('api_url') or LOCAL_LLM_API_URL
    clean_url = api_url.rstrip('/')
    if clean_url.endswith("/chat/completions"): api_url = clean_url
    elif clean_url.endswith("/v1"): api_url = f"{clean_url}/chat/completions"
    else: api_url = f"{clean_url}/v1/chat/completions"

    api_key = config.get('apiKey')
    temp = config.get('temperature', 0.7)
    history = conversation_history or []

    # --- LOCAL PROVIDER ---
    if provider == 'local':
        headers = {"Content-Type": "application/json"}
        
        messages = []
        
        # 1. System Prompt
        system_prompt = config.get("system_prompt")
        if system_prompt:
             messages.append({"role": "system", "content": system_prompt})
             
        # 2. History
        for entry in history:
            if entry.get('content'):
                messages.append({"role": entry.get('role'), "content": entry.get('content')})
        
        # 3. Current Prompt
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "stream": False,
            "n_keep": 1, 
            "cache_prompt": True
        }

        try:
            # Shortened timeout for direct feedback, but long enough for RAG processing
            response = requests.post(api_url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Local LLM Error: {e}")
            raise

    # --- GEMINI PROVIDER ---
    elif provider == 'gemini':
        if not api_key: raise ValueError("Gemini API Key is missing")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        try:
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
