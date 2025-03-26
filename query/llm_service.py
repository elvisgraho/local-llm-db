import requests
from langchain.prompts import ChatPromptTemplate
from query.global_vars import LOCAL_LLM_API_URL, LOCAL_MAIN_MODEL
from query.templates import QUERY_OPTIMIZATION_TEMPLATE

def get_llm_response(prompt: str, temperature: float = 0.7) -> str:
    """Helper function to get response from LLM."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
    response_data = response.json()
    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

def optimize_query(query_text: str) -> str:
    """Optimize the query using a separate LLM call."""
    prompt_template = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_TEMPLATE)
    prompt = prompt_template.format(query=query_text)
    
    # Use a different model for optimization
    optimized_query = get_llm_response(prompt, temperature=0.3).strip()
    return optimized_query if optimized_query else query_text 