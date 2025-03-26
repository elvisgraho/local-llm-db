import requests
from langchain.prompts import ChatPromptTemplate
from query.global_vars import LOCAL_LLM_API_URL, LOCAL_MAIN_MODEL
from query.templates import QUERY_OPTIMIZATION_TEMPLATE
import re
import time
import json
import logging
from typing import Optional
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_llm_response(prompt: str, temperature: float = 0.7) -> str:
    """Helper function to get response from LLM with retries.
    
    Args:
        prompt (str): The prompt to send to the LLM.
        temperature (float): The temperature parameter for the LLM.
        
    Returns:
        str: The response from the LLM.
        
    Raises:
        RequestException: If there's an error communicating with the LLM service.
        ValueError: If the response is invalid or empty.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }
    
    try:
        response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        
        if not response_data.get("choices"):
            raise ValueError("No choices in LLM response")
            
        content = response_data["choices"][0].get("message", {}).get("content", "")
        if not content:
            raise ValueError("Empty response from LLM")
            
        return content
        
    except RequestException as e:
        logger.error(f"Error calling LLM API: {str(e)}")
        raise

def optimize_query(query_text: str) -> str:
    """Optimize the query using a separate LLM call.
    
    Args:
        query_text (str): The original query text.
        
    Returns:
        str: The optimized query text.
    """
    try:
        prompt_template = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_TEMPLATE)
        prompt = prompt_template.format(query=query_text)
        # Use a different model for optimization with lower temperature
        optimized_query = get_llm_response(prompt, temperature=0.3).strip()

        # Filter out <think> tags if present
        if optimized_query:
            # Remove any <think> tags and their contents
            optimized_query = re.sub(r'<think>.*?</think>', '', optimized_query, flags=re.DOTALL).strip()
            
        return optimized_query if optimized_query else query_text
        
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        return query_text 