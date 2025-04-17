import requests
from langchain.prompts import ChatPromptTemplate
from query.global_vars import LOCAL_LLM_API_URL, LOCAL_MAIN_MODEL
from query.templates import QUERY_OPTIMIZATION_TEMPLATE
import re
import time
import json
import logging
from typing import List, Optional, Dict
from requests.exceptions import RequestException
from tenacity import retry, stop_after_attempt, wait_exponential
from google.api_core import exceptions as google_exceptions # Import google api exceptions
import google.generativeai as genai

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
def get_llm_response(
    prompt: str,
    llm_config: Optional[Dict] = None,
    temperature: float = 0.7,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Helper function to get response from LLM with retries, supporting different providers and conversation history.

    Args:
        prompt (str): The current user prompt to send to the LLM.
        llm_config (Optional[Dict]): Configuration for the LLM provider.
                                     Expected keys: 'provider' ('local' or 'gemini'),
                                                    'modelName', 'apiKey' (for gemini).
        temperature (float): The temperature parameter for the LLM.
        conversation_history (Optional[List[Dict[str, str]]]): Previous conversation turns,
                                                                e.g., [{'role': 'user', 'content': '...'},
                                                                       {'role': 'assistant', 'content': '...'}].

    Returns:
        str: The response from the LLM.
        
    Raises:
        RequestException: If there's an error communicating with the LLM service.
        ValueError: If the response is invalid or empty, or config is invalid.
    """
    # Determine provider and settings
    config = llm_config or {}
    provider = config.get('provider', 'local')
    # Expect camelCase from the config dict passed down
    model_name = config.get('modelName')
    api_key = config.get('apiKey') # Expect camelCase 'apiKey'
    history = conversation_history or [] # Ensure history is a list

    logger.info(f"Getting LLM response using provider: {provider}, model: {model_name or 'default'}, history_len: {len(history)}")

    if provider == 'local':
        # Use local LLM (OpenAI compatible API)
        local_model = model_name or LOCAL_MAIN_MODEL
        headers = {"Content-Type": "application/json"}

        # Construct messages including history
        messages = []
        for entry in history:
            # Ensure role is 'user' or 'assistant' (or 'system' if needed later)
            role = entry.get('role')
            content = entry.get('content')
            if role in ['user', 'assistant'] and content:
                 messages.append({"role": role, "content": content})
            else:
                 logger.warning(f"Skipping invalid history entry: {entry}")
        messages.append({"role": "user", "content": prompt}) # Add current prompt

        payload = {
            "model": local_model,
            "messages": messages,
            "temperature": temperature
        }

        try:
            response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            
            if not response_data.get("choices"):
                raise ValueError("No choices in local LLM response")
                
            content = response_data["choices"][0].get("message", {}).get("content", "")
            if not content:
                raise ValueError("Empty response from local LLM")
                
            return content
            
        except RequestException as e:
            logger.error(f"Error calling local LLM API ({LOCAL_LLM_API_URL}): {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing local LLM response: {str(e)}")
            raise ValueError(f"Failed to process local LLM response: {e}")

    elif provider == 'gemini':
        # Use Gemini API
        if not api_key:
            logger.error("Gemini provider selected, but API key is missing in llm_config.")
            raise ValueError("API key required for Gemini provider is missing.")
        # Check model_name (which is now modelName from config)
        if not model_name:
            logger.error("Gemini provider selected, but modelName is missing in llm_config.")
            raise ValueError("Model name (modelName) required for Gemini provider is missing.")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(temperature=temperature)

            # Format history for Gemini API (list of content objects)
            # Gemini expects alternating user/model roles. 'assistant' maps to 'model'.
            gemini_history = []
            for entry in history:
                role = entry.get('role')
                content = entry.get('content')
                if role == 'user' and content:
                    gemini_history.append({'role': 'user', 'parts': [content]})
                elif role == 'assistant' and content:
                    gemini_history.append({'role': 'model', 'parts': [content]})
                else:
                    logger.warning(f"Skipping invalid history entry for Gemini: {entry}")

            # Start a chat session if history exists
            if gemini_history:
                 chat = model.start_chat(history=gemini_history)
                 response = chat.send_message(prompt, generation_config=generation_config)
            else:
                 # Send single message if no history
                 response = model.generate_content(prompt, generation_config=generation_config)

            # Accessing response text might differ based on Gemini library version
            # Check response.text or response.parts
            if hasattr(response, 'text'):
                content = response.text
            elif hasattr(response, 'parts') and response.parts:
                content = "".join(part.text for part in response.parts)
            else:
                # Attempt to access prompt_feedback if generation failed
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback and hasattr(feedback, 'block_reason'):
                     raise ValueError(f"Gemini content generation blocked: {feedback.block_reason}")
                else:
                     raise ValueError("Could not extract text from Gemini response and no block reason found.")

            if not content:
                raise ValueError("Empty response from Gemini LLM")
                
            return content
            
        except google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini API Permission Denied (check API key?): {str(e)}")
            raise ValueError(f"Gemini API Permission Denied: {e}")
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini API Invalid Argument (check model name or parameters?): {str(e)}")
            raise ValueError(f"Gemini API Invalid Argument: {e}")
        except Exception as e:
            # Catch potential exceptions from the genai library or other issues
            logger.error(f"Error calling Gemini API (model: {model_name}): {str(e)}")
            raise ValueError(f"Failed to get response from Gemini: {e}") # Keep original error message propagation
            
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

def optimize_query(query_text: str) -> str:
        raise

# TODO: Decide if optimize_query should also use the configured LLM or always use local.
# Currently, it always uses the local LLM for optimization.
def optimize_query(query_text: str, llm_config: Optional[Dict] = None) -> str:
    """Optimize the query using a separate LLM call (currently defaults to local).
    
    Args:
        query_text (str): The original query text.
        llm_config (Optional[Dict]): LLM configuration (currently unused here, but kept for signature consistency).
        query_text (str): The original query text.
        
    Returns:
        str: The optimized query text.
    """
    try:
        prompt_template = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_TEMPLATE)
        prompt = prompt_template.format(query=query_text)
        # Use a different model for optimization with lower temperature
        # For now, explicitly call with provider='local' for optimization step
        # Pass a minimal config forcing local provider
        optimization_llm_config = {'provider': 'local', 'model_name': LOCAL_MAIN_MODEL}
        optimized_query = get_llm_response(prompt, llm_config=optimization_llm_config, temperature=0.3).strip()

        # Filter out <think> and <reasoning> tags if present
        if optimized_query:
            # Remove any <think> and <reasoning> tags and their contents
            optimized_query = re.sub(r'<think>.*?</think>', '', optimized_query, flags=re.DOTALL)
            optimized_query = re.sub(r'<reasoning>.*?</reasoning>', '', optimized_query, flags=re.DOTALL)
            optimized_query = optimized_query.strip()
            
        return optimized_query if optimized_query else query_text
        
    except Exception as e:
        logger.error(f"Error optimizing query: {str(e)}")
        return query_text 