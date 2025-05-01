import re
import requests
from langchain.prompts import ChatPromptTemplate
from query.global_vars import LOCAL_LLM_API_URL, LOCAL_MAIN_MODEL
from query.templates import REFINE_SEARCH_QUERY_TEMPLATE # Updated import
import logging
from typing import List, Optional, Dict, Tuple # Added Tuple
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

# --- Constants ---
DEFAULT_CONTEXT_LENGTH = 8000
TOKEN_ESTIMATION_FACTOR_SERVICE = 4 # Simple approximation for backend

# --- Local Token Estimation Helper ---
def _estimate_tokens_service(text: str) -> int:
    """Estimates the number of tokens in a string (backend version)."""
    if not text or not isinstance(text, str):
        return 0
    return len(text) // TOKEN_ESTIMATION_FACTOR_SERVICE

# --- Context Length Retrieval ---
def get_model_context_length(llm_config: Optional[Dict] = None) -> int:
    """Gets the context length for the specified model configuration.

    Args:
        llm_config (Optional[Dict]): Configuration containing provider, modelName,
                                     and potentially 'contextLength' for local models.

    Returns:
        int: The context length for the model.
    """
    config = llm_config or {}
    provider = config.get('provider', 'local')
    model_name = config.get('modelName', '') # Expect camelCase

    if provider == 'local':
        # For local models, check if a specific context length is provided in the config
        custom_length = config.get('contextLength') # Expect camelCase from frontend
        if isinstance(custom_length, int) and custom_length > 0:
            logger.debug(f"Using custom context length for local model {model_name}: {custom_length}")
            return custom_length
        else:
            logger.debug(f"Using default context length for local model {model_name}: {DEFAULT_CONTEXT_LENGTH}")
            return DEFAULT_CONTEXT_LENGTH
    elif provider == 'gemini':
        # For Gemini, assume a very large context window. The API handles its own limits.
        # Using 1,000,000 as a practical upper bound for our internal logic.
        large_context_length = 1_000_000
        logger.debug(f"Using large context length for Gemini model {model_name}: {large_context_length}")
        return large_context_length
    else:
        # Fallback for other or unknown providers
        logger.warning(f"Unknown or unsupported provider '{provider}' for context length lookup. Using default: {DEFAULT_CONTEXT_LENGTH}")
        return DEFAULT_CONTEXT_LENGTH

# --- History Truncation ---
def truncate_history(
    conversation_history: Optional[List[Dict[str, str]]],
    max_tokens: int
) -> Tuple[List[Dict[str, str]], int]:
    """Truncates conversation history to fit within a maximum token limit.

    Args:
        conversation_history (Optional[List[Dict[str, str]]]): The full history.
        max_tokens (int): The maximum allowed estimated tokens for the history.

    Returns:
        Tuple[List[Dict[str, str]], int]: A tuple containing the truncated
                                          history list and its estimated token count.
    """
    if not conversation_history or max_tokens <= 0:
        return [], 0

    truncated_history = []
    current_token_count = 0

    # Iterate backwards (newest to oldest)
    for i in range(len(conversation_history) - 1, -1, -1):
        message = conversation_history[i]
        role = message.get('role')
        content = message.get('content', '')

        if not role or not content: # Skip invalid entries
            continue

        # Estimate tokens for this message (role + content + separators)
        # Use a simple format for estimation
        message_text_estimate = f"{role}: {content}\n"
        message_tokens = _estimate_tokens_service(message_text_estimate)

        if current_token_count + message_tokens <= max_tokens:
            # Add to the beginning of our list (will be reversed later)
            truncated_history.insert(0, message)
            current_token_count += message_tokens
        else:
            # Stop adding messages if the limit is exceeded
            logger.info(f"History truncated: Stopped after {len(truncated_history)} messages. Estimated tokens: {current_token_count}/{max_tokens}")
            break # Stop iterating

    return truncated_history, current_token_count


# --- LLM Interaction ---
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_llm_response(
    prompt: str,
    llm_config: Optional[Dict] = None,
    temperature: float = 0.7,
    conversation_history: Optional[List[Dict[str, str]]] = None # This is now potentially truncated history
) -> str:
    """Helper function to get response from LLM with retries, supporting different providers and conversation history.
       Assumes conversation_history has already been truncated if necessary.
    """
    config = llm_config or {}
    provider = config.get('provider', 'local')
    model_name = config.get('modelName')
    api_key = config.get('apiKey')
    # Use the provided history directly (already truncated by caller if needed for context building)
    history = conversation_history or []

    logger.info(f"Getting LLM response using provider: {provider}, model: {model_name or 'default'}, history_len: {len(history)}")

    if provider == 'local':
        local_model = model_name or LOCAL_MAIN_MODEL
        headers = {"Content-Type": "application/json"}
        messages = []
        for entry in history:
            role = entry.get('role')
            content = entry.get('content')
            if role in ['user', 'assistant'] and content:
                 messages.append({"role": role, "content": content})
            else: logger.warning(f"Skipping invalid history entry: {entry}")
        messages.append({"role": "user", "content": prompt})

        payload = {"model": local_model, "messages": messages, "temperature": temperature}

        try:
            response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if not response_data.get("choices"): raise ValueError("No choices in local LLM response")
            content = response_data["choices"][0].get("message", {}).get("content", "")
            if not content: raise ValueError("Empty response from local LLM")
            return content
        except RequestException as e:
            logger.error(f"Error calling local LLM API ({LOCAL_LLM_API_URL}): {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error processing local LLM response: {str(e)}")
            raise ValueError(f"Failed to process local LLM response: {e}")

    elif provider == 'gemini':
        if not api_key: raise ValueError("API key required for Gemini provider is missing.")
        if not model_name: raise ValueError("Model name (modelName) required for Gemini provider is missing.")

        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            generation_config = genai.types.GenerationConfig(temperature=temperature)

            gemini_history = []
            for entry in history:
                role = entry.get('role')
                content = entry.get('content')
                if role == 'user' and content: gemini_history.append({'role': 'user', 'parts': [content]})
                elif role == 'assistant' and content: gemini_history.append({'role': 'model', 'parts': [content]})
                else: logger.warning(f"Skipping invalid history entry for Gemini: {entry}")

            # --- Gemini History Truncation (Internal) ---
            # Gemini library might handle context length internally, but we can add a safety check
            # Note: This is a secondary check; primary truncation happens in query_data.py
            # context_length_gemini = get_model_context_length(llm_config) # Get length again if needed
            # TODO: Implement Gemini-specific token counting and truncation if library doesn't handle it well.
            # For now, rely on the truncation done before calling get_llm_response.
            # --- End Gemini History Truncation ---


            if gemini_history:
                 chat = model.start_chat(history=gemini_history)
                 response = chat.send_message(prompt, generation_config=generation_config)
            else:
                 response = model.generate_content(prompt, generation_config=generation_config)

            if hasattr(response, 'text'): content = response.text
            elif hasattr(response, 'parts') and response.parts: content = "".join(part.text for part in response.parts)
            else:
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback and hasattr(feedback, 'block_reason'): raise ValueError(f"Gemini content generation blocked: {feedback.block_reason}")
                else: raise ValueError("Could not extract text from Gemini response and no block reason found.")

            if not content: raise ValueError("Empty response from Gemini LLM")
            return content

        except google_exceptions.PermissionDenied as e:
            logger.error(f"Gemini API Permission Denied (check API key?): {str(e)}")
            raise ValueError(f"Gemini API Permission Denied: {e}")
        except google_exceptions.InvalidArgument as e:
            logger.error(f"Gemini API Invalid Argument (check model name or parameters?): {str(e)}")
            raise ValueError(f"Gemini API Invalid Argument: {e}")
        except Exception as e:
            logger.error(f"Error calling Gemini API (model: {model_name}): {str(e)}")
            raise ValueError(f"Failed to get response from Gemini: {e}")

    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider specified: {provider}")

# --- Refined Search Query Generation (for Optimized Pipeline) ---
def generate_refined_search_query( # Renamed function
    query_text: str,
    conversation_history: Optional[List[Dict[str, str]]],
    llm_config: Optional[Dict] = None
) -> str:
    """Generates a refined search query using only history and the original query."""
    try:
        logger.info(f"Generating refined search query for original query using LLM config: {llm_config}")

        # Format history for the template placeholder
        history_str = "No history provided."
        if conversation_history:
            # Simple formatting, newest first as per template description
            # Filter out messages indicating insufficient context
            filter_message = "The provided knowledge context does not contain enough information to answer this question."
            history_lines = [
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in reversed(conversation_history)
                if msg.get('content', '') != filter_message
            ]
            history_str = "\n".join(history_lines) if history_lines else "No relevant history provided." # Handle case where all history is filtered

        prompt_template = ChatPromptTemplate.from_template(REFINE_SEARCH_QUERY_TEMPLATE) # Use new template
        prompt = prompt_template.format(query=query_text, history_placeholder=history_str)
        logger.info(f"Prompt for LLM: {prompt}")

        # Get the refined query from the LLM
        refined_query = get_llm_response(prompt, llm_config=llm_config, temperature=0.7, conversation_history=conversation_history).strip()

        if not refined_query:
            logger.warning("LLM returned an empty refined search query. Falling back to original query.")
            refined_query = query_text # Fallback to original query
            
        tag_pattern = r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>|\s*\[/?(?:thinking|thought|reasoning|think)\b[^\]]*\]\s*|\s*\((?:thinking|thought|reasoning|think)\b[^)]*\)\s*' 
        refined_query = re.sub(tag_pattern, '', refined_query, flags=re.DOTALL).strip()
        # Clean up any remaining whitespace and newlines
        refined_query = refined_query.strip()
        logger.debug(f"Generated Refined Search Query: {refined_query}")
        return refined_query

    except Exception as e:
        logger.error(f"Error generating refined search query: {str(e)}.", exc_info=True)
        # Re-raise the exception to be handled by the caller
        raise