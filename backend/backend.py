from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
import sys
import os
import time
import logging
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query.query_data import query_direct, query_graph, query_kag, query_lightrag, query_rag
from query.llm_service import optimize_query
from query.data_service import data_service
import traceback
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import requests # To call local LLM API
from query.global_vars import LOCAL_LLM_API_URL # Get local API URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Debug mode flag
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Frontend routes
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# API routes
@app.route('/api/query', methods=['POST'])
def handle_query():
    start_time = time.time()
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    try:
        data = request.json
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400

        query_text = data.get('query_text', '')
        query_mode = data.get('mode', 'rag')
        optimize = data.get('optimize', False)
        hybrid = data.get('hybrid', False)
        llm_config = data.get('llm_config', {}) # Extract llm_config
        # Extract conversation history if present, default to None or empty list
        conversation_history = data.get('conversation_history', None)

        # Basic validation for llm_config
        provider = llm_config.get('provider')
        # Expect camelCase from frontend payload
        model_name = llm_config.get('modelName')
        api_key = llm_config.get('apiKey') # Expect camelCase 'apiKey'

        if not provider or provider not in ['local', 'gemini']:
            return jsonify({
                'error': 'Invalid or missing LLM provider in llm_config. Must be "local" or "gemini".',
                'status': 'error'
            }), 400

        # Check model_name (which is now modelName from payload)
        if not model_name:
             logging.warning(f"Query {request_id}: LLM model name (modelName) not provided in llm_config.")

        # Check api_key (which is now apiKey from payload)
        if provider == 'gemini' and not api_key:
            # Log warning if apiKey is missing for gemini
            logging.warning(f"Query {request_id}: Gemini provider selected but API key (apiKey) is missing in llm_config.")

        if not query_text:
            return jsonify({
                'error': 'Query text is required',
                'status': 'error'
            }), 400

        if query_mode not in ['rag', 'direct', 'graph', 'lightrag', 'kag']:
            return jsonify({
                'error': 'Invalid query mode. Must be one of: rag, direct, graph, lightrag, kag',
                'status': 'error'
            }), 400

        logging.info(f"Processing query {request_id}: mode={query_mode}, optimize={optimize}, hybrid={hybrid}, provider={provider}, model={model_name}")

        # Optimize query if requested
        # TODO: Pass llm_config to optimize_query if needed in the future
        if optimize:
            try:
                optimize_start = time.time()
                optimized_query = optimize_query(query_text)
                optimize_time = time.time() - optimize_start

                if not optimized_query or optimized_query.startswith('<think>'):
                    raise ValueError("Incomplete query optimization response")

                logging.info(f"Query {request_id} optimized in {optimize_time:.2f}s")
            except Exception as e:
                logging.error(f"Error during query optimization: {str(e)}")
                optimized_query = query_text
        else:
            optimized_query = query_text

        # Call appropriate query function based on mode
        try:
            query_start = time.time()
            # Pass llm_config and conversation_history to the query functions
            if query_mode == 'direct':
                query_response = query_direct(optimized_query, llm_config=llm_config, conversation_history=conversation_history)
            elif query_mode == 'graph':
                query_response = query_graph(optimized_query, hybrid, llm_config=llm_config, conversation_history=conversation_history)
            elif query_mode == 'lightrag':
                query_response = query_lightrag(optimized_query, hybrid, llm_config=llm_config, conversation_history=conversation_history)
            elif query_mode == 'kag':
                query_response = query_kag(optimized_query, hybrid, llm_config=llm_config, conversation_history=conversation_history)
            else: # Default to RAG
                query_response = query_rag(optimized_query, hybrid, llm_config=llm_config, conversation_history=conversation_history)

            query_time = time.time() - query_start

            # Validate query response
            if not query_response or not query_response.get('text'):
                raise ValueError("Empty or invalid query response")

            total_time = time.time() - start_time

            response = {
                'status': 'success',
                'data': query_response,
                'stats': {
                    'total_time': round(total_time, 2),
                    'query_time': round(query_time, 2),
                    'optimize_time': round(optimize_time if optimize else 0, 2),
                    'mode': query_mode,
                    'hybrid': hybrid
                }
            }

            if DEBUG_MODE:
                response['debug'] = {
                    'request_id': request_id,
                    'original_query': query_text,
                    'optimized_query': optimized_query if optimize else None
                }

            logging.info(f"Query {request_id} completed in {total_time:.2f}s")
            return jsonify(response)

        except Exception as e:
            error_msg = f"Error during query processing: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            status_code = 500 # Explicitly define status code
            response_data = {
                'error': error_msg,
                'status': 'error',
                'request_id': request_id,
                'traceback': traceback.format_exc() if DEBUG_MODE else None
            }
            logging.error(f"Query {request_id}: Returning error response with status {status_code}: {response_data}") # Add log here
            return jsonify(response_data), status_code

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'request_id': request_id,
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

# --- New Endpoint for Listing Models ---
@app.route('/api/models/<provider>', methods=['POST']) # Use POST to send API key securely
def list_models(provider):
    """API endpoint to list available models for a given provider."""
    api_key = request.json.get('apiKey') if provider == 'gemini' else None
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # For logging
    logging.info(f"Request {request_id}: Fetching models for provider: {provider}")

    models = []
    error_message = None
    status_code = 200

    try:
        if provider == 'gemini':
            if not api_key:
                raise ValueError("API key is required to list Gemini models.")

            try:
                genai.configure(api_key=api_key)
                # List models and filter for those supporting 'generateContent'
                for m in genai.list_models():
                    # Check based on supported_generation_methods if available
                    if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                         # Prefer 'models/model-name' format if available, else use name
                         model_id = getattr(m, 'name', None) # Usually 'models/...'
                         if model_id:
                             models.append(model_id)
                    # Fallback check if supported_generation_methods isn't present (older versions?)
                    elif not hasattr(m, 'supported_generation_methods'):
                         model_id = getattr(m, 'name', None)
                         if model_id:
                              models.append(model_id)
                              logging.warning(f"Model {model_id} missing supported_generation_methods, including anyway.")


                if not models:
                     logging.warning(f"Request {request_id}: No Gemini models supporting 'generateContent' found with the provided key.")
                     # Don't raise an error, just return empty list, maybe key is valid but has no models?

            except google_exceptions.PermissionDenied as e:
                logging.error(f"Request {request_id}: Gemini API Permission Denied while listing models: {e}")
                raise ValueError(f"Invalid Gemini API Key or insufficient permissions.") from e
            except Exception as e:
                logging.error(f"Request {request_id}: Error listing Gemini models: {e}", exc_info=True)
                raise ValueError(f"Failed to retrieve Gemini models: {e}") from e


        elif provider == 'local':
            # Assuming LM Studio compatible API at LOCAL_LLM_API_URL/v1/models
            # Construct the models URL carefully, removing potential /chat/completions suffix
            base_url = LOCAL_LLM_API_URL
            if base_url.endswith('/chat/completions'):
                base_url = base_url[:-len('/chat/completions')]
            elif base_url.endswith('/'):
                 base_url = base_url[:-1]
            models_url = f"{base_url}/v1/models"
            logging.info(f"Request {request_id}: Attempting to fetch local models from: {models_url}")
            try:
                response = requests.get(models_url, timeout=10) # Add timeout
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                # Structure is often {'data': [{'id': 'model-name', ...}, ...]}
                if 'data' in data and isinstance(data['data'], list):
                    models = [model.get('id') for model in data['data'] if model.get('id')]
                else:
                     logging.warning(f"Request {request_id}: Unexpected format from local models endpoint ({models_url}): {data}")
                     raise ValueError("Received unexpected format from local models endpoint.")

                if not models:
                     logging.warning(f"Request {request_id}: No models found at local endpoint: {models_url}")
                     # Return empty list, server might be running but have no models loaded

            except requests.exceptions.RequestException as e:
                logging.error(f"Request {request_id}: Error connecting to local LLM models endpoint ({models_url}): {e}")
                raise ValueError(f"Could not connect to local LLM at {models_url}. Is the server running?") from e
            except Exception as e:
                logging.error(f"Request {request_id}: Error parsing local LLM models response: {e}", exc_info=True)
                raise ValueError(f"Failed to retrieve or parse local models: {e}") from e

        else:
            raise ValueError(f"Unsupported provider for listing models: {provider}")

    except ValueError as e:
        error_message = str(e)
        status_code = 400 # Bad request (e.g., missing key, unsupported provider)
        logging.error(f"Request {request_id}: Validation error listing models for {provider}: {error_message}")
    except Exception as e:
        error_message = f"An unexpected server error occurred: {str(e)}"
        status_code = 500 # Internal server error
        logging.error(f"Request {request_id}: Unexpected error listing models for {provider}: {error_message}", exc_info=True)

    response_data = {
        'provider': provider,
        'models': sorted(list(set(models))) # Sort and remove duplicates
    }
    if error_message:
        response_data['error'] = error_message
        response_data['status'] = 'error'
    else:
        response_data['status'] = 'success'

    logging.info(f"Request {request_id}: Returning {len(response_data['models'])} models for {provider}. Status: {status_code}")
    return jsonify(response_data), status_code
# --- End New Endpoint ---

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'debug_mode': DEBUG_MODE
    })

@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the data service cache"""
    try:
        data_service.clear_cache()
        logging.info("Cache cleared successfully")
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        error_msg = f"Error clearing cache: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'error': error_msg,
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

if __name__ == '__main__':
    logging.info("🚀 Initializing server...")
    logging.info(f"Debug mode: {DEBUG_MODE}")
    logging.info("🚀 Server is running on http://127.0.0.1:5000/")
    logging.info("Available endpoints:")
    logging.info("- POST /api/query - Query the RAG system")
    logging.info("- POST /api/models/<provider> - List available models (gemini/local)") # Added log
    logging.info("- GET /api/health - Health check endpoint")
    logging.info("- POST /api/clear_cache - Clear data service cache")
    logging.info("- GET / - Frontend application")
    serve(app, host="0.0.0.0", port=5000, threads=10)
