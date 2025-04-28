from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from waitress import serve
import sys
import os
import time
import logging
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query.query_data import query_direct, query_kag, query_lightrag, query_rag
from query.data_service import data_service
from query.database_paths import list_available_dbs, DEFAULT_DB_NAME # Import list_available_dbs
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
        rag_type = data.get('rag_type', 'rag')
        db_name = data.get('db_name', DEFAULT_DB_NAME)
        optimize = data.get('optimize', False)
        hybrid = data.get('hybrid', False)
        llm_config = data.get('llm_config', {}) # Keep llm_config
        # Get full conversation history from frontend
        conversation_history = data.get('conversation_history', None)

        # Basic validation for llm_config
        provider = llm_config.get('provider')
        model_name = llm_config.get('modelName') # Still expect camelCase from frontend
        api_key = llm_config.get('apiKey')

        # --- llm_config validation remains the same ---
        if not provider or provider not in ['local', 'gemini']:
            return jsonify({
                'error': 'Invalid or missing LLM provider in llm_config. Must be "local" or "gemini".',
                'status': 'error'
            }), 400
        if not model_name:
             logging.warning(f"Query {request_id}: LLM model name (modelName) not provided in llm_config.")
        if provider == 'gemini' and not api_key:
            logging.warning(f"Query {request_id}: Gemini provider selected but API key (apiKey) is missing in llm_config.")
        # --- End llm_config validation ---

        if not query_text:
            return jsonify({'error': 'Query text is required', 'status': 'error'}), 400

        valid_rag_types = ['rag', 'direct', 'lightrag', 'kag']
        if rag_type not in valid_rag_types:
            return jsonify({'error': f'Invalid rag_type. Must be one of: {", ".join(valid_rag_types)}', 'status': 'error'}), 400

        if rag_type != 'direct' and not db_name:
             return jsonify({'error': f'db_name is required for rag_type "{rag_type}"', 'status': 'error'}), 400

        # Log received parameters (excluding context_length now)
        logging.info(f"Processing query {request_id}: rag_type={rag_type}, db_name={db_name}, optimize={optimize}, hybrid={hybrid}, provider={provider}, model={model_name}")

        # Call appropriate query function based on mode
        try:
            query_start = time.time()
            # Pass llm_config and the full conversation_history
            # The query functions will handle context length lookup and history truncation
            if rag_type == 'direct':
                # Direct query doesn't need context length for retrieval/truncation here
                query_response = query_direct(query_text, llm_config=llm_config, conversation_history=conversation_history)
            elif rag_type == 'lightrag':
                query_response = query_lightrag(query_text, hybrid, rag_type=rag_type, db_name=db_name, llm_config=llm_config, conversation_history=conversation_history, optimize=optimize)
            elif rag_type == 'kag':
                query_response = query_kag(query_text, hybrid, rag_type=rag_type, db_name=db_name, llm_config=llm_config, conversation_history=conversation_history, optimize=optimize)
            else: # Default to RAG
                query_response = query_rag(query_text, hybrid, rag_type=rag_type, db_name=db_name, llm_config=llm_config, conversation_history=conversation_history, optimize=optimize)

            query_time = time.time() - query_start

            if not query_response or not query_response.get('text'):
                raise ValueError("Empty or invalid query response")

            total_time = time.time() - start_time

            response = {
                'status': 'success',
                'data': query_response, # Includes estimated_context_tokens from query_data
                'stats': {
                    'total_time': round(total_time, 2),
                    'query_time': round(query_time, 2),
                    'rag_type': rag_type,
                    'db_name': db_name if rag_type != 'direct' else None,
                    'hybrid': hybrid
                }
            }

            if DEBUG_MODE:
                response['debug'] = {
                    'request_id': request_id,
                    'original_query': query_text
                }

            logging.info(f"Query {request_id} completed in {total_time:.2f}s")
            return jsonify(response)

        except Exception as e:
            error_msg = f"Error during query processing: {str(e)}"
            logging.error(f"{error_msg}\n{traceback.format_exc()}")
            status_code = 500
            response_data = {
                'error': error_msg,
                'status': 'error',
                'request_id': request_id,
                'traceback': traceback.format_exc() if DEBUG_MODE else None
            }
            logging.error(f"Query {request_id}: Returning error response with status {status_code}: {response_data}")
            return jsonify(response_data), status_code

    except Exception as e:
        error_msg = f"Error processing query request: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'request_id': request_id,
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

# --- Endpoint for Listing Models ---
@app.route('/api/models/<provider>', methods=['POST'])
def list_models(provider):
    """API endpoint to list available models for a given provider."""
    api_key = request.json.get('apiKey') if provider == 'gemini' else None
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
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
                for m in genai.list_models():
                    if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods:
                         model_id = getattr(m, 'name', None)
                         if model_id:
                             models.append(model_id)
                    elif not hasattr(m, 'supported_generation_methods'):
                         model_id = getattr(m, 'name', None)
                         if model_id:
                              models.append(model_id)
                              logging.warning(f"Model {model_id} missing supported_generation_methods, including anyway.")
                if not models:
                     logging.warning(f"Request {request_id}: No Gemini models supporting 'generateContent' found with the provided key.")
            except google_exceptions.PermissionDenied as e:
                logging.error(f"Request {request_id}: Gemini API Permission Denied while listing models: {e}")
                raise ValueError(f"Invalid Gemini API Key or insufficient permissions.") from e
            except Exception as e:
                logging.error(f"Request {request_id}: Error listing Gemini models: {e}", exc_info=True)
                raise ValueError(f"Failed to retrieve Gemini models: {e}") from e

        elif provider == 'local':
            base_url = LOCAL_LLM_API_URL
            if base_url.endswith('/chat/completions'):
                base_url = base_url[:-len('/chat/completions')]
            elif base_url.endswith('/'):
                 base_url = base_url[:-1]
            if base_url.endswith('/v1'):
                models_url = f"{base_url}/models"
            else:
                models_url = f"{base_url}/v1/models"
            logging.info(f"Request {request_id}: Attempting to fetch local models from: {models_url}")
            try:
                response = requests.get(models_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if 'data' in data and isinstance(data['data'], list):
                    models = [model.get('id') for model in data['data'] if model.get('id')]
                else:
                     logging.warning(f"Request {request_id}: Unexpected format from local models endpoint ({models_url}): {data}")
                     raise ValueError("Received unexpected format from local models endpoint.")
                if not models:
                     logging.warning(f"Request {request_id}: No models found at local endpoint: {models_url}")
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
        status_code = 400
        logging.error(f"Request {request_id}: Validation error listing models for {provider}: {error_message}")
    except Exception as e:
        error_message = f"An unexpected server error occurred: {str(e)}"
        status_code = 500
        logging.error(f"Request {request_id}: Unexpected error listing models for {provider}: {error_message}", exc_info=True)

    response_data = {
        'provider': provider,
        'models': sorted(list(set(models)))
    }
    if error_message:
        response_data['error'] = error_message
        response_data['status'] = 'error'
    else:
        response_data['status'] = 'success'

    logging.info(f"Request {request_id}: Returning {len(response_data['models'])} models for {provider}. Status: {status_code}")
    return jsonify(response_data), status_code
# --- End Listing Models Endpoint ---

# --- Endpoint for Listing Databases ---
@app.route('/api/list_dbs/<rag_type>', methods=['GET'])
def get_list_dbs(rag_type):
    request_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    logging.info(f"Request {request_id}: Listing databases for rag_type: {rag_type}")
    valid_rag_types = ['rag', 'kag', 'lightrag']
    if rag_type not in valid_rag_types:
        logging.warning(f"Request {request_id}: Invalid rag_type '{rag_type}' requested.")
        return jsonify({'error': f'Invalid rag_type. Must be one of: {", ".join(valid_rag_types)}', 'status': 'error'}), 400
    try:
        db_names = list_available_dbs(rag_type)
        logging.info(f"Request {request_id}: Found {len(db_names)} databases for {rag_type}: {db_names}")
        return jsonify({'rag_type': rag_type, 'databases': db_names, 'status': 'success'}), 200
    except Exception as e:
        error_message = f"An unexpected server error occurred while listing databases for {rag_type}: {str(e)}"
        logging.error(f"Request {request_id}: {error_message}", exc_info=True)
        return jsonify({'error': error_message, 'status': 'error'}), 500
# --- End Listing Databases Endpoint ---

# --- Health Check Endpoint ---
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'debug_mode': DEBUG_MODE
    })
# --- End Health Check Endpoint ---

# --- Clear Cache Endpoint ---
@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    try:
        data_service.clear_cache()
        logging.info("Cache cleared successfully")
        return jsonify({'status': 'success', 'message': 'Cache cleared successfully', 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        error_msg = f"Error clearing cache: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'error': error_msg, 'traceback': traceback.format_exc() if DEBUG_MODE else None}), 500
# --- End Clear Cache Endpoint ---

if __name__ == '__main__':
    logging.info("🚀 Initializing server...")
    logging.info(f"Debug mode: {DEBUG_MODE}")
    logging.info("🚀 Server is running on http://127.0.0.1:5000/")
    logging.info("Available endpoints:")
    logging.info("- POST /api/query - Query the RAG system")
    logging.info("- POST /api/models/<provider> - List available models (gemini/local)")
    logging.info("- GET /api/list_dbs/<rag_type> - List available databases (rag/kag/lightrag)")
    logging.info("- GET /api/health - Health check endpoint")
    logging.info("- POST /api/clear_cache - Clear data service cache")
    logging.info("- GET / - Frontend application")
    serve(app, host="0.0.0.0", port=5000, threads=10)
