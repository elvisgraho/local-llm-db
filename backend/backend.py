from flask import Flask, request, jsonify
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backend.log'),
        logging.StreamHandler()
    ]
)

app = Flask(__name__)
CORS(app)

# Debug mode flag
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

@app.route('/query', methods=['POST'])
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
        
        logging.info(f"Processing query {request_id}: mode={query_mode}, optimize={optimize}, hybrid={hybrid}")
        
        # Optimize query if requested
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
            if query_mode == 'direct':
                query_response = query_direct(optimized_query)
            elif query_mode == 'graph':
                query_response = query_graph(optimized_query, hybrid)
            elif query_mode == 'lightrag':
                query_response = query_lightrag(optimized_query, hybrid)
            elif query_mode == 'kag':
                query_response = query_kag(optimized_query, hybrid)
            else:
                query_response = query_rag(optimized_query, hybrid)
            
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
            return jsonify({
                'error': error_msg,
                'status': 'error',
                'request_id': request_id,
                'traceback': traceback.format_exc() if DEBUG_MODE else None
            }), 500

    except Exception as e:
        error_msg = f"Error processing query: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        return jsonify({
            'error': error_msg,
            'status': 'error',
            'request_id': request_id,
            'traceback': traceback.format_exc() if DEBUG_MODE else None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running',
        'timestamp': datetime.now().isoformat(),
        'debug_mode': DEBUG_MODE
    })

@app.route('/clear_cache', methods=['POST'])
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
    logging.info("- POST /query - Query the RAG system")
    logging.info("- GET /health - Health check endpoint")
    logging.info("- POST /clear_cache - Clear data service cache")
    serve(app, host="0.0.0.0", port=5000, threads=10)
