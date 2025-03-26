from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from query.query_data import query_direct, query_graph, query_kag, query_lightrag, query_rag
from query.llm_service import optimize_query
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/query', methods=['POST'])
def handle_query():
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
        
        # Optimize query if requested
        if optimize:
            optimized_query = optimize_query(query_text)
            response = {
                'original_query': query_text,
                'optimized_query': optimized_query,
                'optimization_applied': True
            }
        else:
            optimized_query = query_text
            response = {
                'original_query': query_text,
                'optimization_applied': False
            }
        
        # Call appropriate query function based on mode
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
        
        response.update({
            'status': 'success',
            'data': query_response
        })
        
        return jsonify(response)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing query: {error_trace}")
        return jsonify({
            'error': str(e),
            'status': 'error',
            'traceback': error_trace
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify server status"""
    return jsonify({
        'status': 'healthy',
        'message': 'Server is running'
    })

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the data service cache"""
    try:
        data_service.clear_cache()
        return jsonify({
            'status': 'success',
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 Initializing server...")
    #initialize_data_service()
    print("🚀 Server is running on http://127.0.0.1:5000/")
    print("Available endpoints:")
    print("- POST /query - Query the RAG system")
    print("- GET /health - Health check endpoint")
    print("- POST /clear_cache - Clear data service cache")
    serve(app, host="0.0.0.0", port=5000, threads=10)
