from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
from query_data import query_rag, query_direct, query_hybrid, query_graph, optimize_query, query_lightrag
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
        
        if not query_text:
            return jsonify({
                'error': 'Query text is required',
                'status': 'error'
            }), 400
        
        if query_mode not in ['rag', 'direct', 'hybrid', 'graph', 'lightrag']:
            return jsonify({
                'error': 'Invalid query mode. Must be one of: rag, direct, hybrid, graph, lightrag',
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
        elif query_mode == 'hybrid':
            query_response = query_hybrid(optimized_query)
        elif query_mode == 'graph':
            query_response = query_graph(optimized_query)
        elif query_mode == 'lightrag':
            query_response = query_lightrag(optimized_query)
        else:
            query_response = query_rag(optimized_query)
        
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

if __name__ == '__main__':
    print("🚀 Server is running on http://127.0.0.1:5000/")
    print("Available endpoints:")
    print("- POST /query - Query the RAG system")
    print("- GET /health - Health check endpoint")
    serve(app, host="0.0.0.0", port=5000, threads=10)
