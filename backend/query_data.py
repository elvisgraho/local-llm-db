import argparse
import requests
import json
import networkx as nx
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from training.light_rag import load_vectorstore, create_qa_chain

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"
GRAPH_PATH = "graphrag/graph.json"

# Template that restricts the model to only use the provided context
RAG_ONLY_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Template that allows the model to combine its knowledge with the context
HYBRID_TEMPLATE = """
Here is some relevant context that may help answer the question:

{context}

---

Using both the above context and your general knowledge, please answer this question: {question}

If you use information from the context, please indicate this. If you use your general knowledge, please indicate this as well.
"""

# Template for direct queries without RAG
DIRECT_TEMPLATE = """
Please answer this question using your general knowledge: {question}
"""

# Template for query optimization
QUERY_OPTIMIZATION_TEMPLATE = """
Given the following query, optimize it to be more specific and effective for retrieval. 
Keep the core intent but make it more precise and search-friendly:

Original query: {query}

Optimized query:
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--mode", type=str, choices=['rag', 'direct', 'hybrid', 'graph', 'lightrag'], 
                       default='rag', help="Query mode: rag, direct, hybrid, graph, or lightrag")
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the query before processing")
    args = parser.parse_args()
    
    # Optimize query if requested
    query_text = optimize_query(args.query_text) if args.optimize else args.query_text
    
    if args.mode == 'direct':
        result = query_direct(query_text)
    elif args.mode == 'hybrid':
        result = query_hybrid(query_text)
    elif args.mode == 'graph':
        result = query_graph(query_text)
    elif args.mode == 'lightrag':
        result = query_lightrag(query_text)
    else:
        result = query_rag(query_text)
    
    print(result["text"])
    if result.get("sources"):
        print("\nSources:", result["sources"])

def query_direct(query_text: str):
    """Query the model directly without using RAG."""
    prompt_template = ChatPromptTemplate.from_template(DIRECT_TEMPLATE)
    prompt = prompt_template.format(question=query_text)
    
    response_text = _get_llm_response(prompt)
    return {"text": response_text, "sources": []}

def query_hybrid(query_text: str):
    """Query using both RAG context and the model's knowledge."""
    # Prepare the DB and get relevant documents
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Format context and create prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(HYBRID_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Get response
    response_text = _get_llm_response(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    return {"text": response_text, "sources": sources}

def query_rag(query_text: str):
    """Query using only RAG context."""
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(RAG_ONLY_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Get response
    response_text = _get_llm_response(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {"text": response_text, "sources": sources}

def query_graph(query_text: str):
    """Query using the graph structure with semantic search."""
    # Load the graph
    try:
        with open(GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        
        G = nx.DiGraph()
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['data'])
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['data'])
    except Exception as e:
        print(f"Error loading graph: {e}")
        return {"text": "Error loading graph structure", "sources": []}

    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    # Get query embedding
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query_text)
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > 0.5:  # Adjust threshold as needed
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    relevant_nodes = [node for node, _ in relevant_nodes[:5]]  # Take top 5 most relevant nodes

    if not relevant_nodes:
        return {"text": "No relevant information found in the graph structure", "sources": []}

    # Collect context from relevant nodes and their neighbors
    context_parts = []
    sources = set()
    
    for node in relevant_nodes:
        node_data = G.nodes[node]
        context_parts.append(node_data.get('content', ''))
        sources.add(node_data.get('metadata', {}).get('source', 'unknown'))
        
        # Add content from semantically similar nodes
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            if edge_data.get('relation') == 'semantically_similar':
                neighbor_data = G.nodes[neighbor]
                context_parts.append(neighbor_data.get('content', ''))
                sources.add(neighbor_data.get('metadata', {}).get('source', 'unknown'))
        
        # Add content from code blocks and payloads
        for neighbor in G.neighbors(node):
            neighbor_data = G.nodes[neighbor]
            if neighbor_data.get('type') in ['code_block', 'payload']:
                context_parts.append(neighbor_data.get('content', ''))

    context_text = "\n\n---\n\n".join(context_parts)
    prompt_template = ChatPromptTemplate.from_template(RAG_ONLY_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = _get_llm_response(prompt)
    return {"text": response_text, "sources": list(sources)}

def query_lightrag(query_text: str):
    """Query using the light RAG implementation."""
    try:
        # Load the vectorstore and create QA chain
        vectorstore = load_vectorstore()
        qa_chain = create_qa_chain(vectorstore)
        
        # Get response
        response_text = qa_chain.invoke({"query": query_text})["result"]
        
        # Get sources from the retrieved documents
        results = vectorstore.similarity_search_with_score(query_text, k=3)
        sources = [doc.metadata.get("source", "unknown") for doc, _score in results]
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        print(f"Error in light RAG query: {e}")
        return {"text": "Error processing light RAG query", "sources": []}

def optimize_query(query_text: str) -> str:
    """Optimize the query using a separate LLM call."""
    prompt_template = ChatPromptTemplate.from_template(QUERY_OPTIMIZATION_TEMPLATE)
    prompt = prompt_template.format(query=query_text)
    
    # Use a different model for optimization
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1-distill-qwen-14b",  # You can use a different model for optimization
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3  # Lower temperature for more focused optimization
    }
    response = requests.post(api_url, json=payload, headers=headers)
    response_data = response.json()
    optimized_query = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    
    return optimized_query if optimized_query else query_text

def _get_llm_response(prompt: str) -> str:
    """Helper function to get response from LLM."""
    # ---- LM Studio API Call ----
    api_url = "http://localhost:1234/v1/chat/completions"  # Change port if needed
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1-distill-qwen-14b",  # Replace with your actual model name
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    response = requests.post(api_url, json=payload, headers=headers)
    response_data = response.json()
    return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")

if __name__ == "__main__":
    main()
