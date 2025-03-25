import argparse
import requests
import json
import networkx as nx
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from training.populate_lightrag import LMStudioLLM, load_vectorstore, create_qa_chain
from backend.database_paths import (
    CHROMA_PATH, 
    KAG_GRAPH_PATH,
    GRAPHRAG_GRAPH_PATH,
    VECTORSTORE_PATH
)
from get_embedding_function import get_embedding_function
from backend.templates import (
    RAG_ONLY_TEMPLATE,
    KAG_TEMPLATE,
    HYBRID_TEMPLATE,
    DIRECT_TEMPLATE,
    QUERY_OPTIMIZATION_TEMPLATE,
    LIGHTRAG_HYBRID_TEMPLATE,
    KAG_HYBRID_TEMPLATE
)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--mode", type=str, choices=['rag', 'direct', 'hybrid', 'graph', 'lightrag', 'kag'], 
                       default='rag', help="Query mode: rag, direct, hybrid, graph, lightrag, or kag")
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
    elif args.mode == 'kag':
        result = query_kag(query_text)
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

def query_rag(query_text: str, hybrid: bool = False):
    """Query using only RAG context."""
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Use hybrid template if hybrid mode is enabled
    template = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Get response
    response_text = _get_llm_response(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    return {"text": response_text, "sources": sources}

def query_graph(query_text: str, hybrid: bool = False):
    """Query using the graph structure with semantic search."""
    # Load the graph
    try:
        with open(GRAPHRAG_GRAPH_PATH, 'r') as f:
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
    
    # Use hybrid template if hybrid mode is enabled
    template = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    response_text = _get_llm_response(prompt)
    return {"text": response_text, "sources": list(sources)}

def query_lightrag(query_text: str, hybrid: bool = False):
    """Query using the light RAG implementation."""
    try:
        # Load vectorstore and create QA chain using helper functions
        vectorstore = load_vectorstore()
        
        # Get relevant documents
        results = vectorstore.similarity_search_with_score(query_text, k=3)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        sources = [doc.metadata.get("source", "unknown") for doc, _score in results]
        
        # Use hybrid template if enabled
        if hybrid:
            prompt_template = ChatPromptTemplate.from_template(LIGHTRAG_HYBRID_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response_text = _get_llm_response(prompt)
        else:
            # Use regular QA chain for non-hybrid mode
            qa_chain = create_qa_chain(vectorstore)
            response_text = qa_chain.invoke({"query": query_text})["result"]
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        print(f"Error in light RAG query: {e}")
        return {"text": "Error processing light RAG query", "sources": []}

def query_kag(query_text: str, hybrid: bool = False):
    """Query using Knowledge-Augmented Generation (KAG) approach."""
    try:
        # Load the graph
        with open(KAG_GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
        
        G = nx.DiGraph()
        for node in graph_data['nodes']:
            G.add_node(node['id'], **node['data'])
        for edge in graph_data['edges']:
            G.add_edge(edge['source'], edge['target'], **edge['data'])
    except Exception as e:
        print(f"Error loading graph: {e}")
        return {"text": "Error loading knowledge graph", "sources": []}

    # Get query embedding
    embedding_function = get_embedding_function()
    query_embedding = embedding_function.embed_query(query_text)
    
    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > 0.5:  # Lower threshold to get more initial nodes
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    initial_nodes = [node for node, _ in relevant_nodes[:3]]  # Start with top 3 most relevant nodes

    if not initial_nodes:
        return {"text": "No relevant information found in the knowledge graph", "sources": []}

    # Collect context and relationships through graph traversal
    context_parts = []
    relationships = []
    sources = set()
    visited_nodes = set()
    
    def traverse_graph(node, depth=0, max_depth=2):
        """Recursively traverse the graph to collect related information."""
        if depth > max_depth or node in visited_nodes:
            return
        
        visited_nodes.add(node)
        node_data = G.nodes[node]
        
        # Add node content
        context_parts.append(node_data.get('content', ''))
        sources.add(node_data.get('metadata', {}).get('source', 'unknown'))
        
        # Process edges
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            
            # Add relationship information
            if edge_data.get('relation'):
                neighbor_data = G.nodes[neighbor]
                relation_type = edge_data['relation']
                similarity = edge_data.get('similarity', 0)
                
                # Format relationship based on type
                if relation_type == 'semantically_similar':
                    relationships.append(
                        f"- Similar content (similarity: {similarity:.2f}):\n"
                        f"  From: {node_data.get('content', '')[:100]}...\n"
                        f"  To: {neighbor_data.get('content', '')[:100]}..."
                    )
                elif relation_type == 'same_section':
                    relationships.append(
                        f"- Same section ({node_data.get('metadata', {}).get('section_type', 'unknown')}):\n"
                        f"  From: {node_data.get('content', '')[:100]}...\n"
                        f"  To: {neighbor_data.get('content', '')[:100]}..."
                    )
            
            # Recursively traverse
            traverse_graph(neighbor, depth + 1, max_depth)

    # Start traversal from initial nodes
    for node in initial_nodes:
        traverse_graph(node)

    # Sort context parts by relevance
    context_parts.sort(key=lambda x: cosine_similarity([query_embedding], [embedding_function.embed_query(x)])[0][0], reverse=True)
    
    # Format the final context
    context_text = "\n\n---\n\n".join(context_parts[:5])  # Take top 5 most relevant parts
    relationships_text = "\n\n".join(relationships)
    
    # Use hybrid template if enabled
    template = KAG_HYBRID_TEMPLATE if hybrid else KAG_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(
        context=context_text,
        relationships=relationships_text,
        question=query_text
    )
    
    response_text = _get_llm_response(prompt)
    return {"text": response_text, "sources": list(sources)}

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
