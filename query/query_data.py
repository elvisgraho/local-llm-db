import argparse
import networkx as nx
from langchain.prompts import ChatPromptTemplate
from sklearn.metrics.pairwise import cosine_similarity
from database_paths import CHROMA_PATH
from query.data_service import data_service
from query.templates import (
    RAG_ONLY_TEMPLATE,
    KAG_TEMPLATE,
    HYBRID_TEMPLATE,
    DIRECT_TEMPLATE,
    LIGHTRAG_HYBRID_TEMPLATE,
    KAG_HYBRID_TEMPLATE
)
from query.llm_service import get_llm_response, optimize_query
from query.global_vars import (
    RAG_SIMILARITY_THRESHOLD,
    RAG_MAX_DOCUMENTS,
    GRAPH_MAX_DEPTH,
    GRAPH_MAX_NODES,
    GRAPH_MIN_SIMILARITY
)
import os
import logging
from typing import Dict, List, Optional, Union, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--mode", type=str, choices=['rag', 'direct', 'hybrid', 'graph', 'lightrag', 'kag'], 
                       default='rag', help="Query mode: rag, direct, hybrid, graph, lightrag, or kag")
    parser.add_argument("--optimize", action="store_true", help="Whether to optimize the query before processing")
    args = parser.parse_args()
    
    try:
        # Optimize query if requested
        query_text = optimize_query(args.query_text) if args.optimize else args.query_text
        logger.info(f"Processing query in {args.mode} mode: {query_text}")
        
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
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")

def query_direct(query_text: str) -> Dict[str, Union[str, List[str]]]:
    """Query the model directly without using RAG.
    
    Args:
        query_text (str): The query text.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing direct query")
        prompt_template = ChatPromptTemplate.from_template(DIRECT_TEMPLATE)
        prompt = prompt_template.format(question=query_text)
        
        response_text = get_llm_response(prompt)
        return {"text": response_text, "sources": []}
    except Exception as e:
        logger.error(f"Error in direct query: {str(e)}", exc_info=True)
        return {"text": "Error processing direct query", "sources": []}

def query_hybrid(query_text: str) -> Dict[str, Union[str, List[str]]]:
    """Query using both RAG context and the model's knowledge.
    
    Args:
        query_text (str): The query text.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing hybrid query")
        # Initialize only Chroma DB and embedding function
        _ = data_service.embedding_function
        _ = data_service.chroma_db
        
        # Get relevant documents from Chroma with scores
        results = data_service.chroma_db.similarity_search_with_score(query_text, k=RAG_MAX_DOCUMENTS)
        
        if not results:
            logger.warning("No relevant information found in the database")
            return {"text": "No relevant information found in the database", "sources": []}
            
        # Filter results by similarity score
        filtered_results = [(doc, score) for doc, score in results if score >= RAG_SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            logger.warning("No sufficiently relevant information found in the database")
            return {"text": "No sufficiently relevant information found in the database", "sources": []}
        
        # Format context and create prompt
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        prompt_template = ChatPromptTemplate.from_template(HYBRID_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        # Get response
        response_text = get_llm_response(prompt)
        sources = [doc.metadata.get("id", None) for doc, _score in filtered_results]
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in hybrid query: {str(e)}", exc_info=True)
        return {"text": "Error processing hybrid query", "sources": []}

def query_graph(query_text: str, hybrid: bool = False) -> Dict[str, Union[str, List[str]]]:
    """Query using the graph structure with semantic search.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing graph query")
        # Initialize only GraphRAG graph and embedding function
        _ = data_service.embedding_function
        G = data_service.graphrag_graph
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}", exc_info=True)
        return {"text": "Error loading graph structure", "sources": []}

    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    # Get query embedding
    query_embedding = data_service.embedding_function.embed_query(query_text)
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > GRAPH_MIN_SIMILARITY:
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    relevant_nodes = [node for node, _ in relevant_nodes[:GRAPH_MAX_NODES]]

    if not relevant_nodes:
        logger.warning("No relevant information found in the graph structure")
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
    
    response_text = get_llm_response(prompt)
    return {"text": response_text, "sources": list(sources)}

def query_rag(query_text: str, hybrid: bool = False) -> Dict[str, Union[str, List[str]]]:
    """Query using only RAG context.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing RAG query")
        # Initialize only Chroma DB and embedding function
        _ = data_service.embedding_function
        db = data_service.chroma_db
        
        # Verify database exists and has data
        if not os.path.exists(os.path.join(str(CHROMA_PATH), "chroma.sqlite3")):
            logger.error("RAG database not found")
            return {"text": "RAG database not found. Please run populate_rag.py first.", "sources": []}
            
        # Get relevant documents from Chroma with scores
        results = db.similarity_search_with_score(query_text, k=RAG_MAX_DOCUMENTS)
        
        if not results:
            logger.warning("No relevant information found in the database")
            return {"text": "No relevant information found in the database", "sources": []}
            
        # Filter results by similarity score
        filtered_results = [(doc, score) for doc, score in results if score >= RAG_SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            logger.warning("No sufficiently relevant information found in the database")
            return {"text": "No sufficiently relevant information found in the database", "sources": []}

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        
        # Use hybrid template if hybrid mode is enabled
        template = HYBRID_TEMPLATE if hybrid else RAG_ONLY_TEMPLATE
        prompt_template = ChatPromptTemplate.from_template(template)
        prompt = prompt_template.format(context=context_text, question=query_text)

        # Get response
        response_text = get_llm_response(prompt)
        
        # Extract sources from metadata, ensuring they are valid file paths
        sources = []
        for doc, _score in filtered_results:
            source = doc.metadata.get("source")
            if source and isinstance(source, str) and source.strip():
                sources.append(source)
            else:
                logger.warning(f"Invalid or missing source in document metadata: {doc.metadata}")
        
        # Ensure database is persisted after query
        data_service.persist_chroma_db()
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        return {"text": "Error processing RAG query", "sources": []}

def query_lightrag(query_text: str, hybrid: bool = False) -> Dict[str, Union[str, List[str]]]:
    """Query using the light RAG implementation.
    
    LightRAG is a simplified version of RAG that focuses on speed and efficiency:
    1. Uses FAISS for faster similarity search
    2. Simpler document processing
    3. Optional QA chain for faster responses
    4. Less strict filtering and validation
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing light RAG query")
        # Initialize only vectorstore and QA chain
        _ = data_service.vectorstore
        _ = data_service.qa_chain
        
        # Get relevant documents
        results = data_service.vectorstore.similarity_search_with_score(query_text, k=RAG_MAX_DOCUMENTS)
        
        if not results:
            logger.warning("No relevant information found in the database")
            return {"text": "No relevant information found in the database", "sources": []}
        
        # Filter results by similarity score (optional for LightRAG)
        filtered_results = [(doc, score) for doc, score in results if score >= RAG_SIMILARITY_THRESHOLD]
        
        if not filtered_results:
            logger.warning("No sufficiently relevant information found in the database")
            return {"text": "No sufficiently relevant information found in the database", "sources": []}
        
        # Format context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
        
        # Extract sources with basic validation
        sources = []
        for doc, _score in filtered_results:
            source = doc.metadata.get("source")
            if source and isinstance(source, str):
                sources.append(source)
            else:
                logger.debug(f"Missing or invalid source in document metadata: {doc.metadata}")
        
        # Use hybrid template if enabled
        if hybrid:
            prompt_template = ChatPromptTemplate.from_template(LIGHTRAG_HYBRID_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            response_text = get_llm_response(prompt)
        else:
            # Use regular QA chain for non-hybrid mode (faster)
            response_text = data_service.qa_chain.invoke({"query": query_text})["result"]
        
        return {"text": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error in light RAG query: {str(e)}", exc_info=True)
        return {"text": "Error processing light RAG query", "sources": []}

def query_kag(query_text: str, hybrid: bool = False) -> Dict[str, Union[str, List[str]]]:
    """Query using Knowledge-Augmented Generation (KAG) approach.
    
    Args:
        query_text (str): The query text.
        hybrid (bool): Whether to use hybrid mode.
        
    Returns:
        Dict[str, Union[str, List[str]]]: The response containing text and sources.
    """
    try:
        logger.debug("Processing KAG query")
        # Initialize only KAG graph and embedding function
        _ = data_service.embedding_function
        G = data_service.kag_graph
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}", exc_info=True)
        return {"text": "Error loading knowledge graph", "sources": []}

    # Get query embedding
    query_embedding = data_service.embedding_function.embed_query(query_text)
    
    # Find relevant nodes based on semantic similarity
    relevant_nodes = []
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    
    for node in chunk_nodes:
        node_data = G.nodes[node]
        if 'embedding' in node_data:
            similarity = cosine_similarity([query_embedding], [node_data['embedding']])[0][0]
            if similarity > GRAPH_MIN_SIMILARITY:
                relevant_nodes.append((node, similarity))

    # Sort nodes by similarity score
    relevant_nodes.sort(key=lambda x: x[1], reverse=True)
    initial_nodes = [node for node, _ in relevant_nodes[:GRAPH_MAX_NODES]]

    if not initial_nodes:
        logger.warning("No relevant information found in the knowledge graph")
        return {"text": "No relevant information found in the knowledge graph", "sources": []}

    # Collect context and relationships through graph traversal
    context_parts = []
    relationships = []
    sources = set()
    visited_nodes = set()
    
    def traverse_graph(node, depth=0):
        """Recursively traverse the graph to collect related information."""
        if depth > GRAPH_MAX_DEPTH or node in visited_nodes:
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
            traverse_graph(neighbor, depth + 1)

    # Start traversal from initial nodes
    for node in initial_nodes:
        traverse_graph(node)

    # Sort context parts by relevance
    context_parts.sort(key=lambda x: cosine_similarity([query_embedding], [data_service.embedding_function.embed_query(x)])[0][0], reverse=True)
    
    # Format the final context
    context_text = "\n\n---\n\n".join(context_parts[:GRAPH_MAX_NODES])
    relationships_text = "\n\n".join(relationships)
    
    # Use hybrid template if enabled
    template = KAG_HYBRID_TEMPLATE if hybrid else KAG_TEMPLATE
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt = prompt_template.format(
        context=context_text,
        relationships=relationships_text,
        question=query_text
    )
    
    response_text = get_llm_response(prompt)
    return {"text": response_text, "sources": list(sources)}

if __name__ == "__main__":
    main()
