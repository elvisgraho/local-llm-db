"""
GraphRAG (Graph-based Retrieval Augmented Generation) Implementation

This module implements a graph-based approach to RAG that creates a directed graph structure
from document chunks. The graph captures:
1. Hierarchical relationships between documents and their chunks
2. Section types (scenario, mitigation, impact, etc.)
3. Code blocks and payloads
4. Semantic relationships between chunks based on embeddings

The graph structure allows for:
- Complex querying across related chunks
- Understanding document structure and relationships
- Finding similar content across different documents
- Identifying patterns in security documentation
"""

import argparse
import os
import logging
import json
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from sklearn.metrics.pairwise import cosine_similarity
from query.database_paths import GRAPHRAG_GRAPH_PATH, GRAPHRAG_DB_DIR
from extract_metadata_llm import extract_metadata_llm
from load_documents import load_documents, process_single_file, extract_metadata
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GraphRAGConfig:
    """Configuration for GraphRAG processing."""
    chunk_size: int = 1500
    chunk_overlap: int = 200
    similarity_threshold: float = 0.7
    batch_size: int = 100
    max_workers: int = 4
    max_depth: int = 2
    save_interval: int = 10  # Save graph after processing N documents

def validate_graph(graph: nx.DiGraph) -> Tuple[bool, List[str]]:
    """Validate graph structure and data integrity.
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, list of validation errors)
    """
    errors = []
    
    # Check for isolated nodes
    isolated_nodes = list(nx.isolates(graph))
    if isolated_nodes:
        errors.append(f"Found {len(isolated_nodes)} isolated nodes")
    
    # Check for cycles
    try:
        nx.find_cycle(graph)
        errors.append("Graph contains cycles")
    except nx.NetworkXNoCycle:
        pass
    
    # Validate node data
    for node, data in graph.nodes(data=True):
        if 'type' not in data:
            errors.append(f"Node {node} missing type attribute")
        if data.get('type') == 'chunk' and 'embedding' not in data:
            errors.append(f"Chunk node {node} missing embedding")
    
    return len(errors) == 0, errors

def optimize_similarity_search(
    graph: nx.DiGraph,
    query_embedding: List[float],
    k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Tuple[str, float]]:
    """Optimized similarity search using approximate nearest neighbors.
    
    Args:
        graph: The graph to search in
        query_embedding: The query embedding vector
        k: Number of results to return
        similarity_threshold: Minimum similarity score threshold
        
    Returns:
        List of (node_id, similarity_score) tuples
    """
    chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chunk']
    if not chunk_nodes:
        return []
    
    # Get embeddings for all chunk nodes
    embeddings = np.array([graph.nodes[n]['embedding'] for n in chunk_nodes])
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    
    # Get top k results above threshold
    results = [(node, score) for node, score in zip(chunk_nodes, similarities) 
               if score >= similarity_threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:k]

def process_document_batch(
    docs: List[Document],
    config: GraphRAGConfig,
    graph: Optional[nx.DiGraph] = None
) -> nx.DiGraph:
    """Process a batch of documents and update the graph.
    
    Args:
        docs: List of documents to process
        config: GraphRAG configuration
        graph: Optional existing graph to update
        
    Returns:
        Updated graph
    """
    if graph is None:
        graph = load_graph()
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = []
        for doc in docs:
            future = executor.submit(process_document, doc, config, graph)
            futures.append(future)
        
        # Wait for all documents to be processed
        for future in tqdm(futures, desc="Processing documents"):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
    
    return graph

def process_document(
    doc: Document,
    config: GraphRAGConfig,
    graph: Optional[nx.DiGraph] = None
) -> None:
    """Process a single document and update the graph."""
    if graph is None:
        graph = load_graph()
    
    # Split document into chunks
    chunks = split_document(doc)
    if not chunks:
        logger.warning(f"No valid chunks created for document: {doc.metadata.get('source', 'unknown')}")
        return
    
    # Get embeddings for new chunks
    embedding_function = get_embedding_function()
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embedding_function.embed_documents(chunk_texts)
    
    # Process chunks in batches
    for i in range(0, len(chunks), config.batch_size):
        batch_chunks = chunks[i:i + config.batch_size]
        batch_embeddings = chunk_embeddings[i:i + config.batch_size]
        
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            chunk_id = f"{chunk.metadata.get('source', 'unknown')}:{chunk.metadata.get('page', 0)}"
            
            # Skip if node already exists
            if graph.has_node(chunk_id):
                logger.info(f"Skipping existing chunk: {chunk_id}")
                continue
            
            # Get LLM metadata
            llm_metadata = chunk.metadata.get("llm_metadata", {})
            
            # Add chunk node with metadata and embedding
            graph.add_node(chunk_id, 
                          content=chunk.page_content,
                          metadata={
                              "source": chunk.metadata.get("source", "unknown"),
                              "page": chunk.metadata.get("page", 0),
                              "content_type": llm_metadata.get("content_type", "unknown"),
                              "main_topic": llm_metadata.get("main_topic", "unknown"),
                              "key_concepts": llm_metadata.get("key_concepts", ""),
                              "has_code": llm_metadata.get("has_code", False),
                              "has_instructions": llm_metadata.get("has_instructions", False),
                              "is_tutorial": llm_metadata.get("is_tutorial", False),
                              "section_type": llm_metadata.get("section_type", "unknown")
                          },
                          embedding=embedding,
                          type="chunk")
            
            # Add relationships
            add_relationships(graph, chunk_id, chunk, llm_metadata)
    
    # Add semantic relationships between chunks
    add_semantic_relationships(graph, config.similarity_threshold)
    
    return graph

def add_relationships(
    graph: nx.DiGraph,
    chunk_id: str,
    chunk: Document,
    llm_metadata: Dict[str, Any]
) -> None:
    """Add all relationships for a chunk node."""
    # Add file node and connect to chunk
    file_path = chunk.metadata.get('source', 'unknown')
    if not graph.has_node(file_path):
        graph.add_node(file_path, type="file")
    graph.add_edge(file_path, chunk_id, relation="contains")
    
    # Add section type nodes and connect
    section_type = llm_metadata.get("section_type")
    if section_type:
        if not graph.has_node(section_type):
            graph.add_node(section_type, type="section")
        graph.add_edge(chunk_id, section_type, relation="belongs_to")
    
    # Add content type nodes and connect
    content_type = llm_metadata.get("content_type")
    if content_type:
        if not graph.has_node(content_type):
            graph.add_node(content_type, type="content_type")
        graph.add_edge(chunk_id, content_type, relation="has_type")
    
    # Add main topic nodes and connect
    main_topic = llm_metadata.get("main_topic")
    if main_topic:
        if not graph.has_node(main_topic):
            graph.add_node(main_topic, type="topic")
        graph.add_edge(chunk_id, main_topic, relation="about")
    
    # Add key concepts nodes and connect
    key_concepts = llm_metadata.get("key_concepts", "").split(",")
    for concept in key_concepts:
        concept = concept.strip()
        if concept:
            if not graph.has_node(concept):
                graph.add_node(concept, type="concept")
            graph.add_edge(chunk_id, concept, relation="relates_to")
    
    # Add code block nodes if present
    if llm_metadata.get("has_code"):
        code_node_id = f"{chunk_id}_code"
        graph.add_node(code_node_id, type="code_block")
        graph.add_edge(chunk_id, code_node_id, relation="contains_code")
    
    # Add instruction nodes if present
    if llm_metadata.get("has_instructions"):
        instruction_node_id = f"{chunk_id}_instructions"
        graph.add_node(instruction_node_id, type="instructions")
        graph.add_edge(chunk_id, instruction_node_id, relation="contains_instructions")

def add_semantic_relationships(
    graph: nx.DiGraph,
    similarity_threshold: float
) -> None:
    """Add semantic relationships between chunks."""
    chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chunk']
    if not chunk_nodes:
        return
    
    chunk_embeddings = [graph.nodes[n]['embedding'] for n in chunk_nodes]
    similarity_matrix = cosine_similarity(chunk_embeddings)
    
    for i, node1 in enumerate(chunk_nodes):
        for j, node2 in enumerate(chunk_nodes[i+1:], i+1):
            if similarity_matrix[i, j] > similarity_threshold:
                graph.add_edge(node1, node2, 
                              relation="semantically_similar",
                              similarity=float(similarity_matrix[i, j]))
                graph.add_edge(node2, node1, 
                              relation="semantically_similar",
                              similarity=float(similarity_matrix[i, j]))

def split_document(doc: Document) -> List[Document]:
    """Split a single document into chunks with improved parameters for security documentation."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n## ",  # Main headers
            "\n\n### ",  # Subheaders
            "\n\n#### ",  # Sub-subheaders
            "\n```",  # Code blocks
            "\n\n",     # Double newlines
            "\n**",
            "\n",       # Single newlines
            " ",        # Spaces
            ""         # No separator
        ],
        keep_separator=True
    )
    
    try:
        # Pre-process security-specific content
        content = doc.page_content
        
        # Normalize bullet points and numbered lists
        content = re.sub(r'^\s*[-•]\s*', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', content, flags=re.MULTILINE)
        
        # Ensure consistent formatting for code blocks
        content = re.sub(r'```(\w+)?\n', r'```\n', content)
        
        # Update document content
        doc.page_content = content
        
        # Split the document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Process each chunk with LLM metadata extraction
        processed_chunks = []
        for chunk in doc_chunks:
            # Extract LLM-based metadata
            llm_metadata = extract_metadata_llm(chunk.page_content)
            chunk.metadata.update(llm_metadata)
            
            # Add file-based metadata
            chunk.metadata.update(extract_metadata(chunk.metadata.get("source", "")))
            
            processed_chunks.append(chunk)
            
        return processed_chunks
            
    except Exception as e:
        logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        return []

def load_graph() -> nx.DiGraph:
    """Load existing graph or create new one."""
    if os.path.exists(GRAPHRAG_GRAPH_PATH):
        logger.info("Loading existing graph")
        with open(GRAPHRAG_GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
            graph = nx.DiGraph()
            for node in graph_data['nodes']:
                graph.add_node(node['id'], **node['data'])
            for edge in graph_data['edges']:
                graph.add_edge(edge['source'], edge['target'], **edge['data'])
    else:
        logger.info("Creating new graph")
        graph = nx.DiGraph()
    return graph

def save_graph(graph: nx.DiGraph):
    """Save the graph structure to disk."""
    os.makedirs(GRAPHRAG_DB_DIR, exist_ok=True)
    
    # Convert graph to JSON-serializable format
    graph_data = {
        "nodes": [{"id": n, "data": d} for n, d in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, "data": d} for u, v, d in graph.edges(data=True)]
    }
    
    # Save to file
    with open(GRAPHRAG_GRAPH_PATH, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    logger.info(f"Saved graph to {GRAPHRAG_GRAPH_PATH}")

def clear_graph():
    """Clear the GraphRAG database."""
    if os.path.exists(GRAPHRAG_DB_DIR):
        for file in os.listdir(GRAPHRAG_DB_DIR):
            file_path = os.path.join(GRAPHRAG_DB_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")
        logger.info(f"Cleared GraphRAG database at {GRAPHRAG_DB_DIR}")

def process_file_to_graph(file_path: Path) -> None:
    """Process a single file and update the graph."""
    try:
        # Load documents from the file
        documents = process_single_file(file_path)
        if not documents:
            logger.warning(f"No valid documents found in file: {file_path.name}")
            return
            
        # Process each document
        for doc in documents:
            try:
                process_document(doc)
            except Exception as e:
                logger.error(f"Error processing document in {file_path.name}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {str(e)}")

def main():
    """Main function to populate the GraphRAG database file by file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    parser.add_argument("--chunk-size", type=int, default=1500, help="Size of document chunks")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Similarity threshold for semantic relationships")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for processing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum depth for graph traversal")
    parser.add_argument("--save-interval", type=int, default=10, help="Save graph after processing N documents")
    args = parser.parse_args()
    
    # Create configuration
    config = GraphRAGConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        max_depth=args.max_depth,
        save_interval=args.save_interval
    )
    
    if args.reset:
        clear_graph()
        logger.info("Cleared existing GraphRAG database")

    try:
        # Get all documents using load_documents functionality
        all_documents = load_documents()
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return
            
        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
        
        # Load or create graph
        graph = load_graph()
        
        # Process documents in batches
        for i in range(0, total_docs, config.batch_size):
            batch_docs = all_documents[i:i + config.batch_size]
            
            try:
                # Process batch
                graph = process_document_batch(batch_docs, config, graph)
                processed_docs += len(batch_docs)
                
                # Validate graph
                is_valid, errors = validate_graph(graph)
                if not is_valid:
                    logger.warning(f"Graph validation errors: {errors}")
                
                # Save graph periodically
                if (i + config.batch_size) % (config.save_interval * config.batch_size) == 0:
                    save_graph(graph)
                    logger.info(f"Saved graph after processing {processed_docs} documents")
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                failed_docs += len(batch_docs)
                continue
        
        # Final save and validation
        save_graph(graph)
        is_valid, errors = validate_graph(graph)
        if not is_valid:
            logger.warning(f"Final graph validation errors: {errors}")
        
        # Log final statistics
        logger.info(f"GraphRAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed documents: {processed_docs}")
        logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No files were successfully processed")
        else:
            logger.info("Successfully populated GraphRAG database")
        
    except Exception as e:
        logger.error(f"Error in database population: {str(e)}")
        raise

if __name__ == "__main__":
    main() 