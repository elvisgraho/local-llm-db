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
from typing import List, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm
import networkx as nx
import json
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import re
from datetime import datetime
from get_embedding_function import get_embedding_function
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.database_paths import GRAPHRAG_GRAPH_PATH, GRAPHRAG_DB_DIR
from extract_metadata_llm import add_metadata_to_document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_PATH = "data"

def preprocess_text(text: str) -> str:
    """Clean and normalize text before chunking."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract additional metadata from file path and content."""
    path = Path(file_path)
    return {
        "file_type": path.suffix.lower(),
        "file_name": path.name,
        "directory": str(path.parent),
        "created_date": os.path.getctime(file_path),
        "modified_date": os.path.getmtime(file_path)
    }

def validate_document(doc: Document) -> bool:
    """Validate document content and metadata."""
    if not doc.page_content or len(doc.page_content.strip()) < 10:
        logger.warning(f"Document {doc.metadata.get('source', 'unknown')} is too short or empty")
        return False
    return True

def load_documents() -> List[Document]:
    """Load documents from various file types."""
    loaders = {
        "**/*.pdf": PyPDFDirectoryLoader,
        "**/*.txt": TextLoader,
        "**/*.md": UnstructuredMarkdownLoader
    }
    
    all_documents = []
    for glob_pattern, loader_class in loaders.items():
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob=glob_pattern,
                loader_class=loader_class,
                show_progress=True
            )
            documents = loader.load()
            
            for doc in documents:
                doc.page_content = preprocess_text(doc.page_content)
                if validate_document(doc):
                    doc.metadata.update(extract_metadata(doc.metadata.get("source", "")))
                    all_documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {glob_pattern}")
        except Exception as e:
            logger.error(f"Error loading {glob_pattern}: {str(e)}")
    
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with improved parameters for security documentation."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n## ",
            "\n\n### ",
            "\n\n#### ",
            "\n```",
            "\n\n",
            "\n",
            "\n**",
            " ",
            ""
        ],
        keep_separator=True
    )
    
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        try:
            content = doc.page_content
            content = re.sub(r'^\s*[-•]\s*', '• ', content, flags=re.MULTILINE)
            content = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', content, flags=re.MULTILINE)
            content = re.sub(r'```(\w+)?\n', r'```\n', content)
            
            doc.page_content = content
            doc_chunks = text_splitter.split_documents([doc])
            
            for chunk in doc_chunks:
                # Add LLM-extracted metadata to each chunk
                chunk = add_metadata_to_document(chunk)
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
    
    return chunks

def create_graph_structure(chunks: List[Document]) -> nx.DiGraph:
    """Create a directed graph structure from document chunks."""
    # Try to load existing graph
    if os.path.exists(GRAPHRAG_GRAPH_PATH):
        logger.info("Loading existing graph")
        with open(GRAPHRAG_GRAPH_PATH, 'r') as f:
            graph_data = json.load(f)
            G = nx.DiGraph()
            for node in graph_data['nodes']:
                G.add_node(node['id'], **node['data'])
            for edge in graph_data['edges']:
                G.add_edge(edge['source'], edge['target'], **edge['data'])
    else:
        logger.info("Creating new graph")
        G = nx.DiGraph()
    
    # Get embeddings for new chunks
    embedding_function = get_embedding_function()
    chunk_texts = [chunk.page_content for chunk in chunks]
    chunk_embeddings = embedding_function.embed_documents(chunk_texts)
    
    # Add new document nodes with embeddings
    for chunk, embedding in tqdm(zip(chunks, chunk_embeddings), desc="Adding new chunks to graph"):
        chunk_id = f"{chunk.metadata.get('source', 'unknown')}:{chunk.metadata.get('page', 0)}"
        
        # Skip if node already exists
        if G.has_node(chunk_id):
            logger.info(f"Skipping existing chunk: {chunk_id}")
            continue
        
        # Add chunk node with metadata and embedding
        G.add_node(chunk_id, 
                  content=chunk.page_content,
                  metadata=chunk.metadata,
                  embedding=embedding,
                  type="chunk")
        
        # Add file node and connect to chunk
        file_path = chunk.metadata.get('source', 'unknown')
        if not G.has_node(file_path):
            G.add_node(file_path, type="file")
        G.add_edge(file_path, chunk_id, relation="contains")
        
        # Add section type nodes and connect
        section_type = chunk.metadata.get("section_type")
        if section_type:
            if not G.has_node(section_type):
                G.add_node(section_type, type="section")
            G.add_edge(chunk_id, section_type, relation="belongs_to")
        
        # Add code block nodes if present
        if chunk.metadata.get("has_code"):
            code_node_id = f"{chunk_id}_code"
            G.add_node(code_node_id, type="code_block")
            G.add_edge(chunk_id, code_node_id, relation="contains_code")
        
        # Add payload nodes if present
        if chunk.metadata.get("has_payload"):
            payload_node_id = f"{chunk_id}_payload"
            G.add_node(payload_node_id, type="payload")
            G.add_edge(chunk_id, payload_node_id, relation="contains_payload")
    
    # Add semantic relationships between new chunks and existing chunks
    chunk_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'chunk']
    chunk_embeddings = [G.nodes[n]['embedding'] for n in chunk_nodes]
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(chunk_embeddings)
    
    # Add edges for chunks with high semantic similarity
    similarity_threshold = 0.7  # Adjust this threshold as needed
    for i, node1 in enumerate(chunk_nodes):
        for j, node2 in enumerate(chunk_nodes[i+1:], i+1):
            if similarity_matrix[i, j] > similarity_threshold:
                G.add_edge(node1, node2, 
                          relation="semantically_similar",
                          similarity=float(similarity_matrix[i, j]))
                G.add_edge(node2, node1, 
                          relation="semantically_similar",
                          similarity=float(similarity_matrix[i, j]))
    
    return G

def save_graph(G: nx.DiGraph):
    """Save the graph structure to disk."""
    os.makedirs(GRAPHRAG_DB_DIR, exist_ok=True)
    
    # Convert graph to JSON-serializable format
    graph_data = {
        "nodes": [{"id": n, "data": d} for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v, "data": d} for u, v, d in G.edges(data=True)]
    }
    
    # Save to file
    with open(GRAPHRAG_GRAPH_PATH, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    logger.info(f"Graph saved to {GRAPHRAG_GRAPH_PATH}")

def clear_graph():
    """Clear the graph database."""
    if os.path.exists(GRAPHRAG_DB_DIR):
        for file in os.listdir(GRAPHRAG_DB_DIR):
            os.remove(os.path.join(GRAPHRAG_DB_DIR, file))
        logger.info("Graph database cleared")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the graph database.")
    args = parser.parse_args()
    
    if args.reset:
        clear_graph()

    try:
        documents = load_documents()
        if not documents:
            logger.error("No valid documents found to process")
            return
            
        chunks = split_documents(documents)
        if not chunks:
            logger.error("No valid chunks created")
            return
            
        G = create_graph_structure(chunks)
        save_graph(G)
        logger.info("Graph database population completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 