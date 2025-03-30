"""
Graph-based RAG Implementation

This module implements a graph-based approach to RAG using NetworkX for graph structure and FAISS for
vector similarity search. Key features:
1. Document hierarchy and relationship modeling
2. Section type and content categorization
3. Code block and payload tracking
4. Semantic relationship detection

The GraphRAG implementation provides:
- Document structure understanding
- Relationship-based retrieval
- Graph traversal capabilities
- Hybrid query interface
"""

import argparse
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from get_embedding_function import get_embedding_function
from extract_metadata_llm import add_metadata_to_document, format_source_filename
from query.database_paths import VECTORSTORE_PATH, GRAPHRAG_DB_DIR, GRAPHRAG_GRAPH_PATH
from load_documents import load_documents, process_single_file, extract_metadata
import re
import shutil
import json
import networkx as nx
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """Validate document metadata.
    
    Args:
        metadata (Dict[str, Any]): Document metadata
        
    Returns:
        bool: True if metadata is valid, False otherwise
    """
    if not metadata or not isinstance(metadata, dict):
        return False
        
    # Required fields
    required_fields = ["source", "file_name", "file_type"]
    if not all(field in metadata for field in required_fields):
        return False
        
    # Validate source
    if not metadata["source"] or not isinstance(metadata["source"], str):
        return False
        
    # Validate file name
    if not metadata["file_name"] or not isinstance(metadata["file_name"], str):
        return False
        
    # Validate file type
    if not metadata["file_type"] or not isinstance(metadata["file_type"], str):
        return False
        
    return True

def load_graph() -> nx.DiGraph:
    """Load existing graph or create new one.
    
    Returns:
        nx.DiGraph: The loaded or new graph
    """
    try:
        if os.path.exists(GRAPHRAG_GRAPH_PATH):
            logger.info("Loading existing knowledge graph")
            with open(GRAPHRAG_GRAPH_PATH, 'r') as f:
                graph_data = json.load(f)
                graph = nx.DiGraph()
                for node in graph_data['nodes']:
                    graph.add_node(node['id'], **node['data'])
                for edge in graph_data['edges']:
                    graph.add_edge(edge['source'], edge['target'], **edge['data'])
        else:
            logger.info("Creating new knowledge graph")
            graph = nx.DiGraph()
        return graph
    except Exception as e:
        logger.error(f"Error loading graph: {str(e)}")
        return nx.DiGraph()

def save_graph(graph: nx.DiGraph):
    """Save the graph structure to disk.
    
    Args:
        graph (nx.DiGraph): The graph to save
    """
    try:
        os.makedirs(GRAPHRAG_DB_DIR, exist_ok=True)
        
        # Convert graph to JSON-serializable format
        graph_data = {
            "nodes": [{"id": n, "data": d} for n, d in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, "data": d} for u, v, d in graph.edges(data=True)]
        }
        
        # Save to file
        with open(GRAPHRAG_GRAPH_PATH, "w") as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Saved knowledge graph to {GRAPHRAG_GRAPH_PATH}")
    except Exception as e:
        logger.error(f"Error saving graph: {str(e)}")

def split_document(doc: Document, max_chunk_size: int = 1500, max_total_chunks: int = 1000) -> List[Document]:
    """Split a single document into chunks with semantic boundaries.
    
    Args:
        doc (Document): The document to split
        max_chunk_size (int): Maximum size of each chunk
        max_total_chunks (int): Maximum total number of chunks
        
    Returns:
        List[Document]: List of document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n## ",  # Main headers
            "\n\n### ",  # Subheaders
            "\n\n#### ",  # Sub-subheaders
            "\n\n",     # Double newlines
            "\n```",    # Code blocks
            "\n",       # Single newlines
            "\n**",     # Bold text
            " ",        # Spaces
            ""         # No separator
        ],
        keep_separator=True
    )
    
    try:
        # Pre-process content
        content = doc.page_content
        
        # Normalize formatting
        content = re.sub(r'^\s*[-•]\s*', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', content, flags=re.MULTILINE)
        content = re.sub(r'```(\w+)?\n', r'```\n', content)
        
        # Update document content
        doc.page_content = content
        
        # Split the document
        doc_chunks = text_splitter.split_documents([doc])
        
        # Limit total chunks
        if len(doc_chunks) > max_total_chunks:
            logger.warning(f"Document {doc.metadata.get('source', 'unknown')} has too many chunks ({len(doc_chunks)}). Limiting to {max_total_chunks} chunks.")
            doc_chunks = doc_chunks[:max_total_chunks]
        
        # Process chunks with progress bar
        processed_chunks = []
        total_chunks = len(doc_chunks)
        source = doc.metadata.get("source", "unknown")
        # Truncate source filename for display
        display_source = format_source_filename(source)
        
        with tqdm(total=total_chunks, desc=f"Processing {display_source}", unit="chunk", leave=False) as pbar:
            for chunk in doc_chunks:
                try:
                    # Add LLM-based metadata using the helper function
                    chunk = add_metadata_to_document(chunk)
                    
                    # Add file metadata
                    chunk.metadata.update(extract_metadata(chunk.metadata.get("source", "")))
                    
                    # Add chunk-specific metadata
                    chunk.metadata.update({
                        "chunk_index": len(processed_chunks),
                        "total_chunks": len(doc_chunks),
                        "processed_at": datetime.now().isoformat()
                    })
                    
                    processed_chunks.append(chunk)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing chunk from {source}: {str(e)}")
                    continue
            
        return processed_chunks
            
    except Exception as e:
        logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        return []

def process_document(doc: Document, graph: nx.DiGraph, vectorstore: FAISS, reset: bool = False) -> None:
    """Process a single document and update the knowledge graph.
    
    Args:
        doc (Document): The document to process
        graph (nx.DiGraph): The knowledge graph
        vectorstore (FAISS): The FAISS vectorstore
        reset (bool): Whether to reset the vectorstore
    """
    try:
        # Validate metadata
        if not validate_metadata(doc.metadata):
            logger.warning(f"Invalid metadata for document: {doc}")
            return
            
        # Extract source
        source = doc.metadata.get("source")
        if not source or not isinstance(source, str):
            logger.warning(f"Missing or invalid source in document metadata: {doc.metadata}")
            return
            
        # Split document into chunks
        chunks = split_document(doc)
        if not chunks:
            logger.warning(f"No valid chunks created for document: {source}")
            return
            
        # Get embeddings for new chunks
        try:
            embedding_function = get_embedding_function()
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = embedding_function.embed_documents(chunk_texts)
        except Exception as e:
            logger.error(f"Error getting embeddings for document {source}: {str(e)}")
            return
        
        # Add new document nodes with embeddings
        for chunk, embedding in zip(chunks, chunk_embeddings):
            try:
                chunk_id = f"{source}:{chunk.metadata.get('chunk_index', 0)}"
                
                # Skip if node already exists
                if graph.has_node(chunk_id):
                    logger.info(f"Skipping existing chunk: {chunk_id}")
                    continue
                
                # Add chunk node with metadata and embedding
                graph.add_node(chunk_id, 
                              content=chunk.page_content,
                              metadata={
                                  "source": source,
                                  "chunk_index": chunk.metadata.get("chunk_index", 0),
                                  "content_type": chunk.metadata.get("content_type", "unknown"),
                                  "main_topic": chunk.metadata.get("main_topic", "unknown"),
                                  "key_concepts": chunk.metadata.get("key_concepts", ""),
                                  "has_code": chunk.metadata.get("has_code", False),
                                  "has_instructions": chunk.metadata.get("has_instructions", False),
                                  "is_tutorial": chunk.metadata.get("is_tutorial", False),
                                  "section_type": chunk.metadata.get("section_type", "unknown")
                              },
                              embedding=embedding,
                              type="chunk")
                
                # Add file node and connect to chunk
                if not graph.has_node(source):
                    graph.add_node(source, type="file")
                graph.add_edge(source, chunk_id, relation="contains")
                
                # Add section type nodes and connect
                section_type = chunk.metadata.get("section_type")
                if section_type:
                    if not graph.has_node(section_type):
                        graph.add_node(section_type, type="section")
                    graph.add_edge(chunk_id, section_type, relation="belongs_to")
                
                # Add content type nodes and connect
                content_type = chunk.metadata.get("content_type")
                if content_type:
                    if not graph.has_node(content_type):
                        graph.add_node(content_type, type="content_type")
                    graph.add_edge(chunk_id, content_type, relation="has_type")
                
                # Add main topic nodes and connect
                main_topic = chunk.metadata.get("main_topic")
                if main_topic:
                    if not graph.has_node(main_topic):
                        graph.add_node(main_topic, type="topic")
                    graph.add_edge(chunk_id, main_topic, relation="about")
                
                # Add key concepts nodes and connect
                key_concepts = chunk.metadata.get("key_concepts", "").split(",")
                for concept in key_concepts:
                    concept = concept.strip()
                    if concept:
                        if not graph.has_node(concept):
                            graph.add_node(concept, type="concept")
                        graph.add_edge(chunk_id, concept, relation="relates_to")
                
                # Add code block nodes if present
                if chunk.metadata.get("has_code"):
                    code_node_id = f"{chunk_id}_code"
                    graph.add_node(code_node_id, type="code_block")
                    graph.add_edge(chunk_id, code_node_id, relation="contains_code")
                
                # Add instruction nodes if present
                if chunk.metadata.get("has_instructions"):
                    instruction_node_id = f"{chunk_id}_instructions"
                    graph.add_node(instruction_node_id, type="instructions")
                    graph.add_edge(chunk_id, instruction_node_id, relation="contains_instructions")
                
                # Add to vectorstore
                vectorstore.add_documents([chunk])
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
                continue
        
        # Add semantic relationships between new chunks and existing chunks
        try:
            chunk_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'chunk']
            chunk_embeddings = [graph.nodes[n]['embedding'] for n in chunk_nodes]
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(chunk_embeddings)
            
            # Add edges for chunks with high semantic similarity
            similarity_threshold = 0.7  # Adjust this threshold as needed
            for i, node1 in enumerate(chunk_nodes):
                for j, node2 in enumerate(chunk_nodes[i+1:], i+1):
                    if similarity_matrix[i, j] > similarity_threshold:
                        graph.add_edge(node1, node2, 
                                      relation="semantically_similar",
                                      similarity=float(similarity_matrix[i, j]))
                        graph.add_edge(node2, node1, 
                                      relation="semantically_similar",
                                      similarity=float(similarity_matrix[i, j]))
        except Exception as e:
            logger.error(f"Error adding semantic relationships: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)

def clear_graph():
    """Clear the GraphRAG database."""
    try:
        if os.path.exists(GRAPHRAG_DB_DIR):
            for file in os.listdir(GRAPHRAG_DB_DIR):
                file_path = os.path.join(GRAPHRAG_DB_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
            logger.info("Cleared GraphRAG database")
    except Exception as e:
        logger.error(f"Error clearing graph: {str(e)}")

def initialize_vectorstore(reset: bool = False) -> Optional[FAISS]:
    """Initialize or load the FAISS vectorstore.
    
    Args:
        reset (bool): Whether to reset the vectorstore
        
    Returns:
        Optional[FAISS]: The initialized vectorstore or None if initialization failed
    """
    try:
        if reset:
            logger.info("Resetting vectorstore...")
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
            
        # Create initial vectorstore
        vectorstore = FAISS.from_texts(
            ["Initial empty document"],
            embedding_function=get_embedding_function(),
            metadatas=[{
                "source": "initial",
                "file_name": "initial.txt",
                "file_type": "text",
                "processed_at": datetime.now().isoformat()
            }]
        )
        
        # Save initial vectorstore
        vectorstore.save_local(VECTORSTORE_PATH)
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {str(e)}")
        return None

def main():
    """Main function to populate the GraphRAG database."""
    parser = argparse.ArgumentParser(description="Populate GraphRAG database with documents")
    parser.add_argument("--reset", action="store_true", help="Reset the database before processing")
    args = parser.parse_args()
    
    try:
        # Initialize graph and vectorstore
        if args.reset:
            clear_graph()
            logger.info("Cleared existing GraphRAG database")
            
        graph = load_graph()
        vectorstore = initialize_vectorstore(args.reset)
        
        if not vectorstore:
            logger.error("Failed to initialize vectorstore")
            return
            
        # Process documents
        documents = load_documents()
        if not documents:
            logger.error("No documents found to process")
            return
            
        total_docs = len(documents)
        processed_docs = 0
        failed_docs = 0
        
        with tqdm(total=total_docs, desc="Processing documents", unit="doc") as pbar:
            for doc in documents:
                try:
                    process_document(doc, graph, vectorstore, args.reset)
                    processed_docs += 1
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    failed_docs += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": processed_docs,
                        "failed": failed_docs
                    })
                    
        # Save final state
        save_graph(graph)
        vectorstore.save_local(VECTORSTORE_PATH)
        
        # Log statistics
        logger.info(f"GraphRAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed: {processed_docs}")
        logger.info(f"- Failed: {failed_docs}")
        logger.info(f"- Total nodes in graph: {len(graph.nodes)}")
        logger.info(f"- Total edges in graph: {len(graph.edges)}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated GraphRAG database")
            
    except Exception as e:
        logger.error(f"Error populating GraphRAG database: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 