"""
Knowledge Augmented Graph (KAG) Implementation

This module implements a knowledge graph-based approach to RAG that creates a directed graph
structure from document chunks. The graph captures:
1. Document hierarchy and relationships
2. Section types and content categorization
3. Code blocks and payloads
4. Semantic relationships between chunks

The knowledge graph enables:
- Semantic search across related content
- Understanding document structure
- Finding similar content
- Pattern recognition in security documentation

This implementation is similar to GraphRAG but with a focus on knowledge representation
and semantic relationships.
"""

import argparse
import os
import logging
import json
import networkx as nx
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from sklearn.metrics.pairwise import cosine_similarity
from query.database_paths import KAG_GRAPH_PATH, KAG_DB_DIR
from extract_metadata_llm import extract_metadata_llm
from load_documents import load_documents, process_single_file, extract_metadata
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_graph() -> nx.DiGraph:
    """Load existing graph or create new one."""
    if os.path.exists(KAG_GRAPH_PATH):
        logger.info("Loading existing knowledge graph")
        with open(KAG_GRAPH_PATH, 'r') as f:
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

def save_graph(graph: nx.DiGraph):
    """Save the graph structure to disk."""
    os.makedirs(KAG_DB_DIR, exist_ok=True)
    
    # Convert graph to JSON-serializable format
    graph_data = {
        "nodes": [{"id": n, "data": d} for n, d in graph.nodes(data=True)],
        "edges": [{"source": u, "target": v, "data": d} for u, v, d in graph.edges(data=True)]
    }
    
    # Save to file
    with open(KAG_GRAPH_PATH, "w") as f:
        json.dump(graph_data, f, indent=2)
    
    logger.info(f"Saved knowledge graph to {KAG_GRAPH_PATH}")

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

def process_document(doc: Document) -> None:
    """Process a single document and update the knowledge graph."""
    try:
        # Load existing graph
        graph = load_graph()
        
        # Split document into chunks
        chunks = split_document(doc)
        if not chunks:
            logger.warning(f"No valid chunks created for document: {doc.metadata.get('source', 'unknown')}")
            return
        
        # Get embeddings for new chunks
        try:
            embedding_function = get_embedding_function()
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = embedding_function.embed_documents(chunk_texts)
        except Exception as e:
            logger.error(f"Error getting embeddings for document {doc.metadata.get('source', 'unknown')}: {str(e)}")
            return
        
        # Add new document nodes with embeddings
        for chunk, embedding in zip(chunks, chunk_embeddings):
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
        
        # Add semantic relationships between new chunks and existing chunks
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
        
        # Save the updated graph
        save_graph(graph)
        
    except Exception as e:
        logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        raise

def clear_graph():
    """Clear the KAG database."""
    if os.path.exists(KAG_DB_DIR):
        for file in os.listdir(KAG_DB_DIR):
            file_path = os.path.join(KAG_DB_DIR, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")
        logger.info(f"Cleared KAG database at {KAG_DB_DIR}")

def process_file_to_graph(file_path: Path) -> None:
    """Process a single file and update the knowledge graph."""
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
    """Main function to populate the KAG database file by file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()
    
    if args.reset:
        clear_graph()
        logger.info("Cleared existing KAG database")

    try:
        # Get all documents using load_documents functionality
        all_documents = load_documents()
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return
            
        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
            
        # Process documents one by one
        for doc in tqdm(all_documents, desc="Processing documents", total=total_docs):
            try:
                logger.info(f"Processing document {processed_docs + 1}/{total_docs}")
                process_document(doc)
                processed_docs += 1
                    
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                failed_docs += 1
                continue
        
        # Log final statistics
        logger.info(f"KAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed documents: {processed_docs}")
        logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated KAG database")
        
    except Exception as e:
        logger.error(f"Error in database population: {str(e)}")
        raise

if __name__ == "__main__":
    main() 