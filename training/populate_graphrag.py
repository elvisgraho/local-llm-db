import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
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
import logging
from pathlib import Path
from tqdm import tqdm
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from get_embedding_function import get_embedding_function
from query.database_paths import get_db_paths
from load_documents import load_documents
from training.processing_utils import validate_metadata, split_document, initialize_vectorstore
import shutil
import json
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- validate_metadata function removed, imported from processing_utils ---

def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load existing graph or create new one.

    Args:
        graph_path (Path): Path to the graph JSON file.

    Returns:
        nx.DiGraph: The loaded or new graph
    """
    try:
        if graph_path.exists():
            logger.info(f"Loading existing knowledge graph from {graph_path}")
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
                graph = nx.DiGraph()
                for node in graph_data['nodes']:
                    graph.add_node(node['id'], **node['data'])
                for edge in graph_data['edges']:
                    graph.add_edge(edge['source'], edge['target'], **edge['data'])
        else:
            logger.info(f"Creating new knowledge graph at {graph_path}")
            graph = nx.DiGraph()
        return graph
    except Exception as e:
        logger.error(f"Error loading graph from {graph_path}: {str(e)}")
        return nx.DiGraph()

def save_graph(graph: nx.DiGraph, graph_path: Path, db_dir: Path):
    """Save the graph structure to disk.

    Args:
        graph (nx.DiGraph): The graph to save.
        graph_path (Path): Path to save the graph JSON file.
        db_dir (Path): Directory to ensure exists.
    """
    try:
        db_dir.mkdir(parents=True, exist_ok=True) # Use the passed db_dir

        # Convert graph to JSON-serializable format
        graph_data = {
            "nodes": [{"id": n, "data": d} for n, d in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, "data": d} for u, v, d in graph.edges(data=True)]
        }
        
        # Save to file
        with open(graph_path, "w") as f: # Use the passed graph_path
            json.dump(graph_data, f, indent=2)

        logger.info(f"Saved knowledge graph to {graph_path}")
    except Exception as e:
        logger.error(f"Error saving graph to {graph_path}: {str(e)}")

# --- split_document function removed, imported from processing_utils ---

def process_document(doc: Document, graph: nx.DiGraph, vectorstore: FAISS, add_tags_llm: bool) -> None:
    """Process a single document, split it, add metadata, and update the knowledge graph and vectorstore.
    
    Args:
        doc (Document): The document to process
        graph (nx.DiGraph): The knowledge graph
        vectorstore (FAISS): The FAISS vectorstore
        add_tags_llm (bool): Whether to use LLM for tag extraction.
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
            
        # Split document into chunks and add metadata
        chunks = split_document(doc, add_tags_llm=add_tags_llm)
        if not chunks:
            logger.warning(f"No valid chunks created or metadata added for document: {source}")
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

def clear_graph(db_dir: Path):
    """Clear the GraphRAG database directory."""
    try:
        if db_dir.exists():
            logger.info(f"Clearing GraphRAG database directory: {db_dir}")
            for item in db_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.error(f"Error removing {item}: {str(e)}")
            logger.info(f"Cleared GraphRAG database directory: {db_dir}")
        else:
            logger.info(f"GraphRAG database directory {db_dir} does not exist, nothing to clear.")
    except Exception as e:
        logger.error(f"Error clearing graph directory {db_dir}: {str(e)}")

# --- initialize_vectorstore function removed, imported from processing_utils ---

def main():
    """Main function to populate the GraphRAG database."""
    parser = argparse.ArgumentParser(description="Populate the GraphRAG database (Graph + FAISS).")
    parser.add_argument("--name", type=str, default="graphrag", help="Name for the database instance (determines directory).")
    parser.add_argument("--reset", action="store_true", help="Reset the database before processing.")
    parser.add_argument("--add-tags", action="store_true", help="Enable LLM-based tag generation if tags are not found in the document content.")
    args = parser.parse_args()

    # --- Get dynamic paths based on name ---
    db_paths = get_db_paths(args.name)
    db_dir = db_paths["db_dir"]
    graph_path = db_paths["graph_path"]
    vectorstore_path = db_paths["vectorstore_path"]
    logger.info(f"Using database name: {args.name}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Graph path: {graph_path}")
    logger.info(f"Vectorstore path: {vectorstore_path}")

    try:
        # Initialize graph and vectorstore using dynamic paths
        if args.reset:
            clear_graph(db_dir) # Pass db_dir
            logger.info(f"Cleared existing GraphRAG database '{args.name}'")

        graph = load_graph(graph_path) # Pass graph_path
        vectorstore = initialize_vectorstore(vectorstore_path, args.reset) # Pass vectorstore_path

        if not vectorstore:
            logger.error(f"Failed to initialize vectorstore for '{args.name}'")
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
                    process_document(doc, graph, vectorstore, add_tags_llm=args.add_tags)
                    processed_docs += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    failed_docs += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": processed_docs,
                        "failed": failed_docs
                    })
                    
        # Save final state using dynamic paths
        save_graph(graph, graph_path, db_dir) # Pass paths
        vectorstore.save_local(str(vectorstore_path)) # FAISS expects string path

        # Log statistics
        logger.info(f"GraphRAG database population completed for '{args.name}':")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed: {processed_docs}")
        logger.info(f"- Failed: {failed_docs}")
        logger.info(f"- Total nodes in graph: {len(graph.nodes)}")
        logger.info(f"- Total edges in graph: {len(graph.edges)}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info(f"Successfully populated GraphRAG database '{args.name}'")

    except Exception as e:
        logger.error(f"Error populating GraphRAG database '{args.name}': {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 