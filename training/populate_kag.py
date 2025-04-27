import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
"""
Knowledge-Augmented Generation (KAG) Implementation

This module implements a knowledge-augmented approach to RAG using a NetworkX graph for storing
document chunks, embeddings, and relationships. Key features:
1. Document chunking using shared utilities
2. Knowledge graph construction storing chunks and embeddings
3. Hybrid retrieval combining vector similarity and graph traversal
4. Integration with LM Studio for local LLM inference

The KAG implementation provides:
- Semantic document understanding
- Relationship-based retrieval
- Knowledge graph traversal
- Hybrid query interface
"""

import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
import json # Added for graph saving
import networkx as nx # Added for graph operations
from langchain.schema.document import Document
from query.database_paths import get_db_paths # Updated import
from load_documents import load_documents # Removed extract_metadata
# Removed initialize_vectorstore, added get_embedding_function
from training.processing_utils import validate_metadata, split_document
from training.get_embedding_function import get_embedding_function # Added
# Removed unused 're' and 'datetime' imports
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- validate_metadata function removed, imported from processing_utils ---

# --- split_document function removed, imported from processing_utils ---

# --- Graph Loading/Saving Functions (adapted from populate_graphrag.py) ---

def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load existing graph or create new one."""
    try:
        if graph_path.exists():
            logger.info(f"Loading existing KAG graph from {graph_path}")
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
                graph = nx.DiGraph()
                # Simplified loading assuming basic node/edge structure for KAG
                for node in graph_data.get('nodes', []):
                    graph.add_node(node.get('id'), **node.get('data', {}))
                for edge in graph_data.get('edges', []):
                    graph.add_edge(edge.get('source'), edge.get('target'), **edge.get('data', {}))
        else:
            logger.info(f"Creating new KAG graph at {graph_path}")
            graph = nx.DiGraph()
        return graph
    except Exception as e:
        logger.error(f"Error loading KAG graph from {graph_path}: {str(e)}")
        return nx.DiGraph()

def save_graph(graph: nx.DiGraph, graph_path: Path, db_dir: Path):
    """Save the graph structure to disk."""
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        graph_data = {
            "nodes": [{"id": n, "data": d} for n, d in graph.nodes(data=True)],
            "edges": [{"source": u, "target": v, "data": d} for u, v, d in graph.edges(data=True)]
        }
        with open(graph_path, "w") as f:
            json.dump(graph_data, f, indent=2)
        logger.info(f"Saved KAG graph to {graph_path}")
    except Exception as e:
        logger.error(f"Error saving KAG graph to {graph_path}: {str(e)}")

# --- Document Processing Function (modified for graph) ---

def process_document_to_graph(doc: Document, graph: nx.DiGraph, add_tags_llm: bool) -> None:
    """
    Process a single document using utilities, split it, add metadata,
    and update the knowledge graph with chunks and embeddings.

    Args:
        doc (Document): The document to process.
        graph (nx.DiGraph): The knowledge graph to update.
        add_tags_llm (bool): Whether to use LLM for tag extraction via split_document.
    """
    # Removed old docstring content and text_splitter definition
    # Removed duplicate function definition below
    # --- Start of function body ---
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
            return # Corrected: Just return None if no chunks
            
        # Get embeddings for new chunks
        try:
            embedding_function = get_embedding_function()
            chunk_texts = [chunk.page_content for chunk in chunks]
            chunk_embeddings = embedding_function.embed_documents(chunk_texts)
        except Exception as e:
            logger.error(f"Error getting embeddings for document {source}: {str(e)}")
            return

        # Add new document nodes with embeddings to the graph
        for chunk, embedding in zip(chunks, chunk_embeddings):
            try:
                # Use a consistent chunk ID format
                chunk_index = chunk.metadata.get("chunk_index", 0) # Assuming split_document adds this
                chunk_id = f"{source}:{chunk_index}"

                # Skip if node already exists (optional, allows updates if needed)
                if graph.has_node(chunk_id):
                    logger.debug(f"Skipping existing chunk node: {chunk_id}")
                    continue

                # Add chunk node with content, metadata, and embedding
                graph.add_node(chunk_id,
                               content=chunk.page_content,
                               metadata=chunk.metadata, # Use metadata from split_document
                               embedding=embedding,
                               type="chunk") # Add type for potential filtering

                # Add file node and connect to chunk (basic relationship)
                if not graph.has_node(source):
                    graph.add_node(source, type="file")
                graph.add_edge(source, chunk_id, relation="contains")

                # KAG specific: Potentially add relationships based on metadata later
                # (e.g., same_section, based_on_topic, etc.)
                # For now, just adding chunks and embeddings as query_kag expects.

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id} for graph: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error processing document {source} for graph: {str(e)}", exc_info=True)


def clear_graph_dir(db_dir: Path):
    """Clear the KAG database directory (including graph file)."""
    try:
        if db_dir.exists():
            logger.info(f"Clearing KAG database directory: {db_dir}")
            shutil.rmtree(db_dir) # Remove the whole directory
            logger.info(f"Cleared KAG database directory: {db_dir}")
        else:
            logger.info(f"KAG database directory {db_dir} does not exist, nothing to clear.")
    except Exception as e:
        logger.error(f"Error clearing KAG directory {db_dir}: {str(e)}")

# --- initialize_vectorstore function removed, imported from processing_utils ---

def main():
    """Main function to populate the KAG Graph database."""
    parser = argparse.ArgumentParser(description="Populate a specific KAG (Graph) database instance.") # Updated description
    # Changed --name to --db_name and set a default
    parser.add_argument("--db_name", type=str, default="default", help="Name for the database instance under the 'kag' directory.")
    parser.add_argument("--reset", action="store_true", help="Reset the graph database before processing.") # Updated help text
    parser.add_argument("--add-tags", action="store_true", help="Enable LLM-based tag generation if tags are not found in the document content.")
    args = parser.parse_args()

    # --- Get dynamic paths based on db_name for 'kag' type ---
    rag_type = 'kag' # Explicitly set rag_type for this script
    db_name = args.db_name
    try:
        db_paths = get_db_paths(rag_type, db_name)
    except ValueError as e:
         logger.error(f"Error getting database paths: {e}")
         sys.exit(1) # Exit if paths are invalid

    db_dir = db_paths["db_dir"] # Main directory for the instance (databases/kag/db_name)
    graph_path = db_paths.get("graph_path") # Use .get() for safety

    if not graph_path:
        logger.error(f"Could not determine Graph path for rag_type='{rag_type}', db_name='{db_name}'")
        sys.exit(1)

    logger.info(f"Target RAG Type: {rag_type}")
    logger.info(f"Target DB Name: {db_name}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Graph path: {graph_path}") # Log graph path

    try:
        # Reset directory if requested
        if args.reset:
            clear_graph_dir(db_dir) # Clear the main instance directory
            logger.info(f"Cleared existing KAG database '{db_name}' at {db_dir}")

        # Load or create graph
        graph = load_graph(graph_path)

        # Process documents and add to graph
        documents = load_documents()
        if not documents:
            logger.error("No documents found to process")
            return
            
        total_docs = len(documents)
        processed_docs = 0
        failed_docs = 0

        logger.info(f"Found {total_docs} documents to process for '{rag_type}/{db_name}'.")
        with tqdm(total=total_docs, desc=f"Processing documents for '{rag_type}/{db_name}'", unit="doc") as pbar:
            for doc in documents:
                try:
                    # Process document and add directly to the graph
                    process_document_to_graph(doc, graph, add_tags_llm=args.add_tags)
                    processed_docs += 1
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')} into graph: {str(e)}")
                    failed_docs += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": processed_docs,
                        "failed": failed_docs
                    })

        # Save the final graph
        save_graph(graph, graph_path, db_dir)

        # Log statistics
        logger.info(f"KAG graph population completed for '{rag_type}/{db_name}':")
        logger.info(f"- Total documents found: {total_docs}")
        logger.info(f"- Documents successfully processed into graph: {processed_docs}")
        logger.info(f"- Failed: {failed_docs}")
        logger.info(f"- Total nodes in graph: {len(graph.nodes)}") # Added graph stats
        logger.info(f"- Total edges in graph: {len(graph.edges)}") # Added graph stats

        if processed_docs == 0:
            logger.error("No documents were successfully processed into the graph")
        else:
            logger.info(f"Successfully populated KAG graph database '{rag_type}/{db_name}'")

    except Exception as e:
        logger.error(f"Error populating KAG graph database '{rag_type}/{db_name}': {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 