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
from training.config import EMBEDDING_CONTEXT_LENGTH # Import the constant
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

# --- Document Processing Function removed, logic moved to main ---

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

        # 1. Load and process documents to get all chunks
        documents = load_documents()
        if not documents:
            logger.error("No documents found to process")
            return

        total_docs = len(documents)
        all_chunks = []
        processed_docs_count = 0
        failed_docs_count = 0

        logger.info(f"Found {total_docs} documents. Processing and splitting into chunks...")
        with tqdm(total=total_docs, desc=f"Splitting documents for '{rag_type}/{db_name}'", unit="doc") as pbar:
            for doc in documents:
                try:
                    if not validate_metadata(doc.metadata):
                         logger.warning(f"Invalid metadata for document: {doc.metadata.get('source', 'unknown')}, skipping.")
                         failed_docs_count += 1
                         continue

                    chunks = split_document(doc, add_tags_llm=args.add_tags)
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_docs_count += 1
                    else:
                        logger.warning(f"Document {doc.metadata.get('source', 'unknown')} resulted in no chunks.")
                        failed_docs_count += 1 # Count as failed if no chunks produced

                except Exception as e:
                    logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    failed_docs_count += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_docs_count, "failed": failed_docs_count})

        if not all_chunks:
            logger.error("No valid chunks were created from any documents. Aborting graph population.")
            return

        logger.info(f"Successfully split {processed_docs_count} documents into {len(all_chunks)} chunks.")
        if failed_docs_count > 0:
            logger.warning(f"{failed_docs_count} documents failed during processing/splitting.")

        # 2. Generate embeddings for all chunks in batches
        logger.info("Generating embeddings for all chunks...")
        try:
            embedding_function = get_embedding_function()
            chunk_texts = [chunk.page_content for chunk in all_chunks]
            all_embeddings = []
            batch_size = EMBEDDING_CONTEXT_LENGTH

            with tqdm(total=len(chunk_texts), desc=f"Generating embeddings for '{rag_type}/{db_name}'", unit="chunk") as pbar_embed:
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    try:
                        batch_embeddings = embedding_function.embed_documents(batch_texts)
                        all_embeddings.extend(batch_embeddings)
                        pbar_embed.update(len(batch_texts))
                    except Exception as embed_error:
                        logger.error(f"Error embedding batch starting at index {i}: {embed_error}")
                        # Handle error: either stop or add placeholders/skip batch
                        # For now, we'll stop if a batch fails critically
                        raise RuntimeError(f"Failed to embed batch starting at index {i}") from embed_error

            if len(all_embeddings) != len(all_chunks):
                 raise RuntimeError(f"Mismatch between number of chunks ({len(all_chunks)}) and generated embeddings ({len(all_embeddings)}).")

            logger.info("Embeddings generated successfully.")

        except Exception as e:
            logger.error(f"Fatal error during embedding generation: {str(e)}")
            sys.exit(1) # Exit if embeddings fail

        # 3. Populate the graph using chunks and embeddings
        logger.info("Populating the knowledge graph...")
        graph = load_graph(graph_path) # Load or create graph here
        nodes_added = 0
        edges_added = 0

        with tqdm(total=len(all_chunks), desc=f"Populating graph for '{rag_type}/{db_name}'", unit="chunk") as pbar_graph:
            for chunk, embedding in zip(all_chunks, all_embeddings):
                try:
                    source = chunk.metadata.get("source")
                    chunk_index = chunk.metadata.get("chunk_index", 0)
                    chunk_id = f"{source}:{chunk_index}"

                    # Add chunk node if it doesn't exist
                    if not graph.has_node(chunk_id):
                        graph.add_node(chunk_id,
                                       content=chunk.page_content,
                                       metadata=chunk.metadata,
                                       embedding=embedding,
                                       type="chunk")
                        nodes_added += 1

                        # Add file node and edge if they don't exist
                        if source and not graph.has_node(source):
                            graph.add_node(source, type="file")
                            nodes_added += 1
                        if source and not graph.has_edge(source, chunk_id):
                             graph.add_edge(source, chunk_id, relation="contains")
                             edges_added += 1
                    else:
                        # Optionally update existing node data if needed
                        # graph.nodes[chunk_id]['embedding'] = embedding # Example update
                        pass # For now, skip if node exists

                except Exception as graph_error:
                    logger.error(f"Error adding chunk {chunk_id} to graph: {graph_error}")
                finally:
                    pbar_graph.update(1)

        logger.info(f"Graph population complete. Added {nodes_added} nodes and {edges_added} edges.")

        # 4. Save the final graph
        save_graph(graph, graph_path, db_dir)

        # 5. Log final statistics
        logger.info(f"KAG graph population completed for '{rag_type}/{db_name}':")
        logger.info(f"- Total documents found: {total_docs}")
        logger.info(f"- Documents processed into chunks: {processed_docs_count}")
        logger.info(f"- Documents failed processing/splitting: {failed_docs_count}")
        logger.info(f"- Total chunks generated: {len(all_chunks)}")
        logger.info(f"- Total embeddings generated: {len(all_embeddings)}")
        logger.info(f"- Final nodes in graph: {len(graph.nodes)}")
        logger.info(f"- Final edges in graph: {len(graph.edges)}")

        if len(all_chunks) == 0:
            logger.error("No chunks were successfully processed into the graph")
        else:
            logger.info(f"Successfully populated KAG graph database '{rag_type}/{db_name}'")

    except Exception as e:
        logger.error(f"Error populating KAG graph database '{rag_type}/{db_name}': {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 