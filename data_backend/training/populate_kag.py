"""
Knowledge-Augmented Generation (KAG) Implementation

This module implements a knowledge-augmented approach to RAG using a NetworkX graph for storing
document chunks, embeddings, and relationships.

Key features:
1. Document chunking using shared utilities
2. Knowledge graph construction storing chunks and embeddings
3. Hybrid retrieval combining vector similarity and graph traversal
"""

import sys
import logging
import json
import shutil
import argparse
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

# --- Add project root to path ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Internal Imports ---
from query.database_paths import get_db_paths
from training.load_documents import load_documents
from training.processing_utils import split_document, validate_metadata
from training.get_embedding_function import get_embedding_function
from query.global_vars import EMBEDDING_CONTEXT_LENGTH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_graph(graph_path: Path) -> nx.DiGraph:
    """Load existing graph from JSON or create a new one."""
    try:
        if graph_path.exists():
            logger.info(f"Loading existing KAG graph from {graph_path}")
            with open(graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
                
            graph = nx.DiGraph()
            # Restore nodes
            for node in graph_data.get('nodes', []):
                graph.add_node(node.get('id'), **node.get('data', {}))
            # Restore edges
            for edge in graph_data.get('edges', []):
                graph.add_edge(edge.get('source'), edge.get('target'), **edge.get('data', {}))
                
            logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")
            return graph
        else:
            logger.info(f"Creating new KAG graph at {graph_path}")
            return nx.DiGraph()
    except Exception as e:
        logger.error(f"Error loading graph, starting fresh: {e}")
        return nx.DiGraph()

def save_graph(graph: nx.DiGraph, graph_path: Path, db_dir: Path):
    """Save the graph structure to disk."""
    try:
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Helper to handle non-serializable data (like numpy arrays)
        def _serialize_data(data: Dict[str, Any]) -> Dict[str, Any]:
            clean_data = {}
            for k, v in data.items():
                if hasattr(v, 'tolist'): # Check for numpy array
                    clean_data[k] = v.tolist()
                else:
                    clean_data[k] = v
            return clean_data

        graph_data = {
            "nodes": [
                {"id": n, "data": _serialize_data(d)} 
                for n, d in graph.nodes(data=True)
            ],
            "edges": [
                {"source": u, "target": v, "data": _serialize_data(d)} 
                for u, v, d in graph.edges(data=True)
            ]
        }
        
        with open(graph_path, "w", encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2)
            
        logger.info(f"Saved KAG graph to {graph_path}")
    except Exception as e:
        logger.error(f"Error saving KAG graph: {e}", exc_info=True)

def clear_graph_dir(db_dir: Path):
    """Clear the KAG database directory."""
    try:
        if db_dir.exists():
            logger.info(f"Clearing KAG database directory: {db_dir}")
            shutil.rmtree(db_dir)
            logger.info("Directory cleared.")
        else:
            logger.info("Directory does not exist, skipping clear.")
    except Exception as e:
        logger.error(f"Error clearing directory: {e}")

def main():
    parser = argparse.ArgumentParser(description="Populate a KAG (Graph) database instance.")
    parser.add_argument("--db_name", type=str, default="default", help="Name of the DB instance.")
    parser.add_argument("--reset", action="store_true", help="Delete existing graph and start fresh.")
    parser.add_argument("--add-tags", action="store_true", help="Use LLM to generate metadata tags.")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files.")
    
    args = parser.parse_args()
    rag_type = 'kag'
    db_name = args.db_name

    # --- Configuration & Paths ---
    try:
        db_paths = get_db_paths(rag_type, db_name)
    except ValueError as e:
         logger.error(f"Configuration Error: {e}")
         sys.exit(1)

    db_dir = db_paths["db_dir"]
    graph_path = db_paths.get("graph_path")
    
    if not graph_path:
        logger.error(f"Graph path undefined for {rag_type}/{db_name}")
        sys.exit(1)

    logger.info(f"Target: {rag_type.upper()} | DB Name: {db_name}")

    try:
        # 1. Reset if requested
        if args.reset:
            clear_graph_dir(db_dir)
            
        db_dir.mkdir(parents=True, exist_ok=True)

        # 2. Load Documents
        ignore_registry = (not args.resume) or args.reset
        
        documents = load_documents(
            db_dir=db_dir,
            ignore_processed=ignore_registry
        )
        
        if not documents:
            logger.warning("No documents found to process. Exiting.")
            return

        # 3. Process & Chunk Documents
        total_docs = len(documents)
        all_chunks = []
        processed_count = 0
        failed_count = 0

        logger.info(f"Processing {total_docs} documents...")
        
        with tqdm(total=total_docs, desc="Splitting Docs", unit="doc") as pbar:
            for doc in documents:
                try:
                    if not validate_metadata(doc.metadata):
                         failed_count += 1
                         continue

                    chunks = split_document(doc, add_tags_llm=args.add_tags)
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Error processing doc {doc.metadata.get('source')}: {e}")
                    failed_count += 1
                finally:
                    pbar.update(1)

        if not all_chunks:
            logger.error("No valid chunks generated. Aborting.")
            return

        logger.info(f"Generated {len(all_chunks)} chunks from {processed_count} docs.")

        # 4. Generate Embeddings (Batched)
        logger.info("Generating embeddings...")
        embedding_function = get_embedding_function()
        all_embeddings = []
        chunk_texts = [c.page_content for c in all_chunks]
        
        # Batch size for embedding API calls
        batch_size = EMBEDDING_CONTEXT_LENGTH

        with tqdm(total=len(chunk_texts), desc="Embedding", unit="chunk") as pbar:
            for i in range(0, len(chunk_texts), batch_size):
                batch = chunk_texts[i : i + batch_size]
                try:
                    # embed_documents returns list of list of floats
                    embeds = embedding_function.embed_documents(batch)
                    all_embeddings.extend(embeds)
                except Exception as e:
                    logger.critical(f"Embedding failure at batch {i}: {e}")
                    raise
                finally:
                    pbar.update(len(batch))

        if len(all_embeddings) != len(all_chunks):
             raise RuntimeError(f"Embedding mismatch: {len(all_chunks)} chunks vs {len(all_embeddings)} vectors.")

        # 5. Build Graph
        logger.info("Building Knowledge Graph...")
        graph = load_graph(graph_path) # Load existing or create new
        
        # Prepare batch lists for optimized graph update
        new_nodes = []
        new_edges = []
        
        for chunk, embedding in zip(all_chunks, all_embeddings):
            source_file = chunk.metadata.get("source", "unknown")
            chunk_idx = chunk.metadata.get("chunk_index", 0)
            
            # ID Scheme: filename:chunk_index
            # sanitize filename slightly for ID if needed, but path is usually fine
            chunk_id = f"{source_file}:{chunk_idx}"

            # Create Chunk Node
            node_attrs = {
                "content": chunk.page_content,
                "metadata": chunk.metadata,
                "embedding": embedding,
                "type": "chunk"
            }
            new_nodes.append((chunk_id, node_attrs))

            # Create File Node (parent)
            new_nodes.append((source_file, {"type": "file"}))
            
            # Create Edge: File -> Chunk
            new_edges.append((source_file, chunk_id, {"relation": "contains"}))

        # Batch Add to NetworkX (Much faster than adding one by one)
        logger.info(f"Adding {len(new_nodes)} nodes and {len(new_edges)} edges to graph...")
        graph.add_nodes_from(new_nodes)
        graph.add_edges_from(new_edges)

        # 6. Save
        save_graph(graph, graph_path, db_dir)

        # 7. Final Report
        logger.info("="*40)
        logger.info("KAG Population Complete")
        logger.info(f"Total Nodes : {graph.number_of_nodes()}")
        logger.info(f"Total Edges : {graph.number_of_edges()}")
        logger.info(f"Docs Processed: {processed_count}")
        logger.info(f"Failed Docs   : {failed_count}")
        logger.info("="*40)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()