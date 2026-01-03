"""
LightRAG (Lightweight Retrieval Augmented Generation) Implementation

This module orchestrates the database population pipeline:
1. Load Documents (Hybrid PDF/Text loader)
2. Extract Metadata (Manual Tags + LLM Fallback)
3. Split & Flatten (Graph entities -> Strings)
4. Index to ChromaDB
"""
import logging
import sys
import argparse
from pathlib import Path
from typing import List
from tqdm import tqdm

# --- Add project root to path ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Modern LangChain Imports ---
from langchain_core.documents import Document

# --- Internal Imports ---
from common import get_db_paths
from training.load_documents import load_documents
from training.processing_utils import (
    manage_db_configuration, 
    split_document, 
    initialize_chroma_vectorstore, 
    validate_metadata
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_document(doc: Document, add_tags_llm: bool, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Process a single document: validate, split, and ensure metadata.
    """
    # 1. Validate Metadata (Basic checks)
    if not validate_metadata(doc.metadata):
        logger.warning(f"Skipping document with invalid metadata: {doc.metadata.get('source', 'unknown')}")
        return []

    # 2. Split Document 
    # This calls the ROBUST split_document from processing_utils.py which handles:
    # - Manual Tag Extraction (stripping "Tags:" header)
    # - LLM Tag Generation (if add_tags_llm=True)
    # - Entity Flattening (List -> String for Chroma)
    try:
        source = doc.metadata.get("source")

        chunks = split_document(
            doc, 
            add_tags_llm=add_tags_llm,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        if not chunks:
            # If chunks is empty, it might be because the LLM flagged it as non-technical/junk
            # Logger warning is handled inside split_document usually, but good to have here too
            return []
        
        # 3. Final Safety Check on Metadata
        for chunk in chunks:
            if "source" not in chunk.metadata:
                 chunk.metadata["source"] = source

        return chunks

    except Exception as e:
        logger.error(f"Error processing document {source}: {e}")
        return []

def main():
    """Main function to populate the LightRAG database (using ChromaDB)."""
    parser = argparse.ArgumentParser(description="Populate a LightRAG (Chroma) database instance.")
    parser.add_argument("--db_name", type=str, default="default", help="Name of the DB instance (e.g., 'default', 'medical').")
    parser.add_argument("--reset", action="store_true", help="Delete existing DB and start fresh.")
    parser.add_argument("--add-tags", action="store_true", help="Use LLM to enrich metadata (Entities/Summaries).")
    parser.add_argument("--resume", action="store_true", help="Skip already processed files.")
    parser.add_argument("--ocr", action="store_true", default=True, help="Enable OCR for extracting text from images in PDFs (default: True).")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chars per chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap chars.")

    args = parser.parse_args()
    
    logger.info(f"Configuration -> Reset: {args.reset} | AI Tagging: {args.add_tags}")

    # --- Configuration ---
    rag_type = 'lightrag'
    db_name = args.db_name

    # --- Path Resolution ---
    try:
        db_paths = get_db_paths(rag_type, db_name)
    except ValueError as e:
         logger.error(f"Configuration Error: {e}")
         sys.exit(1)

    chroma_path = db_paths.get("chroma_path") or db_paths.get("vectorstore_path")
    db_dir = db_paths["db_dir"]
    
    # Check/Create DB Config
    manage_db_configuration(db_paths["db_dir"], "lightrag", args)

    if not chroma_path:
        logger.error(f"Could not determine Chroma path for {rag_type}/{db_name}")
        sys.exit(1)

    logger.info(f"Target: {rag_type.upper()} | DB Name: {db_name}")
    logger.info(f"Path: {chroma_path}")

    try:
        # 1. Initialize Vector Store
        vectorstore = initialize_chroma_vectorstore(chroma_path, reset=args.reset)
        if not vectorstore:
             logger.error("Failed to initialize Chroma vectorstore.")
             sys.exit(1)
             
        # 2. Load Documents
        ignore_registry = (not args.resume) or args.reset
        loaded_docs = load_documents(
            db_dir=db_dir,
            ignore_processed=ignore_registry,
            is_ocr_enabled=args.ocr
        )
        
        total_docs = len(loaded_docs)
        if total_docs == 0:
            logger.warning("No documents found in data directory. Exiting.")
            sys.exit(0)

        logger.info(f"Loaded {total_docs} source documents.")

        # 3. Process & Chunk Documents
        all_processed_chunks = []

        with tqdm(total=total_docs, desc="Processing Docs", unit="doc") as pbar:
            for doc in loaded_docs:
                chunks = process_document(
                    doc, 
                    add_tags_llm=args.add_tags,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap
                )
                if chunks:
                    all_processed_chunks.extend(chunks)

                pbar.update(1)

        # 4. Index Chunks (Batching)
        if all_processed_chunks:
            total_chunks = len(all_processed_chunks)
            logger.info(f"Indexing {total_chunks} chunks into ChromaDB (Batch Size: {args.chunk_size})...")

            # We reuse chunk_size arg as batch size for insertion, or hardcode 100
            BATCH_SIZE = 100 
            
            with tqdm(total=total_chunks, desc="Indexing", unit="chunk") as pbar:
                for i in range(0, total_chunks, BATCH_SIZE):
                    batch = all_processed_chunks[i : i + BATCH_SIZE]
                    try:
                        vectorstore.add_documents(batch)
                        pbar.update(len(batch))
                    except Exception as e:
                        logger.error(f"Batch indexing failed at index {i}. Retrying individually... Error: {e}")
                        
                        # Fallback to individual insertion
                        for doc in batch:
                            try:
                                vectorstore.add_documents([doc])
                            except Exception as inner_e:
                                src = doc.metadata.get('source', 'unknown')
                                logger.error(f"Failed to index specific doc {src}: {inner_e}")
                            finally:
                                pbar.update(1)

            logger.info("Indexing complete.")
            
        else:
             logger.warning("No valid chunks generated. Database remains empty.")

        logger.info(f"LightRAG database '{db_name}' ready at: {chroma_path}")

    except KeyboardInterrupt:
        logger.info("\nOperation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical failure during population: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()