import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
"""
LightRAG (Lightweight Retrieval Augmented Generation) Implementation

This module implements a lightweight approach to RAG using FAISS vectorstore for efficient
document retrieval. Key features:
1. Simple document chunking and embedding
2. Fast similarity search using FAISS
3. Integration with LM Studio for local LLM inference
4. Basic question-answering capabilities

The lightweight implementation provides:
- Faster document processing
- Lower memory requirements
- Quick similarity search
- Simple query interface

This is a simpler alternative to the graph-based approaches (GraphRAG and KAG) that focuses
on speed and efficiency over complex relationship modeling.
"""

import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain.schema.document import Document
from query.database_paths import get_db_paths # Updated import
from load_documents import load_documents, process_single_file
from training.processing_utils import split_document, initialize_chroma_vectorstore # Changed import
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- split_document function removed, imported from processing_utils ---

def process_document(doc: Document, add_tags_llm: bool) -> List[Document]:
    """Process a single document, split it, add metadata. Returns processed chunks.
    
    LightRAG uses a simplified document processing approach:
    1. Basic metadata validation
    2. Simple chunking without complex validation
    3. Direct FAISS indexing without duplicate checks
    4. No explicit persistence (handled by FAISS)
    
    Args:
        doc (Document): The document to process
        vectorstore (FAISS): The FAISS vectorstore
        reset (bool): Whether to reset the vectorstore
    """
    try:
        # Basic metadata validation
        if not doc.metadata or not isinstance(doc.metadata, dict):
            logger.warning(f"Invalid metadata for document: {doc}")
            return
            
        # Extract source with basic validation
        source = doc.metadata.get("source")
        if not source or not isinstance(source, str):
            logger.debug(f"Missing or invalid source in document metadata: {doc.metadata}")
            return
            
        # Split document into chunks and add metadata using the imported function
        chunks = split_document(doc, add_tags_llm=add_tags_llm) # Calls the imported function now
        if not chunks:
             logger.warning(f"Splitting document {source} resulted in no chunks.")
             return []

        # Ensure source metadata is present in all chunks (this part remains specific to lightrag's processing)
        for chunk in chunks:
            if not chunk.metadata:
                chunk.metadata = {}
            if "source" not in chunk.metadata:
                 chunk.metadata["source"] = source # Add source if missing after splitting/metadata steps

        return chunks # Return the processed chunks
            
    except Exception as e:
        logger.error(f"Error processing document {source}: {str(e)}", exc_info=True)
        return [] # Return empty list on error

# --- clear_vectorstore function removed, rely on initialize_vectorstore's reset logic ---

def process_file_to_vectorstore(file_path: Path) -> None:
    """Process a single file and update the vectorstore."""
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
    """Main function to populate the LightRAG database."""
    parser = argparse.ArgumentParser(description="Populate the LightRAG FAISS database.")
    parser.add_argument("--name", type=str, default="lightrag", help="Name for the database instance (determines directory).")
    parser.add_argument("--reset", action="store_true", help="Reset the vectorstore before processing.")
    parser.add_argument("--add-tags", action="store_true", help="Enable LLM-based tag generation if tags are not found in the document content.")
    args = parser.parse_args()

    # --- Get dynamic paths based on name ---
    db_paths = get_db_paths(args.name)
    db_dir = db_paths["db_dir"] # Main directory for the instance
    chroma_path = db_paths.get("chroma_path", db_paths["vectorstore_path"]) # Use chroma_path if available, fallback for safety
    logger.info(f"Using database name: {args.name}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Chroma path: {chroma_path}") # Updated log

    try:
        # Initialize Chroma vectorstore using dynamic path and imported function
        vectorstore = initialize_chroma_vectorstore(chroma_path, args.reset) # Use Chroma initializer
        if not vectorstore:
             logger.error(f"Failed to initialize Chroma vectorstore for '{args.name}' at {chroma_path}") # Updated log
             return # Exit if vectorstore fails

        # Process documents and collect chunks
        all_processed_chunks = []
        loaded_docs = load_documents()
        total_docs = len(loaded_docs)
        
        logger.info(f"Found {total_docs} documents to process.")
        
        with tqdm(total=total_docs, desc="Processing documents", unit="doc") as pbar:
            for doc in loaded_docs:
                try:
                    processed_chunks = process_document(doc, add_tags_llm=args.add_tags)
                    if processed_chunks:
                        all_processed_chunks.extend(processed_chunks)
                except Exception as e:
                     logger.error(f"Failed processing document {doc.metadata.get('source', 'unknown')}: {e}")
                finally:
                    pbar.update(1)

        # Add collected chunks to vectorstore if any exist
        if all_processed_chunks:
            logger.info(f"Adding {len(all_processed_chunks)} processed chunks to the vectorstore...")
            vectorstore.add_documents(all_processed_chunks)
            logger.info("Chunks added successfully.")
        else:
             logger.warning("No valid chunks were processed to add to the Chroma vectorstore.") # Updated log

        # Persist vectorstore (Chroma persists automatically when initialized with persist_directory)
        logger.info(f"Chroma vectorstore for '{args.name}' at {chroma_path} is up-to-date.") # Updated log
        logger.info(f"LightRAG (using Chroma) database '{args.name}' populated successfully") # Updated log

    except Exception as e:
        logger.error(f"Error populating LightRAG (using Chroma) database '{args.name}': {str(e)}", exc_info=True) # Updated log
        raise

if __name__ == "__main__":
    main() 