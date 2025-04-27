import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
"""
RAG (Retrieval Augmented Generation) Implementation

This module implements a standard RAG approach using Chroma vectorstore for document
retrieval. Key features:
1. Document chunking and embedding
2. Vector similarity search
3. Integration with LM Studio for local LLM inference
4. Basic question-answering capabilities

The standard implementation provides:
- Efficient document retrieval
- Semantic search capabilities
- Simple query interface
- Integration with local LLM
"""

import os
import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain.schema.document import Document
from query.database_paths import get_db_paths # Updated import
from load_documents import load_documents, extract_metadata, process_single_file
from training.processing_utils import split_document, initialize_chroma_vectorstore, clear_db_directory, validate_metadata # Added validate_metadata
import re
import argparse
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- RAGSystem class and RAGResponse dataclass removed, functionality moved to main/utils ---

# --- Local split_document function removed, imported from processing_utils ---

# --- This incorrect local definition of process_document is removed ---

def process_document(doc: Document, add_tags_llm: bool) -> List[Document]:
    """Process a single document using utility functions, split it, add metadata. Returns processed chunks."""
    # Validate metadata (optional but good practice)
    if not validate_metadata(doc.metadata):
         logger.warning(f"Invalid metadata for document: {doc.metadata.get('source', 'unknown')}")
         return []

    source = doc.metadata.get("source") # Already validated

    # Split document into chunks and add metadata using the imported function
    chunks = split_document(doc, add_tags_llm=add_tags_llm)
    if not chunks:
        logger.warning(f"Splitting document {source} resulted in no chunks.")
        return []

    # Ensure source metadata is present in all chunks (can be redundant if split_document handles it)
    for chunk in chunks:
        if not chunk.metadata:
            chunk.metadata = {}
        if "source" not in chunk.metadata or not chunk.metadata["source"]:
             chunk.metadata["source"] = source # Add source if missing

    return chunks # Return the processed chunks for batch adding later

# --- clear_vectorstore function removed, use clear_db_directory from utils ---

def process_file_to_vectorstore(file_path: Path, add_tags_llm: bool) -> None:
    """Clear the RAG database."""
    if os.path.exists(RAG_DB_DIR):
        for root, dirs, files in os.walk(RAG_DB_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        logger.info("Cleared RAG database")

def process_file_to_vectorstore(file_path: Path, add_tags_llm: bool) -> None:
    """Process a single file, extract documents, add metadata, and update the vectorstore."""
    try:
        # Load documents from the file
        documents = process_single_file(file_path)
        if not documents:
            logger.warning(f"No valid documents found in file: {file_path.name}")
            return
            
        # Process each document
        for doc in documents:
            try:
                process_document(doc, add_tags_llm=add_tags_llm)
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata.get('source', file_path.name)}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {str(e)}")

def main():
    """Main function to populate the RAG database file by file."""
    parser = argparse.ArgumentParser(description="Populate the RAG Chroma database.")
    parser.add_argument("--name", type=str, default="rag", help="Name for the database instance (determines directory).")
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating.")
    parser.add_argument("--add-tags", action="store_true", help="Enable LLM-based tag generation if tags are not found in the document content.")
    args = parser.parse_args()

    # --- Get dynamic paths based on name ---
    db_paths = get_db_paths(args.name)
    db_dir = db_paths["db_dir"] # Main directory for the instance
    chroma_path = db_paths["chroma_path"] # Path within the instance dir
    logger.info(f"Using database name: {args.name}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Chroma path: {chroma_path}")

    if args.reset:
        clear_db_directory(chroma_path) # Use utility function with specific chroma path
        logger.info(f"Cleared existing RAG database '{args.name}' at {chroma_path}")

    try:
        # Initialize Chroma vectorstore using dynamic path and imported function
        vectorstore = initialize_chroma_vectorstore(chroma_path, args.reset) # Reset is handled internally now
        if not vectorstore:
             logger.error(f"Failed to initialize Chroma vectorstore for '{args.name}' at {chroma_path}")
             return # Exit if vectorstore fails

        # Get all documents using load_documents functionality
        all_documents = load_documents()

        if not all_documents:
            logger.error("No valid documents found to process")
            return

        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
        all_processed_chunks = []

        # Process documents and collect chunks
        logger.info(f"Found {total_docs} documents to process for '{args.name}'.")
        with tqdm(total=total_docs, desc=f"Processing documents for '{args.name}'", unit="doc") as pbar:
            for doc in all_documents:
                try:
                    processed_chunks = process_document(doc, add_tags_llm=args.add_tags)
                    if processed_chunks:
                        all_processed_chunks.extend(processed_chunks)
                        processed_docs += 1
                    else:
                        # If process_document returned empty list due to error or no chunks
                        failed_docs += 1
                        logger.warning(f"Document {doc.metadata.get('source', 'unknown')} resulted in no processed chunks.")
                except Exception as e:
                    logger.error(f"Error processing document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    failed_docs += 1
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"processed": processed_docs, "failed": failed_docs})

        # Add collected chunks to vectorstore if any exist
        if all_processed_chunks:
            logger.info(f"Adding {len(all_processed_chunks)} processed chunks to the Chroma vectorstore for '{args.name}'...")
            # Consider batching adds for Chroma if performance is an issue for very large datasets
            vectorstore.add_documents(all_processed_chunks)
            logger.info("Chunks added successfully.")
        else:
             logger.warning("No valid chunks were processed to add to the vectorstore.")

        # Persist the vectorstore (Chroma persists automatically when initialized with persist_directory)
        # No explicit persist call needed here for Chroma. Changes are saved.
        logger.info(f"Chroma vectorstore for '{args.name}' at {chroma_path} is up-to-date.")


        # Log final statistics
        logger.info(f"RAG database population completed for '{args.name}':")
        logger.info(f"- Total documents: {total_docs}")
        if failed_docs > 0:
            logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info(f"Successfully populated RAG database '{args.name}'")

    except Exception as e:
        logger.error(f"Error in database population for '{args.name}': {str(e)}")
        raise

if __name__ == "__main__":
    main()
