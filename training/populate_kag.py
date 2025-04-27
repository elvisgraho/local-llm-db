import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
"""
Knowledge-Augmented Generation (KAG) Implementation

This module implements a knowledge-augmented approach to RAG using FAISS vectorstore for efficient
document retrieval and knowledge graph construction. Key features:
1. Advanced document chunking with semantic boundaries
2. Knowledge graph construction from document relationships
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
from langchain.schema.document import Document
from query.database_paths import get_db_paths # Updated import
from load_documents import load_documents, extract_metadata
from training.processing_utils import validate_metadata, split_document, initialize_vectorstore
import re
import argparse
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- validate_metadata function removed, imported from processing_utils ---

# --- split_document function removed, imported from processing_utils ---

def process_document(doc: Document, add_tags_llm: bool) -> List[Document]:
    """
    Split a single document into chunks with semantic boundaries.
    Optionally adds LLM-based metadata if add_tags_llm is True and no tags found in content.
    
    Args:
        doc (Document): The document to split
        add_tags_llm (bool): Whether to use LLM for tag extraction.
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
        # Truncate source filename for display and get just the filename
        display_source = format_source_filename(source)
            
        with tqdm(total=total_chunks, desc=f"Processing {display_source}", unit="chunk", leave=False) as pbar:
            for chunk in doc_chunks:
                try:
                    # Add metadata (from content or LLM based on flag)
                    chunk = add_metadata_to_document(chunk, add_tags_llm=add_tags_llm)
                    
                    # Add file metadata (ensure it doesn't overwrite extracted/LLM tags)
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

    """Process a single document using utility functions, split it, add metadata. Returns processed chunks.

    Args:
        doc (Document): The document to process
        add_tags_llm (bool): Whether to use LLM for tag extraction.
        
    Returns:
        List[Document]: List of processed document chunks, or empty list on error.
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
            return [] # Return empty list if splitting/metadata fails
            
        # Ensure source metadata is present in all chunks
        for chunk in chunks:
             if not chunk.metadata:
                 chunk.metadata = {}
             if "source" not in chunk.metadata:
                  chunk.metadata["source"] = source # Add source if missing

        return chunks # Return the processed chunks
            
    except Exception as e:
        logger.error(f"Error processing document {source}: {str(e)}", exc_info=True)
        return [] # Return empty list on error

def clear_vectorstore_dir(db_dir: Path):
    """Clear the KAG database directory."""
    try:
        if db_dir.exists():
            logger.info(f"Clearing KAG database directory: {db_dir}")
            for item in db_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        # Be careful here, ensure it's the correct directory structure
                        # If vectorstore is a sub-directory, target that specifically
                        # Assuming db_dir IS the vectorstore directory for KAG based on old structure
                        shutil.rmtree(item)
                except Exception as e:
                    logger.error(f"Error removing {item}: {str(e)}")
            logger.info(f"Cleared KAG database directory: {db_dir}")
        else:
            logger.info(f"KAG database directory {db_dir} does not exist, nothing to clear.")
    except Exception as e:
        logger.error(f"Error clearing KAG directory {db_dir}: {str(e)}")

# --- initialize_vectorstore function removed, imported from processing_utils ---

def main():
    """Main function to populate the KAG FAISS database."""
    parser = argparse.ArgumentParser(description="Populate the KAG FAISS database.")
    parser.add_argument("--name", type=str, default="kag", help="Name for the database instance (determines directory).")
    parser.add_argument("--reset", action="store_true", help="Reset the vectorstore before processing.")
    parser.add_argument("--add-tags", action="store_true", help="Enable LLM-based tag generation if tags are not found in the document content.")
    args = parser.parse_args()

    # --- Get dynamic paths based on name ---
    # KAG primarily uses a vectorstore, potentially within its named directory
    db_paths = get_db_paths(args.name)
    db_dir = db_paths["db_dir"] # This is the main directory for the instance, e.g., databases/kag_custom
    vectorstore_path = db_paths["vectorstore_path"] # Path within the instance dir, e.g., databases/kag_custom/vectorstore
    logger.info(f"Using database name: {args.name}")
    logger.info(f"Database directory: {db_dir}")
    logger.info(f"Vectorstore path: {vectorstore_path}")


    try:
        # Initialize vectorstore using dynamic path
        # Note: The initialize_vectorstore from utils handles reset internally based on the path
        vectorstore = initialize_vectorstore(vectorstore_path, args.reset)
        if not vectorstore:
            logger.error(f"Failed to initialize vectorstore for '{args.name}' at {vectorstore_path}")
            return
            
        # Process documents
        documents = load_documents()
        if not documents:
            logger.error("No documents found to process")
            return
            
        total_docs = len(documents)
        processed_docs = 0
        failed_docs = 0
        all_processed_chunks = []
        
        with tqdm(total=total_docs, desc="Processing documents", unit="doc") as pbar:
            for doc in documents:
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
                    pbar.set_postfix({
                        "processed": processed_docs,
                        "failed": failed_docs
                    })
        
        # Add collected chunks to vectorstore if any exist
        if all_processed_chunks:
            logger.info(f"Adding {len(all_processed_chunks)} processed chunks to the vectorstore...")
            vectorstore.add_documents(all_processed_chunks)
            logger.info("Chunks added successfully.")
        else:
             logger.warning("No valid chunks were processed to add to the vectorstore.")

        # Save final vectorstore using dynamic path
        logger.info(f"Saving vectorstore for '{args.name}' to {vectorstore_path}...")
        vectorstore.save_local(str(vectorstore_path)) # FAISS expects string path

        # Log statistics
        logger.info(f"KAG database population completed for '{args.name}':")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed: {processed_docs}")
        logger.info(f"- Failed: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info(f"Successfully populated KAG database '{args.name}'")

    except Exception as e:
        logger.error(f"Error populating KAG database '{args.name}': {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 