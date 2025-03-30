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
from query.database_paths import VECTORSTORE_PATH, KAG_DB_DIR
from load_documents import load_documents, process_single_file, extract_metadata
import re
import argparse
import shutil
import json
from datetime import datetime

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
        # Truncate source filename for display and get just the filename
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

def process_document(doc: Document, vectorstore: FAISS, reset: bool = False) -> None:
    """Process a single document and add it to the vectorstore.
    
    Args:
        doc (Document): The document to process
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
            
        # Add chunks to vectorstore
        for chunk in chunks:
            try:
                # Ensure chunk has source metadata
                if not chunk.metadata:
                    chunk.metadata = {}
                chunk.metadata["source"] = source
                
                # Add to vectorstore
                vectorstore.add_documents([chunk])
            except Exception as e:
                logger.error(f"Error adding chunk to vectorstore: {str(e)}")
                continue
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)

def clear_vectorstore():
    """Clear the KAG database."""
    try:
        if os.path.exists(KAG_DB_DIR):
            for file in os.listdir(KAG_DB_DIR):
                file_path = os.path.join(KAG_DB_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    logger.error(f"Error removing {file_path}: {str(e)}")
            logger.info("Cleared KAG database")
    except Exception as e:
        logger.error(f"Error clearing vectorstore: {str(e)}")

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
            clear_vectorstore()
            
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
    """Main function to populate the KAG database."""
    parser = argparse.ArgumentParser(description="Populate KAG database with documents")
    parser.add_argument("--reset", action="store_true", help="Reset the vectorstore before processing")
    args = parser.parse_args()
    
    try:
        # Initialize vectorstore
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
                    process_document(doc, vectorstore, args.reset)
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
                    
        # Save final vectorstore
        vectorstore.save_local(VECTORSTORE_PATH)
        
        # Log statistics
        logger.info(f"KAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed: {processed_docs}")
        logger.info(f"- Failed: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated KAG database")
            
    except Exception as e:
        logger.error(f"Error populating KAG database: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 