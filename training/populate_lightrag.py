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

import os
import logging
from typing import List
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from get_embedding_function import get_embedding_function
from extract_metadata_llm import add_metadata_to_document, format_source_filename
from query.database_paths import VECTORSTORE_PATH, LIGHT_RAG_DB_DIR
from load_documents import load_documents, process_single_file, extract_metadata
import re
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_document(doc: Document, max_chunk_size: int = 1500, max_total_chunks: int = 1000) -> List[Document]:
    """Split a single document into chunks with improved parameters for security documentation."""
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
            "\n```",  # Code blocks
            "\n",       # Single newlines
            "\n**",
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
        
        # Limit total number of chunks if document is too large
        if len(doc_chunks) > max_total_chunks:
            logger.warning(f"Document {doc.metadata.get('source', 'unknown')} has too many chunks ({len(doc_chunks)}). Limiting to {max_total_chunks} chunks.")
            doc_chunks = doc_chunks[:max_total_chunks]
        
        # Process each chunk with LLM metadata extraction
        processed_chunks = []
        total_chunks = len(doc_chunks)
        source = doc.metadata.get("source", "unknown")
        # Truncate source filename for display
        display_source = format_source_filename(source)
        
        with tqdm(total=total_chunks, desc=f"Processing {display_source}", unit="chunk", leave=False) as pbar:
            for chunk in doc_chunks:
                try:
                    # Add LLM-based metadata using the helper function
                    chunk = add_metadata_to_document(chunk)
                    
                    # Add file-based metadata
                    chunk.metadata.update(extract_metadata(chunk.metadata.get("source", "")))
                    
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
            
        # Split document into chunks
        chunks = split_document(doc)
        
        # Add chunks to vectorstore
        for chunk in chunks:
            # Ensure chunk has source metadata
            if not chunk.metadata:
                chunk.metadata = {}
            chunk.metadata["source"] = source
            
            # Add to vectorstore
            vectorstore.add_documents([chunk])
            
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)

def clear_vectorstore():
    """Clear the Light RAG database."""
    if os.path.exists(LIGHT_RAG_DB_DIR):
        for file in os.listdir(LIGHT_RAG_DB_DIR):
            os.remove(os.path.join(LIGHT_RAG_DB_DIR, file))
        logger.info("Cleared Light RAG database")

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
    parser = argparse.ArgumentParser(description="Populate LightRAG database with documents")
    parser.add_argument("--reset", action="store_true", help="Reset the vectorstore before processing")
    args = parser.parse_args()
    
    try:
        # Initialize vectorstore
        if args.reset:
            logger.info("Resetting vectorstore...")
            if os.path.exists(VECTORSTORE_PATH):
                shutil.rmtree(VECTORSTORE_PATH)
            vectorstore = FAISS.from_texts(
                ["Initial empty document"],
                embedding_function=get_embedding_function(),
                metadatas=[{"source": "initial"}]
            )
        else:
            try:
                vectorstore = FAISS.load_local(VECTORSTORE_PATH, get_embedding_function())
                logger.info("Loaded existing vectorstore")
            except Exception as e:
                logger.warning(f"Could not load existing vectorstore: {str(e)}")
                vectorstore = FAISS.from_texts(
                    ["Initial empty document"],
                    embedding_function=get_embedding_function(),
                    metadatas=[{"source": "initial"}]
                )
        
        # Process documents
        for doc in load_documents():
            process_document(doc, vectorstore, args.reset)
            
        # Save vectorstore
        vectorstore.save_local(VECTORSTORE_PATH)
        logger.info("LightRAG database populated successfully")
        
    except Exception as e:
        logger.error(f"Error populating LightRAG database: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 