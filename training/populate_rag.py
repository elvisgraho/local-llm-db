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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from .get_embedding_function import get_embedding_function
from .extract_metadata_llm import extract_metadata_llm
from query.database_paths import CHROMA_PATH, RAG_DB_DIR
from .load_documents import load_documents, extract_metadata, process_single_file
import re
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_document(doc: Document) -> List[Document]:
    """Split a single document into chunks with improved parameters for security documentation."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n## ",  # Main headers
            "\n\n### ",  # Subheaders
            "\n\n#### ",  # Sub-subheaders
            "\n```",  # Code blocks
            "\n\n",     # Double newlines
            "\n**",
            "\n",       # Single newlines
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
        
        # Process each chunk with LLM metadata extraction
        processed_chunks = []
        for chunk in doc_chunks:
            # Extract LLM-based metadata
            llm_metadata = extract_metadata_llm(chunk.page_content)
            chunk.metadata.update(llm_metadata)
            
            # Add file-based metadata
            chunk.metadata.update(extract_metadata(chunk.metadata.get("source", "")))
            
            processed_chunks.append(chunk)
            
        return processed_chunks
            
    except Exception as e:
        logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        return []

def process_document(doc: Document) -> None:
    """Process a single document and update the vectorstore."""
    embeddings = get_embedding_function()
    
    # Split document into chunks
    chunks = split_document(doc)
    if not chunks:
        logger.warning(f"No valid chunks created for document: {doc.metadata.get('source', 'unknown')}")
        return
    
    # Initialize or update vectorstore
    try:
        vectorstore_path = str(CHROMA_PATH)
        if os.path.exists(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "chroma.sqlite3")):
            logger.info("Loading existing vectorstore")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            
            # Check for duplicates before adding
            existing_docs = vectorstore.get()
            existing_sources = set(doc.metadata.get("source", "") for doc in existing_docs["documents"])
            new_chunks = [chunk for chunk in chunks if chunk.metadata.get("source", "") not in existing_sources]
            
            if new_chunks:
                vectorstore.add_documents(new_chunks)
                logger.info(f"Added {len(new_chunks)} new chunks to vectorstore")
            else:
                logger.info("No new chunks to add - all chunks already exist in vectorstore")
        else:
            logger.info("Creating new vectorstore")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=vectorstore_path)
    except Exception as e:
        logger.warning(f"Failed to load existing vectorstore: {str(e)}")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=vectorstore_path)
    
    # Save after each document processing
    os.makedirs(RAG_DB_DIR, exist_ok=True)
    vectorstore.persist_directory = vectorstore_path  # Set persist directory
    logger.info(f"Updated vectorstore saved to {vectorstore_path}")

def clear_vectorstore():
    """Clear the RAG database."""
    if os.path.exists(RAG_DB_DIR):
        for root, dirs, files in os.walk(RAG_DB_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        logger.info("Cleared RAG database")

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
    """Main function to populate the RAG database file by file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()
    
    if args.reset:
        clear_vectorstore()
        logger.info("Cleared existing RAG database")

    try:
        # Get all documents using load_documents functionality
        all_documents = load_documents()
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return
            
        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
            
        # Process documents one by one
        for doc in tqdm(all_documents, desc="Processing documents", total=total_docs):
            try:
                logger.info(f"Processing document {processed_docs + 1}/{total_docs}")
                process_document(doc)
                processed_docs += 1
                    
            except Exception as e:
                logger.error(f"Error processing document from {doc.metadata.get('source', 'unknown')}: {str(e)}")
                failed_docs += 1
                continue
        
        # Log final statistics
        logger.info(f"RAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed documents: {processed_docs}")
        logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated RAG database")
        
    except Exception as e:
        logger.error(f"Error in database population: {str(e)}")
        raise

if __name__ == "__main__":
    main()
