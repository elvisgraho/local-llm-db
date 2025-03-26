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
from .get_embedding_function import get_embedding_function
from .extract_metadata_llm import extract_metadata_llm
from backend.database_paths import VECTORSTORE_PATH, LIGHT_RAG_DB_DIR
from .load_documents import load_documents, process_single_file, extract_metadata
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
    # Get embedding function
    embedding_function = get_embedding_function()
    
    # Split document into chunks
    chunks = split_document(doc)
    if not chunks:
        logger.warning(f"No valid chunks created for document: {doc.metadata.get('source', 'unknown')}")
        return
    
    # Initialize or update vectorstore
    try:
        # Check if the vectorstore directory exists and has the required files
        if os.path.exists(VECTORSTORE_PATH) and os.path.exists(os.path.join(VECTORSTORE_PATH, "index.faiss")):
            logger.info("Loading existing vectorstore")
            vectorstore = FAISS.load_local(
                VECTORSTORE_PATH, 
                embedding_function,
                allow_dangerous_deserialization=True  # Safe since we created the file ourselves
            )
            # Add new chunks to existing vectorstore
            vectorstore.add_documents(chunks)
        else:
            logger.info("Creating new vectorstore")
            vectorstore = FAISS.from_documents(chunks, embedding_function)
    except Exception as e:
        logger.warning(f"Failed to load existing vectorstore: {str(e)}")
        # If loading fails, create a new vectorstore with current chunks
        vectorstore = FAISS.from_documents(chunks, embedding_function)
    
    # Save after each document processing
    os.makedirs(LIGHT_RAG_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info(f"Updated vectorstore saved to {VECTORSTORE_PATH}")

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
    """Main function to populate the LightRAG database file by file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()
    
    if args.reset:
        clear_vectorstore()
        logger.info("Cleared existing LightRAG database")

    try:
        # Get all documents using load_documents functionality
        all_documents = load_documents()
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return
            
        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
            
        # Process files one by one
        for doc in tqdm(all_documents, desc="Processing documents", total=total_docs):
            try:
                logger.info(f"Processing document {processed_docs + 1}/{total_docs}")
                # Use process_file_to_vectorstore for LightRAG implementation
                process_file_to_vectorstore(Path(doc.metadata.get("source", "")))
                processed_docs += 1
                    
            except Exception as e:
                logger.error(f"Error processing document from {doc.metadata.get('source', 'unknown')}: {str(e)}")
                failed_docs += 1
                continue
        
        # Log final statistics
        logger.info(f"LightRAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed documents: {processed_docs}")
        logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated LightRAG database")
        
    except Exception as e:
        logger.error(f"Error in database population: {str(e)}")
        raise

if __name__ == "__main__":
    main() 