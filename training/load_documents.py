"""
Shared document loading functionality for RAG implementations.

This module provides a unified interface for loading documents from various file types,
including PDFs, text files, and markdown files. It handles:
1. Document loading and preprocessing
2. Content validation
3. Metadata extraction
4. Consistent formatting
"""

import logging
from typing import List, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.schema.document import Document
import re
from config import DATA_DIR

logger = logging.getLogger(__name__)

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dictionary containing basic file metadata
    """
    try:
        path = Path(file_path)
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "file_extension": path.suffix,
            "file_type": path.suffix[1:].upper() if path.suffix else "UNKNOWN"
        }
        # Remove producer field if it exists
        if "producer" in metadata:
            del metadata["producer"]
        return metadata
    except Exception as e:
        logger.error(f"Error extracting file metadata: {str(e)}")
        return {
            "source": file_path,
            "file_name": "unknown",
            "file_extension": "",
            "file_type": "UNKNOWN"
        }

def preprocess_text(text: str) -> str:
    """Clean and normalize text before chunking."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def validate_document(doc: Document) -> bool:
    """Validate document content and metadata."""
    if not doc.page_content:
        return False
        
    content_length = len(doc.page_content.strip())
    if content_length < 10:
        return False
        
    return True

def process_single_file(file_path: Path) -> List[Document]:
    """Process a single file and return its documents."""
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            if not documents:
                return []
            
            # Combine all pages into a single document
            combined_content = "\n".join(doc.page_content for doc in documents)
            combined_metadata = documents[0].metadata.copy()
            combined_metadata.update(extract_metadata(str(file_path)))
            # Remove page-specific metadata
            combined_metadata.pop('page', None)
            combined_metadata.pop('page_label', None)
            combined_metadata.pop('total_pages', None)
            
            documents = [Document(
                page_content=combined_content,
                metadata=combined_metadata
            )]
        elif file_path.suffix.lower() == '.txt':
            loader = TextLoader(str(file_path))
            documents = loader.load()
        elif file_path.suffix.lower() == '.md':
            loader = UnstructuredMarkdownLoader(str(file_path))
            documents = loader.load()
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []

        if not documents:
            return []

        # Preprocess and validate documents
        valid_docs = []
        for doc in documents:
            # Remove producer field from metadata if it exists
            if "producer" in doc.metadata:
                del doc.metadata["producer"]
            
            # Remove creator field if it's longer than 60 characters
            if "creator" in doc.metadata and len(doc.metadata["creator"]) > 60:
                del doc.metadata["creator"]
                
            doc.page_content = preprocess_text(doc.page_content)
            if validate_document(doc):
                doc.metadata.update(extract_metadata(str(file_path)))
                valid_docs.append(doc)

        return valid_docs
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return []

def load_documents() -> List[Document]:
    """Load and preprocess documents from various file types."""
    all_documents = []
    logger.info(f"Loading documents from {DATA_DIR}")
    
    # Get all supported files
    supported_extensions = {'.pdf', '.txt', '.md'}
    all_files = []
    for ext in supported_extensions:
        all_files.extend(list(DATA_DIR.glob(f"**/*{ext}")))
    
    if not all_files:
        logger.error("No supported files found to process")
        return all_documents
        
    total_files = len(all_files)
    processed_files = 0
    failed_files = 0
    
    # Process files one by one
    for file_path in all_files:
        try:
            logger.debug(f"Processing file {processed_files + 1}/{total_files}: {file_path.name}")
            documents = process_single_file(file_path)
            
            if documents:
                all_documents.extend(documents)
                processed_files += 1
            else:
                failed_files += 1
                logger.warning(f"No valid documents found in file: {file_path.name}")
                
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}")
            failed_files += 1
            continue
    
    # Log final statistics
    logger.info(f"Total files: {total_files}")
    if failed_files > 0:
        logger.info(f"Failed files: {failed_files}")
    
    return all_documents 