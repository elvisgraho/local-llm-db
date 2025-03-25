import argparse
import os
import shutil
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def preprocess_text(text: str) -> str:
    """Clean and normalize text before chunking."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """Extract additional metadata from file path and content."""
    path = Path(file_path)
    return {
        "file_type": path.suffix.lower(),
        "file_name": path.name,
        "directory": str(path.parent),
        "created_date": os.path.getctime(file_path),
        "modified_date": os.path.getmtime(file_path)
    }

def validate_document(doc: Document) -> bool:
    """Validate document content and metadata."""
    if not doc.page_content or len(doc.page_content.strip()) < 10:
        logger.warning(f"Document {doc.metadata.get('source', 'unknown')} is too short or empty")
        return False
    return True

def load_documents() -> List[Document]:
    """Load documents from various file types."""
    loaders = {
        "**/*.pdf": PyPDFDirectoryLoader,
        "**/*.txt": TextLoader,
        "**/*.md": UnstructuredMarkdownLoader
    }
    
    all_documents = []
    for glob_pattern, loader_class in loaders.items():
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob=glob_pattern,
                loader_class=loader_class,
                show_progress=True
            )
            documents = loader.load()
            
            # Preprocess and validate documents
            for doc in documents:
                doc.page_content = preprocess_text(doc.page_content)
                if validate_document(doc):
                    # Add additional metadata
                    doc.metadata.update(extract_metadata(doc.metadata.get("source", "")))
                    all_documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {glob_pattern}")
        except Exception as e:
            logger.error(f"Error loading {glob_pattern}: {str(e)}")
    
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks with improved parameters for security documentation."""
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
    
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
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
            
            # Post-process chunks to ensure they maintain context
            for chunk in doc_chunks:
                # Add section context if available
                if "Scenario:" in chunk.page_content:
                    chunk.metadata["section_type"] = "scenario"
                elif "Mitigation" in chunk.page_content:
                    chunk.metadata["section_type"] = "mitigation"
                elif "Impact:" in chunk.page_content:
                    chunk.metadata["section_type"] = "impact"
                elif "Steps to Reproduce:" in chunk.page_content:
                    chunk.metadata["section_type"] = "reproduction"
                elif "Proof of Concept:" in chunk.page_content:
                    chunk.metadata["section_type"] = "poc"
                
                # Add technical indicators
                if "```" in chunk.page_content:
                    chunk.metadata["has_code"] = True
                if "Payload:" in chunk.page_content or "Example Payloads:" in chunk.page_content:
                    chunk.metadata["has_payload"] = True
                
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
    
    return chunks

def add_to_chroma(chunks: List[Document]):
    """Add documents to Chroma with improved error handling and progress tracking."""
    try:
        embedding_function = get_embedding_function()
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )

        chunks_with_ids = calculate_chunk_ids(chunks)
        existing_items = db.get(include=[])
        existing_ids = set(existing_items["ids"])
        logger.info(f"Number of existing documents in DB: {len(existing_ids)}")

        new_chunks = []
        for chunk in tqdm(chunks_with_ids, desc="Processing chunks"):
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if new_chunks:
            logger.info(f"Adding {len(new_chunks)} new documents")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            logger.info("No new documents to add")

    except Exception as e:
        logger.error(f"Error adding documents to Chroma: {str(e)}")
        raise

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    
    if args.reset:
        logger.info("Clearing Database")
        clear_database()

    try:
        documents = load_documents()
        if not documents:
            logger.error("No valid documents found to process")
            return
            
        chunks = split_documents(documents)
        if not chunks:
            logger.error("No valid chunks created")
            return
            
        add_to_chroma(chunks)
        logger.info("Database population completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
