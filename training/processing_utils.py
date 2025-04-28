import sys
import os
from pathlib import Path

# Add project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
import os
import logging
import re
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter # Moved back to top level
# from langchain_text_splitters import RecursiveCharacterTextSplitter # Will be imported inside the function
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma # Added Chroma import

# Assuming these are potentially shared or can be passed as arguments if needed
from get_embedding_function import get_embedding_function
from extract_metadata_llm import add_metadata_to_document, format_source_filename
from load_documents import extract_metadata # Keep extract_metadata here if it's tightly coupled

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
        logger.debug(f"Metadata missing required fields: {required_fields}. Found: {list(metadata.keys())}")
        return False

    # Validate source
    if not metadata["source"] or not isinstance(metadata["source"], str):
        logger.debug(f"Invalid source in metadata: {metadata.get('source')}")
        return False

    # Validate file name
    if not metadata["file_name"] or not isinstance(metadata["file_name"], str):
        logger.debug(f"Invalid file_name in metadata: {metadata.get('file_name')}")
        return False

    # Validate file type
    if not metadata["file_type"] or not isinstance(metadata["file_type"], str):
        logger.debug(f"Invalid file_type in metadata: {metadata.get('file_type')}")
        return False

    return True


def split_document(doc: Document, add_tags_llm: bool, max_chunk_size: int = 1500, max_total_chunks: int = 1000) -> List[Document]:
    """
    Split a single document into chunks with semantic boundaries.
    Optionally adds LLM-based metadata if add_tags_llm is True and no tags found in content.

    Args:
        doc (Document): The document to split
        add_tags_llm (bool): Whether to use LLM for tag extraction.
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
        # Truncate source filename for display
        display_source = format_source_filename(source)

        with tqdm(total=total_chunks, desc=f"Processing {display_source}", unit="chunk", leave=False) as pbar:
            for i, chunk in enumerate(doc_chunks):
                try:
                    # Add metadata (from content or LLM based on flag)
                    # Note: add_metadata_to_document might need adjustment if its dependencies change
                    chunk = add_metadata_to_document(chunk, add_tags_llm=add_tags_llm)

                    # Add file metadata (ensure it doesn't overwrite extracted/LLM tags)
                    # Note: extract_metadata might need adjustment if its dependencies change
                    file_metadata = extract_metadata(chunk.metadata.get("source", ""))
                    # Merge carefully, prioritizing existing keys from add_metadata_to_document
                    merged_metadata = {**file_metadata, **chunk.metadata}
                    chunk.metadata = merged_metadata


                    # Add chunk-specific metadata
                    chunk.metadata.update({
                        "chunk_index": i, # Use enumerate index
                        "total_chunks": total_chunks,
                        "processed_at": datetime.now().isoformat()
                    })

                    processed_chunks.append(chunk)
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing chunk {i} from {source}: {str(e)}")
                    continue

        return processed_chunks

    except Exception as e:
        logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        return []


def initialize_vectorstore(vectorstore_path: Path, reset: bool = False) -> Optional[FAISS]:
    """Initialize or load the FAISS vectorstore.

    Args:
        vectorstore_path (Path): Path to the vectorstore directory.
        reset (bool): Whether to reset the vectorstore.

    Returns:
        Optional[FAISS]: The initialized vectorstore or None if initialization failed.
    """
    try:
        embedding_function = get_embedding_function() # Get embedding function here

        if reset:
            logger.info(f"Resetting vectorstore at {vectorstore_path}...")
            if vectorstore_path.exists():
                shutil.rmtree(vectorstore_path)
            # Ensure parent directory exists after potential removal
            vectorstore_path.parent.mkdir(parents=True, exist_ok=True)

        # Create initial vectorstore if it doesn't exist or if reset
        if not vectorstore_path.exists() or reset:
            logger.info(f"Creating initial vectorstore at {vectorstore_path}")
            # Create with a dummy document to initialize structure
            vectorstore = FAISS.from_texts(
                ["Initial empty document"],
                embedding=embedding_function,
                metadatas=[{
                    "source": "initial",
                    "file_name": "initial.txt",
                    "file_type": "text",
                    "processed_at": datetime.now().isoformat()
                }]
            )
            vectorstore.save_local(str(vectorstore_path)) # FAISS expects string path
            logger.info(f"Initial vectorstore created and saved at {vectorstore_path}")
            return vectorstore
        else:
            # Load existing vectorstore
            logger.info(f"Loading existing vectorstore from {vectorstore_path}")
            # Allow dangerous deserialization for FAISS loading if needed
            vectorstore = FAISS.load_local(str(vectorstore_path), embedding_function, allow_dangerous_deserialization=True)
            logger.info(f"Existing vectorstore loaded from {vectorstore_path}")
            return vectorstore

    except Exception as e:
        logger.error(f"Error initializing/loading FAISS vectorstore at {vectorstore_path}: {str(e)}")
        return None


def initialize_chroma_vectorstore(chroma_path: Path, reset: bool = False) -> Optional[Chroma]:
    """Initialize or load the Chroma vectorstore.

    Args:
        chroma_path (Path): Path to the Chroma database directory.
        reset (bool): Whether to reset the vectorstore.

    Returns:
        Optional[Chroma]: The initialized Chroma vectorstore or None if initialization failed.
    """
    try:
        embedding_function = get_embedding_function() # Get embedding function here
        vectorstore_path_str = str(chroma_path)

        if reset:
            logger.info(f"Resetting Chroma vectorstore at {vectorstore_path_str}...")
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            # Ensure parent directory exists after potential removal
            chroma_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if the Chroma database likely exists (presence of key files)
        # Chroma structure might vary, but checking for the directory is a start
        if chroma_path.exists() and not reset:
             # Attempt to load existing vectorstore
            logger.info(f"Loading existing Chroma vectorstore from {vectorstore_path_str}")
            vectorstore = Chroma(persist_directory=vectorstore_path_str, embedding_function=embedding_function)
            logger.info(f"Existing Chroma vectorstore loaded from {vectorstore_path_str}")
            return vectorstore
        else:
             # Create new vectorstore (Chroma initializes directory on first add/persist)
            logger.info(f"Creating new Chroma vectorstore instance for path {vectorstore_path_str}")
            # Chroma doesn't need dummy data like FAISS for initialization,
            # but needs the directory path and embedding function.
            # The directory will be created/used when data is added and persisted.
            vectorstore = Chroma(persist_directory=vectorstore_path_str, embedding_function=embedding_function)
            logger.info(f"New Chroma vectorstore instance created for {vectorstore_path_str}")
            return vectorstore

    except Exception as e:
        logger.error(f"Error initializing/loading Chroma vectorstore at {vectorstore_path_str}: {str(e)}")
        return None


def clear_db_directory(db_dir: Path):
    """Safely clear the contents of a database directory."""
    try:
        if db_dir.exists() and db_dir.is_dir():
            logger.info(f"Clearing database directory: {db_dir}")
            for item in db_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                        logger.debug(f"Removed file: {item}")
                    elif item.is_dir():
                        shutil.rmtree(item)
                        logger.debug(f"Removed directory: {item}")
                except Exception as e:
                    logger.error(f"Error removing item {item} in {db_dir}: {str(e)}")
            logger.info(f"Successfully cleared directory: {db_dir}")
        else:
            logger.info(f"Database directory {db_dir} does not exist or is not a directory, nothing to clear.")
    except Exception as e:
        logger.error(f"Error clearing directory {db_dir}: {str(e)}")