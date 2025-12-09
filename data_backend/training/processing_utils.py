import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Modern LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma

# --- Add project root to path for local imports ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Local Imports ---
from training.get_embedding_function import get_embedding_function
from training.extract_metadata_llm import add_metadata_to_document
from training.load_documents import extract_metadata

logger = logging.getLogger(__name__)

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate document metadata for required fields.
    """
    if not metadata or not isinstance(metadata, dict):
        return False

    required_fields = ["source", "file_name", "file_type"]
    
    # Check for missing keys
    if not all(field in metadata for field in required_fields):
        missing = [f for f in required_fields if f not in metadata]
        logger.debug(f"Metadata missing fields: {missing}")
        return False

    # Check for empty values
    for field in required_fields:
        if not metadata[field] or not isinstance(metadata[field], str):
            logger.debug(f"Invalid value for metadata field '{field}': {metadata.get(field)}")
            return False

    return True

def split_document(
    doc: Document, 
    add_tags_llm: bool, 
    max_chunk_size: int = 512, 
    max_total_chunks: int = 1000
) -> List[Document]:
    """
    Split a document into semantic chunks.
    OPTIMIZED: Generates metadata ONCE per document, then splits.
    """
    # --- 1. Generate Metadata (ONCE per file) ---
    if add_tags_llm:
        try:
            # Generate tags based on the first 5000 chars of the document
            doc = add_metadata_to_document(doc, add_tags_llm=True)
        except Exception as e:
            # Safe logging of the filename
            source = doc.metadata.get('source', 'unknown')
            try:
                safe_source = source.encode('utf-8', 'replace').decode('utf-8')
            except:
                safe_source = "unknown_file"
            
            logger.error(f"❌ CRITICAL: Tagging failed for '{safe_source}'. Error: {e}")
            return [] 

    # --- 2. Configure Splitter ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n##", "\n###", "\n", ". ", " ", ""],
        keep_separator=True
    )

    try:
        # --- 3. Pre-process Content ---
        content = doc.page_content or ""
        content = re.sub(r'^\s*[-•]\s*', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+\.)(?! )', r'\1 ', content, flags=re.MULTILINE)
        content = re.sub(r'```(\w+)?\n', r'```\n', content)
        doc.page_content = content

        # --- 4. Split (Metadata is automatically inherited) ---
        doc_chunks = text_splitter.split_documents([doc])

        # Limit Chunks
        if len(doc_chunks) > max_total_chunks:
            source_name = doc.metadata.get('source', 'unknown')
            try:
                logger.warning(f"Limiting {source_name} to {max_total_chunks} chunks.")
            except UnicodeEncodeError:
                logger.warning(f"Limiting document to {max_total_chunks} chunks (name hidden due to encoding).")
            doc_chunks = doc_chunks[:max_total_chunks]

        # --- 5. Finalize Metadata ---
        processed_chunks = []
        total_chunks = len(doc_chunks)
        source = doc.metadata.get("source", "unknown")
        
        for i, chunk in enumerate(doc_chunks):
            # Extract base file metadata
            file_metadata = extract_metadata(source)
            
            # Merge: File Meta + LLM Meta (Inherited) + Structural Meta
            merged_metadata = {**file_metadata, **chunk.metadata}
            merged_metadata.update({
                "chunk_index": i,
                "total_chunks": total_chunks,
                "processed_at": datetime.now().isoformat()
            })
            
            chunk.metadata = merged_metadata
            processed_chunks.append(chunk)

        return processed_chunks

    except Exception as e:
        # Safe logging for top-level error
        try:
            source = doc.metadata.get('source', 'unknown')
            logger.error(f"Failed to split document {source}: {e}")
        except UnicodeEncodeError:
            logger.error(f"Failed to split document (unknown source): {e}")
        return []
    
    
def initialize_vectorstore(vectorstore_path: Path, reset: bool = False) -> Optional[FAISS]:
    """
    Initialize or load the LEGACY FAISS vectorstore.
    """
    try:
        embedding_function = get_embedding_function()

        if reset and vectorstore_path.exists():
            logger.info(f"Resetting FAISS vectorstore at {vectorstore_path}...")
            shutil.rmtree(vectorstore_path)
        
        vectorstore_path.parent.mkdir(parents=True, exist_ok=True)

        if not vectorstore_path.exists() or reset:
            logger.info(f"Creating new FAISS index at {vectorstore_path}")
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
            vectorstore.save_local(str(vectorstore_path))
            return vectorstore
        else:
            logger.info(f"Loading FAISS index from {vectorstore_path}")
            return FAISS.load_local(
                str(vectorstore_path), 
                embedding_function, 
                allow_dangerous_deserialization=True
            )

    except Exception as e:
        logger.error(f"Error managing FAISS vectorstore: {e}")
        return None

def initialize_chroma_vectorstore(chroma_path: Path, reset: bool = False) -> Optional[Chroma]:
    """
    Initialize or load the Chroma vectorstore.
    """
    try:
        embedding_function = get_embedding_function()
        path_str = str(chroma_path)

        if reset:
            logger.info(f"Resetting ChromaDB at {path_str}...")
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            chroma_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing ChromaDB at {path_str}")
        
        vectorstore = Chroma(
            persist_directory=path_str, 
            embedding_function=embedding_function
        )
        
        return vectorstore

    except Exception as e:
        logger.error(f"Error managing ChromaDB at {chroma_path}: {e}")
        return None

def clear_db_directory(db_dir: Path) -> None:
    """
    Safely clear the contents of a database directory.
    """
    try:
        if db_dir.exists() and db_dir.is_dir():
            logger.info(f"Clearing directory: {db_dir}")
            for item in db_dir.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.error(f"Failed to remove {item}: {e}")
            logger.info("Directory cleared.")
        else:
            logger.info(f"Directory {db_dir} does not exist or is not a dir.")
    except Exception as e:
        logger.error(f"Error clearing {db_dir}: {e}")