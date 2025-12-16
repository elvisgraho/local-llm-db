import json
import logging
import re
import shutil
import sys
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Modern LangChain Imports ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings

# --- Add project root to path for local imports ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Local Imports ---
from training.get_embedding_function import get_embedding_function
from training.extract_metadata_llm import add_metadata_to_document
from training.load_documents import extract_metadata

logger = logging.getLogger(__name__)

RE_BULLET = re.compile(r'^\s*[-•]\s*', flags=re.MULTILINE)
RE_NUMBERED_LIST = re.compile(r'^\s*(\d+\.)(?! )', flags=re.MULTILINE)
RE_CODE_BLOCK = re.compile(r'```(\w+)?\n')
RE_FILENAME_SANITIZE = re.compile(r'[^a-z0-9_]')

def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate document metadata for required fields efficiently.
    """
    if not isinstance(metadata, dict):
        return False

    required_fields = ["source", "file_name"]
    for field in required_fields:
        val = metadata.get(field)
        if not val or not isinstance(val, str):
            # Allow file_name to be missing if source is present (will be extracted later)
            if field == "file_name" and metadata.get("source"):
                continue
            return False
    return True

def split_document(
    doc: Document, 
    add_tags_llm: bool, 
    max_total_chunks: int = 1500,
    chunk_overlap: int = 100, 
    chunk_size: Optional[int] = None
) -> List[Document]:
    """
    Split a document into semantic chunks with NOISE FILTERING and FLATTENING for Chroma.
    """
    
    # 1. Generate Metadata (LLM or Manual Tags)
    if add_tags_llm:
        try:
            # This now handles Manual Tags + LLM internally
            doc = add_metadata_to_document(doc, add_tags_llm=True)
            
            if doc is None:
                return [] # Dropped as non-technical/junk
            
        except Exception as e:
            source = doc.metadata.get('source', 'unknown')
            logger.error(f"❌ CRITICAL: Tagging failed for '{source}'. Error: {e}")
            # Continue even if tagging fails, we just lose rich metadata
            pass 

    # 2. Configure Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
        keep_separator=True
    )

    try:
        # 3. Pre-process Content (Fast Regex)
        content = doc.page_content or ""
        content = RE_BULLET.sub('• ', content)
        content = RE_NUMBERED_LIST.sub(r'\1 ', content)
        content = RE_CODE_BLOCK.sub(r'```\n', content)
        doc.page_content = content

        # 4. Split
        raw_chunks = text_splitter.split_documents([doc])

        valid_chunks = []
        for chunk in raw_chunks:
            text = chunk.page_content.strip()
            
            # A. Skip Empty/Noise
            if not text or len(text) < 50:
                continue
                
            # B. Skip "Header Orphans"
            if text.startswith('#') and len(text) < 150 and text.count('\n') < 2:
                continue
                
            valid_chunks.append(chunk)

        # Fallback: If strict filtering removed everything, keep raw
        doc_chunks = valid_chunks if valid_chunks else (raw_chunks if len(doc.page_content) > 50 else [])

        # Limit Chunks
        if len(doc_chunks) > max_total_chunks:
            doc_chunks = doc_chunks[:max_total_chunks]

        # 5. Finalize Metadata (FLATTENING STEP)
        processed_chunks = []
        total_chunks = len(doc_chunks)
        
        # Base metadata from file path
        source = doc.metadata.get("source", "unknown")
        file_metadata = extract_metadata(source)
        base_timestamp = datetime.now().isoformat()

        for i, chunk in enumerate(doc_chunks):
            # Start with file info
            merged_metadata = file_metadata.copy()
            # Update with doc-level metadata (LLM tags, manual tags)
            merged_metadata.update(doc.metadata) 
            # Update with chunk-specifics (if any exist)
            merged_metadata.update(chunk.metadata)

            # --- CRITICAL: FLATTEN FOR CHROMA ---
            # ChromaDB crashes if metadata values are Lists or Dictionaries.
            # We must convert them to strings.
            clean_metadata = {}
            
            for key, value in merged_metadata.items():
                if value is None:
                    clean_metadata[key] = ""
                    
                # Handle Entities List [{'name': 'X', 'category': 'Y'}]
                elif key == "entities" and isinstance(value, list):
                    # Format: "Mimikatz (TOOL), CVE-2023-1 (VULN)"
                    formatted_entities = []
                    for item in value:
                        if isinstance(item, dict):
                            formatted_entities.append(f"{item.get('name')} ({item.get('category', 'UNK')})")
                        else:
                            formatted_entities.append(str(item))
                    clean_metadata[key] = ", ".join(formatted_entities)

                # Handle Standard Lists (tactics, code_languages)
                elif isinstance(value, list):
                    clean_metadata[key] = ", ".join(map(str, value))
                    
                # Handle Booleans/Numbers/Strings
                else:
                    clean_metadata[key] = value

            clean_metadata.update({
                "chunk_index": i,
                "total_chunks": total_chunks,
                "processed_at": base_timestamp
            })
            
            chunk.metadata = clean_metadata
            processed_chunks.append(chunk)

        return processed_chunks

    except Exception as e:
        source = doc.metadata.get('source', 'unknown')
        logger.error(f"Failed to split document {source}: {e}")
        return []

def initialize_chroma_vectorstore(chroma_path: Path, reset: bool = False) -> Optional[Chroma]:
    try:
        embedding_function = get_embedding_function()
        path_str = str(chroma_path)

        if reset:
            clear_db_directory(chroma_path)

        logger.info(f"Initializing ChromaDB at {path_str}")
        vectorstore = Chroma(
            persist_directory=path_str, 
            embedding_function=embedding_function,
            client_settings=Settings(anonymized_telemetry=False)
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Error managing ChromaDB at {chroma_path}: {e}")
        return None

def manage_db_configuration(db_dir: Path, rag_type: str, args) -> None:
    config_path = db_dir / "db_config.json"
    
    # Validation Logic
    if config_path.exists() and not args.reset:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            old_chunk = existing.get("chunk_size")
            old_overlap = existing.get("chunk_overlap")
            
            if old_chunk is not None:
                if old_chunk != args.chunk_size or old_overlap != args.chunk_overlap:
                    logger.error(f"⛔ CONFIGURATION MISMATCH for '{args.db_name}'")
                    logger.error(f"   Stored:    Chunk={old_chunk}, Overlap={old_overlap}")
                    logger.error(f"   Requested: Chunk={args.chunk_size}, Overlap={args.chunk_overlap}")
                    sys.exit(1)
            logger.info("✅ Configuration match verified.")
        except Exception as e:
            logger.warning(f"Could not read existing config: {e}")

    # Creation Logic
    if args.reset or not config_path.exists():
        db_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "rag_type": rag_type,
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "created_at": str(datetime.now()),
                    "db_name": args.db_name
                }, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved database configuration to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save DB config: {e}")

def clear_db_directory(db_dir: Path) -> None:
    """
    Safely clear directory with Windows file lock handling.
    """
    if not db_dir.exists(): return

    logger.info(f"Clearing directory: {db_dir}")
    
    # Force Garbage Collection to release file handles (Crucial for Chroma on Windows)
    gc.collect() 
    
    # Retry loop for Windows permissions
    for attempt in range(3):
        try:
            if db_dir.exists():
                shutil.rmtree(db_dir)
            break
        except PermissionError:
            logger.warning(f"File locked (Attempt {attempt+1}/3). Waiting 1s...")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error clearing {db_dir}: {e}")
            break
    
    # Ensure it's recreated
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directory cleared and ready.")

def get_unique_path(out_dir: Path, filename: str) -> Path:
    """Guarantees no overwrites by appending a counter."""
    safe_name = RE_FILENAME_SANITIZE.sub('', filename.lower())
    if not safe_name: 
        safe_name = "untitled_doc"
    
    candidate = out_dir / f"{safe_name}.txt"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{safe_name}_{counter}.txt"
        counter += 1
    return candidate