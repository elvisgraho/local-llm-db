import logging
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from tqdm import tqdm

# --- Modern LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from query.database_paths import RAW_FILES_DIR

logger = logging.getLogger(__name__)

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a file path using pathlib.
    """
    try:
        path = Path(file_path)
        metadata = {
            "source": str(path.resolve()),
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "file_type": path.suffix[1:].upper() if path.suffix else "UNKNOWN"
        }
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return {
            "source": str(file_path),
            "file_name": "unknown",
            "file_type": "UNKNOWN"
        }

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text while PRESERVING paragraph structure.
    """
    if not text:
        return ""
        
    # 1. Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # 2. Collapse runs of horizontal whitespace (spaces, tabs) into a single space
    #    We do NOT match \n here to preserve line breaks
    text = re.sub(r'[ \t]+', ' ', text)
    
    # 3. Collapse multiple newlines into a max of two (paragraph break)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # 4. Remove leading/trailing whitespace
    return text.strip()

def validate_document(doc: Document) -> bool:
    """
    Validate document content and metadata.
    Returns False if content is too short or empty.
    """
    if not doc.page_content:
        return False
        
    # Filter out files that are just noise/empty (less than 10 chars)
    if len(doc.page_content.strip()) < 10:
        return False
        
    return True

# ==========================================
# 2. REGISTRY MANAGER (Incremental Loading)
# ==========================================

class RegistryManager:
    """
    Tracks processed files to support incremental indexing.
    Persists state to 'processed_files.json' in the database directory.
    """
    def __init__(self, db_dir: Path):
        self.registry_path = db_dir / "processed_files.json"
        self.registry: Dict[str, float] = self._load()

    def _load(self) -> Dict[str, float]:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}

    def save(self):
        """Save registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save registry: {e}")

    def is_processed(self, file_path: Path) -> bool:
        """
        Check if file has been processed and hasn't changed since.
        """
        key = str(file_path.resolve())
        if key not in self.registry:
            return False
        
        # Check if file was modified since last process
        try:
            last_mtime = self.registry[key]
            current_mtime = file_path.stat().st_mtime
            return current_mtime <= last_mtime
        except FileNotFoundError:
            return False

    def mark_processed(self, file_path: Path):
        """Update registry with current file modification time."""
        key = str(file_path.resolve())
        try:
            self.registry[key] = file_path.stat().st_mtime
        except FileNotFoundError:
            pass

# ==========================================
# 3. MAIN LOADING LOGIC
# ==========================================

def _process_single_file(file_path: Path) -> List[Document]:
    """
    Internal helper to load, clean, and validate a single file.
    """
    ext = file_path.suffix.lower()
    documents = []

    try:
        # --- 1. Load Raw Content ---
        if ext == '.pdf':
            loader = PyPDFLoader(str(file_path))
            raw_docs = loader.load()
            if raw_docs:
                # Merge pages to avoid arbitrary splitting
                text = "\n\n".join(doc.page_content for doc in raw_docs)
                # Use metadata from first page
                meta = raw_docs[0].metadata.copy() if raw_docs else {}
                documents = [Document(page_content=text, metadata=meta)]
        
        elif ext in ['.txt', '.md', '.markdown', '.log', '.json']:
            loader = TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
        else:
            return []

        # --- 2. Process & Validate ---
        valid_docs = []
        file_meta = extract_metadata(str(file_path))

        for doc in documents:
            # Apply text cleaning
            doc.page_content = preprocess_text(doc.page_content)
            
            if validate_document(doc):
                # Update with reliable file metadata
                doc.metadata.update(file_meta)
                
                # Cleanup noisy keys often added by PDF loaders
                for k in ["producer", "creation_date", "mod_date", "total_pages"]:
                    doc.metadata.pop(k, None)
                
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return []

def load_documents(directory: Optional[Path] = None, db_dir: Optional[Path] = None, ignore_processed: bool = False) -> List[Document]:
    """
    Load documents from the data directory.
    
    Args:
        directory: Source directory for files (defaults to global RAW_FILES_DIR).
        db_dir: Database directory to store/read the processing registry.
        ignore_processed: If True, re-process all files regardless of registry.
    """
    target_dir = directory or RAW_FILES_DIR
    
    if not target_dir.exists():
        logger.error(f"Data directory does not exist: {target_dir}")
        return []

    logger.info(f"Scanning for documents in: {target_dir}")
    
    # Initialize Registry if db_dir is provided
    registry = None
    if db_dir and not ignore_processed:
        registry = RegistryManager(db_dir)
    
    # Gather Files
    supported_extensions = {'.pdf', '.txt', '.md', '.markdown', '.log', '.json'}
    all_files = [
        p for p in target_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]
    
    if not all_files:
        logger.warning("No supported files found to process.")
        return []

    # Filter files based on registry
    files_to_process = []
    if registry:
        for f in all_files:
            if not registry.is_processed(f):
                files_to_process.append(f)
        
        skipped_count = len(all_files) - len(files_to_process)
        if skipped_count > 0:
            logger.info(f"Skipping {skipped_count} already processed files.")
        
        if not files_to_process:
            logger.info("All files are up to date.")
            return []
    else:
        files_to_process = all_files

    logger.info(f"Processing {len(files_to_process)} files...")

    # Load and Process
    all_documents = []
    with tqdm(total=len(files_to_process), desc="Loading Documents", unit="file") as pbar:
        for file_path in files_to_process:
            docs = _process_single_file(file_path)
            if docs:
                all_documents.extend(docs)
                # Mark as processed immediately if successful
                if registry:
                    registry.mark_processed(file_path)
            pbar.update(1)
    
    # Persist registry updates
    if registry:
        registry.save()
    
    success_count = len(set(d.metadata['source'] for d in all_documents))
    logger.info(f"Successfully loaded {len(all_documents)} documents from {success_count} files.")
    
    return all_documents