import logging
import json
import re
import fitz
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional
from query.database_paths import RAW_FILES_DIR

# --- Modern LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

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
    Clean and normalize text while preserving code blocks (```...```).
    """
    if not text:
        return ""

    # Split text by code blocks. Odd indices will be the code content.
    # non-greedy match for code blocks
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL) 

    for i in range(len(parts)):
        # Only normalize parts that are NOT code blocks (even indices)
        if i % 2 == 0:
            # Normalize line endings
            s = parts[i].replace('\r\n', '\n').replace('\r', '\n')
            # Collapse horizontal whitespace
            s = re.sub(r'[ \t]+', ' ', s)
            # Collapse multiple newlines into paragraph break
            s = re.sub(r'\n\s*\n', '\n\n', s)
            parts[i] = s

    return "".join(parts).strip()

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
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}

    def save(self):
        """Save registry to disk."""
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
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
    Optimized loader using PyMuPDF for high-performance PDF parsing.
    """
    ext = file_path.suffix.lower()
    documents = []

    try:
        # --- 1. Load Raw Content ---
        if ext == '.pdf':
            with fitz.open(file_path) as doc:
                # A. Attempt split by chapter (Zero-copy pass)
                chapter_docs = try_load_pdf_chapters(doc, file_path.name)
                
                if chapter_docs:
                    logger.info(f"Split {file_path.name} into {len(chapter_docs)} chapters.")
                    documents = chapter_docs
                else:
                    # B. Fallback: Fast linear load (C-speed text extraction)
                    text = "".join(page.get_text() for page in doc)
                    if text:
                        meta = extract_metadata(str(file_path))
                        meta.update(doc.metadata)
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
            doc.page_content = preprocess_text(doc.page_content)
            
            if validate_document(doc):
                doc.metadata.update(file_meta)
                # Cleanup PyMuPDF/Loader metadata keys
                for k in ["producer", "creationDate", "modDate", "total_pages", "format", "encryption"]:
                    doc.metadata.pop(k, None)
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return []

def try_load_pdf_chapters(doc: fitz.Document, filename: str, min_pages_for_split: int = 20) -> Optional[List[Document]]:
    """
    Uses PyMuPDF's native get_toc() for O(1) outline extraction.
    """
    try:
        if doc.page_count < min_pages_for_split:
            return None

        # get_toc(simple=True) returns [lvl, title, page_num, ...]
        # page_num is 1-based in PyMuPDF
        toc = doc.get_toc(simple=True)
        
        if not toc:
            return None

        documents = []
        total_pages = doc.page_count

        # Filter and Deduplicate Logic
        # We only care about the page flow, so we flatten nested structures naturally
        cleaned_chapters = []
        for entry in toc:
            lvl, title, page_num = entry[0], entry[1], entry[2]
            if page_num > 0 and page_num <= total_pages:
                 # Deduplicate: if same page as last chapter, append title
                if cleaned_chapters and cleaned_chapters[-1]['page'] == (page_num - 1):
                    cleaned_chapters[-1]['title'] += f" > {title}"
                else:
                    cleaned_chapters.append({
                        "page": page_num - 1, # Convert to 0-based
                        "title": title.strip()
                    })

        # Heuristic: If useless outline (only 1 chapter at page 0), abort
        if len(cleaned_chapters) < 2 and cleaned_chapters[0]['page'] == 0:
            return None

        # Extraction Loop
        for i, chapter in enumerate(cleaned_chapters):
            start_page = chapter["page"]
            title = chapter["title"]
            
            # Determine end page
            if i + 1 < len(cleaned_chapters):
                end_page = cleaned_chapters[i+1]["page"]
            else:
                end_page = total_pages

            if start_page >= end_page: 
                continue

            # Fast Text Extraction
            # Use chr(12) (Form Feed) as page delimiter if strictly needed, or \n\n
            chapter_text = "\n\n".join(doc[p].get_text() for p in range(start_page, end_page))

            meta = {
                "chapter_title": title,
                "chapter_start_index": start_page,
                "chapter_end_index": end_page - 1,
                "source_type": "book_chapter",
                "page_count": end_page - start_page,
                "source": filename
            }
            
            documents.append(Document(page_content=chapter_text, metadata=meta))

        return documents

    except Exception as e:
        logger.warning(f"Fast PDF split failed for {filename}: {str(e)}")
        return None
    
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