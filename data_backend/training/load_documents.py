import logging
import json
import re
import fitz  # PyMuPDF
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Modern LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# --- Local Imports ---
try:
    from query.database_paths import RAW_FILES_DIR
except ImportError:
    # Fallback for standalone testing
    RAW_FILES_DIR = Path(__file__).resolve().parent.parent / "data"

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
    Crucial for ensuring the 'Tags:' regex matches correctly later.
    """
    if not text:
        return ""

    # Remove Null Bytes (Crucial for ChromaDB/SQLite stability)
    text = text.replace('\x00', '')

    # Split text by code blocks. Odd indices will be the code content.
    parts = re.split(r'(```.*?```)', text, flags=re.DOTALL) 

    for i in range(len(parts)):
        # Only normalize parts that are NOT code blocks (even indices)
        if i % 2 == 0:
            s = parts[i]
            # Normalize Windows/Mac line endings
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            # Collapse horizontal whitespace (tabs/spaces) but keep newlines
            s = re.sub(r'[ \t]+', ' ', s)
            # Collapse excessive newlines (3+) into 2 (paragraph break)
            s = re.sub(r'\n\s*\n', '\n\n', s)
            parts[i] = s

    return "".join(parts).strip()

def validate_document(doc: Document) -> bool:
    """
    Validate document content and metadata.
    Returns False if content is just noise/empty.
    """
    if not doc.page_content:
        return False
        
    # Filter out files that are just noise (less than 10 chars)
    if len(doc.page_content.strip()) < 10:
        return False
        
    return True

# ==========================================
# 2. REGISTRY MANAGER (Incremental Loading)
# ==========================================

class RegistryManager:
    """
    Tracks processed files to support incremental indexing.
    """
    def __init__(self, db_dir: Path):
        self.registry_path = db_dir / "processed_files.json"
        self.registry: Dict[str, float] = self._load()

    def _load(self) -> Dict[str, float]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
                return {}
        return {}

    def save(self):
        try:
            with open(self.registry_path, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save registry: {e}")

    def is_processed(self, file_path: Path) -> bool:
        key = str(file_path.resolve())
        if key not in self.registry:
            return False
        
        try:
            last_mtime = self.registry[key]
            current_mtime = file_path.stat().st_mtime
            # Process only if file is newer than registry record
            return current_mtime <= last_mtime
        except FileNotFoundError:
            return False

    def mark_processed(self, file_path: Path):
        key = str(file_path.resolve())
        try:
            self.registry[key] = file_path.stat().st_mtime
        except FileNotFoundError:
            pass

# ==========================================
# 3. PDF LOGIC (Chapter Splitting)
# ==========================================

def try_load_pdf_chapters(doc: fitz.Document, filename: str, min_pages_for_split: int = 15) -> Optional[List[Document]]:
    """
    Uses PyMuPDF's outline (TOC) to split PDFs by chapter.
    Returns None if no usable outline is found.
    """
    try:
        if doc.page_count < min_pages_for_split:
            return None

        # [level, title, page_num]
        toc = doc.get_toc(simple=True) 
        if not toc:
            return None

        documents = []
        total_pages = doc.page_count
        cleaned_chapters = []

        # Deduplicate and flatten
        for entry in toc:
            lvl, title, page_num = entry[0], entry[1], entry[2]
            # PyMuPDF pages are 1-based in TOC
            if 0 < page_num <= total_pages:
                # If this chapter starts on the same page as the previous one, append title
                if cleaned_chapters and cleaned_chapters[-1]['page'] == (page_num - 1):
                    cleaned_chapters[-1]['title'] += f" > {title}"
                else:
                    cleaned_chapters.append({
                        "page": page_num - 1, # Convert to 0-based
                        "title": title.strip()
                    })

        # Heuristic: If TOC is useless (e.g., 1 chapter), abort
        if len(cleaned_chapters) < 2:
            return None

        # Extract Text per Chapter
        for i, chapter in enumerate(cleaned_chapters):
            start_page = chapter["page"]
            title = chapter["title"]
            
            if i + 1 < len(cleaned_chapters):
                end_page = cleaned_chapters[i+1]["page"]
            else:
                end_page = total_pages

            if start_page >= end_page: continue

            # Extract text from page range
            # Use sort=True to ensure multi-column PDFs (research papers) are read correctly
            chapter_text = ""
            for p in range(start_page, end_page):
                chapter_text += doc[p].get_text(sort=True) + "\n\n"

            meta = {
                "chapter_title": title,
                "source_type": "book_chapter",
                "page_count": end_page - start_page,
                "source": filename
            }
            documents.append(Document(page_content=chapter_text, metadata=meta))
            
        return documents

    except Exception as e:
        logger.warning(f"PDF Chapter split failed for {filename}: {e}")
        return None

# ==========================================
# 4. MAIN LOADING FUNCTION
# ==========================================

def _process_single_file(file_path: Path) -> List[Document]:
    """
    Loads a file, choosing the best strategy based on extension.
    """
    ext = file_path.suffix.lower()
    documents = []
    
    # 1. Calculate base metadata
    file_meta = extract_metadata(str(file_path))

    try:
        # --- A. PDF Strategy ---
        if ext == '.pdf':
            with fitz.open(file_path) as doc:
                # Try sophisticated chapter split
                documents = try_load_pdf_chapters(doc, file_path.name)
                
                # Fallback: Linear text extraction
                if not documents:
                    # [OPTIMIZATION] sort=True handles multi-column layouts (Research Papers)
                    text = "".join(page.get_text(sort=True) for page in doc)
                    
                    if text:
                        # Merge PDF metadata (Producer, Title) with caution
                        # We prefer our calculated file_meta for source/filename
                        pdf_meta = doc.metadata
                        documents = [Document(page_content=text, metadata=pdf_meta)]
                    else:
                        # [EDGE CASE] Scanned PDF Detection
                        if doc.page_count > 0:
                            logger.warning(f"⚠️  Empty text in {file_path.name}. This might be a SCANNED PDF (images only).")

        # --- B. Text/Code Strategy ---
        elif ext in ['.txt', '.md', '.markdown', '.log', '.json', '.py', '.js']:
            loader = TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
            
        else:
            return []

        # --- C. Cleanup & Validation ---
        valid_docs = []
        for doc in documents:
            # 1. Preprocess Text (Clean newlines, spaces, null bytes)
            doc.page_content = preprocess_text(doc.page_content)
            
            if validate_document(doc):
                # 2. Enforce File Metadata (Overwrite PDF specific/random metadata keys)
                doc.metadata.update(file_meta)
                
                # 3. Clean Garbage Keys
                for k in ["producer", "creationDate", "modDate", "total_pages", "format", "encryption"]:
                    doc.metadata.pop(k, None)
                    
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return []

def load_documents(directory: Optional[Path] = None, db_dir: Optional[Path] = None, ignore_processed: bool = False) -> List[Document]:
    """
    Main entry point. Scans directory, checks registry, loads files.
    """
    target_dir = directory or RAW_FILES_DIR
    
    if not target_dir.exists():
        logger.error(f"Data directory does not exist: {target_dir}")
        return []

    logger.info(f"Scanning for documents in: {target_dir}")
    
    # Init Registry
    registry = None
    if db_dir and not ignore_processed:
        registry = RegistryManager(db_dir)
    
    # Find Files
    supported_extensions = {'.pdf', '.txt', '.md', '.markdown', '.log', '.json', '.py'}
    all_files = [
        p for p in target_dir.rglob("*") 
        if p.is_file() and p.suffix.lower() in supported_extensions
    ]
    
    if not all_files:
        logger.warning("No supported files found.")
        return []

    # Filter
    files_to_process = []
    if registry:
        for f in all_files:
            if not registry.is_processed(f):
                files_to_process.append(f)
        
        skipped = len(all_files) - len(files_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files.")
    else:
        files_to_process = all_files

    if not files_to_process:
        logger.info("All files are up to date.")
        return []

    logger.info(f"Processing {len(files_to_process)} files...")

    # Load Loop
    all_documents = []
    SAVE_INTERVAL = 50 

    with tqdm(total=len(files_to_process), desc="Loading Documents", unit="file") as pbar:
        for i, file_path in enumerate(files_to_process):
            docs = _process_single_file(file_path)
            if docs:
                all_documents.extend(docs)
                if registry:
                    registry.mark_processed(file_path)
            
            if registry and i % SAVE_INTERVAL == 0:
                registry.save()
                
            pbar.update(1)
    
    if registry:
        registry.save()
    
    # Summary
    success_count = len(set(d.metadata['source'] for d in all_documents))
    logger.info(f"Successfully loaded {len(all_documents)} documents from {success_count} files.")
    
    return all_documents