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
import pypdf

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

def try_load_pdf_chapters(file_path: str, min_pages_for_split: int = 20) -> Optional[List[Document]]:
    """
    Robustly attempts to split a PDF into chapters using recursive outline parsing.
    
    Improvements over basic implementation:
    1. Handles nested bookmarks (recursive parsing).
    2. Resolves 'Named Destinations' (links that aren't direct page numbers).
    3. Merges bookmarks pointing to the same page (avoids empty documents).
    4. sanitizes titles and handles metadata inheritance.
    """
    try:
        reader = pypdf.PdfReader(file_path)
        total_pages = len(reader.pages)
        
        # 1. Basic Heuristic Checks
        if total_pages < min_pages_for_split:
            return None
        
        if not reader.outline:
            return None

        # 2. Recursive Outline Extraction
        # We want a flat list of: (PageNumber, Title, Level)
        # Level helps us distinguish "Part I" from "Chapter 1" if needed, 
        # though here we flatten for content splitting.
        
        flat_outline = []

        def _recursive_parse(outlines, level=0):
            for item in outlines:
                if isinstance(item, list):
                    _recursive_parse(item, level + 1)
                elif isinstance(item, pypdf.generic.Destination):
                    try:
                        # Robustly get page number (handles named destinations & integers)
                        page_num = reader.get_destination_page_number(item)
                        
                        # Sanity check: page_num must be valid
                        if page_num is not None and 0 <= page_num < total_pages:
                            title = item.title.strip() if item.title else "Untitled Section"
                            flat_outline.append({
                                "page": page_num,
                                "title": title,
                                "level": level
                            })
                    except Exception:
                        # If a specific bookmark is corrupted, skip it but continue processing
                        continue

        _recursive_parse(reader.outline)

        if not flat_outline:
            return None

        # 3. Smart Deduplication & Sorting
        # Sort primarily by page number
        flat_outline.sort(key=lambda x: x["page"])

        # Consolidate bookmarks that point to the same page.
        # Logic: If "Part 1" and "Chapter 1" are on Page 5, we call it "Part 1 - Chapter 1"
        # and start the chunk there.
        unique_chapters = []
        
        if flat_outline:
            current_chapter = flat_outline[0]
            
            for next_item in flat_outline[1:]:
                if next_item["page"] == current_chapter["page"]:
                    # Merge titles (e.g., "Part 1 > Chapter 1")
                    current_chapter["title"] += f" > {next_item['title']}"
                else:
                    unique_chapters.append(current_chapter)
                    current_chapter = next_item
            
            # Append the last one
            unique_chapters.append(current_chapter)

        # Heuristic: If we only found 1 chapter (and it's page 0), 
        # the outline is useless (it's just "The Book"). Return None to use default loader.
        if len(unique_chapters) < 2 and unique_chapters[0]['page'] == 0:
            return None

        # 4. Load Content (Bulk Load)
        # We load all pages once using PyPDFLoader to ensure we get LangChain compatible objects
        # This is safer than reading raw text from pypdf because PyPDFLoader handles some encoding edge cases.
        loader = PyPDFLoader(file_path)
        all_pages = loader.load()

        if len(all_pages) != total_pages:
            logger.warning(f"Page count mismatch in {file_path}. Outline: {total_pages}, Loader: {len(all_pages)}. Fallback.")
            return None

        documents = []

        # 5. Slicing logic
        for i, chapter in enumerate(unique_chapters):
            start_page = chapter["page"]
            title = chapter["title"]
            
            # Determine End Page
            if i + 1 < len(unique_chapters):
                end_page = unique_chapters[i+1]["page"]
            else:
                end_page = total_pages
            
            # Skip empty ranges (should represent error states)
            if start_page >= end_page:
                continue

            # Extract pages for this chapter
            chapter_pages = all_pages[start_page:end_page]
            
            if not chapter_pages:
                continue

            # Merge Content
            # We add double newlines to separate pages clearly
            chapter_text = "\n\n".join(p.page_content for p in chapter_pages)
            
            # Create Document with enriched metadata
            # We take the metadata from the first page of the chapter as the base
            meta = chapter_pages[0].metadata.copy()
            meta.update({
                "chapter_title": title,
                "chapter_start_index": start_page,
                "chapter_end_index": end_page - 1,
                "source_type": "book_chapter",
                "page_count": len(chapter_pages)
            })

            documents.append(Document(page_content=chapter_text, metadata=meta))

        logger.info(f"PDF Split Success: {len(documents)} chapters extracted from {file_path}")
        return documents

    except Exception as e:
        logger.warning(f"Smart PDF split failed for {file_path} (falling back to standard load): {str(e)}")
        return None

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
            # A. Attempt to split by chapter first (Logic for "Large Books")
            chapter_docs = try_load_pdf_chapters(str(file_path))
            
            if chapter_docs:
                logger.info(f"Split {file_path.name} into {len(chapter_docs)} chapters.")
                documents = chapter_docs
            else:
                # B. Fallback: Standard load (for small files or no-outline files)
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
                # Note: We use update() so we don't overwrite chapter metadata if it exists
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