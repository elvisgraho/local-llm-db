import logging
import json
import re
import fitz  # PyMuPDF
import base64
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Modern LangChain Imports ---
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document

# --- Local Imports ---
try:
    from common import RAW_FILES_DIR
except ImportError:
    # Fallback for standalone testing
    RAW_FILES_DIR = Path(__file__).resolve().parent.parent / "volumes" / "raw_files"

logger = logging.getLogger(__name__)

# ==========================================
# 0. CONSTANTS & COMPILED REGEX
# ==========================================

# 4x Zoom = ~288 DPI. Defined once to avoid re-instantiation in loops.
ZOOM_MATRIX = fitz.Matrix(4, 4)

# Regex for splitting text while preserving code blocks and specific tags
# Dotall flag is handled via definition or passed during use, but compiling 
# with (?s) inside the string or flags=re.DOTALL works best.
RE_SPLIT_BLOCKS = re.compile(r'(```.*?```|<<IMAGE_OCR_DATA:.*?>>)', flags=re.DOTALL)
RE_WS_COLLAPSE = re.compile(r'[ \t]+')
RE_NEWLINE_COLLAPSE = re.compile(r'\n\s*\n')

# Keys to remove from PDF metadata
METADATA_KEYS_TO_REMOVE = {
    "producer", "creationDate", "modDate", "total_pages", "format", "encryption"
}

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract basic metadata from a file path using pathlib.
    """
    try:
        # Use Path object directly if it allows, otherwise instantiate
        path = Path(file_path)
        return {
            "source": str(path.resolve()),
            "file_name": path.name,
            "file_extension": path.suffix.lower(),
            "file_type": path.suffix[1:].upper() if path.suffix else "UNKNOWN"
        }
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
        return {
            "source": str(file_path),
            "file_name": "unknown",
            "file_type": "UNKNOWN"
        }

def preprocess_text(text: str) -> str:
    """
    Clean and normalize text while preserving:
    1. Code blocks (```...```)
    2. Image OCR Data Tags (<<IMAGE_OCR_DATA:...>>)
    """
    if not text:
        return ""

    # Remove Null Bytes (Crucial for ChromaDB/SQLite stability)
    if '\x00' in text:
        text = text.replace('\x00', '')

    # Optimization: If no special blocks exist, skip the split logic
    if "```" not in text and "<<IMAGE_OCR_DATA" not in text:
        # Fast path
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        text = RE_WS_COLLAPSE.sub(' ', text)
        text = RE_NEWLINE_COLLAPSE.sub('\n\n', text)
        return text.strip()

    parts = RE_SPLIT_BLOCKS.split(text)

    # Use a list comprehension for slightly better performance than loop
    processed_parts = []
    for i, part in enumerate(parts):
        # Only normalize parts that are NOT special blocks (even indices)
        if i % 2 == 0:
            if not part: continue # skip empty strings from split
            s = part.replace('\r\n', '\n').replace('\r', '\n')
            s = RE_WS_COLLAPSE.sub(' ', s)
            s = RE_NEWLINE_COLLAPSE.sub('\n\n', s)
            processed_parts.append(s)
        else:
            # Special blocks: ensure breathing room
            p = part
            if not p.startswith("\n"): p = "\n" + p
            if not p.endswith("\n"): p = p + "\n"
            processed_parts.append(p)

    return "".join(processed_parts).strip()

def validate_document(doc: Document) -> bool:
    """
    Validate document content and metadata.
    Returns False if content is just noise/empty.
    """
    content = doc.page_content
    if not content:
        return False
        
    # [CRITICAL UPDATE] If doc contains Image Data, it is valid even if text is short
    if "<<IMAGE_OCR_DATA:" in content:
        return True

    # Filter out files that are just noise (less than 10 chars)
    if len(content.strip()) < 10:
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
        # Optimization: use str(resolve()) once
        key = str(file_path.resolve())
        last_mtime = self.registry.get(key)
        
        if last_mtime is None:
            return False
            
        try:
            current_mtime = file_path.stat().st_mtime
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
# 3. PDF LOGIC (Layout Aware + In-Memory Image Optimization)
# ==========================================

def _optimize_and_encode_image(pix: fitz.Pixmap) -> Optional[str]:
    """
    Converts a PyMuPDF Pixmap into a Base64 encoded PNG string.
    """
    try:
        # Optimization: check criteria before creating new Pixmap
        # If already RGB (n=3) and no alpha, we can skip conversion (rare but possible)
        needs_conversion = (pix.n >= 4 or pix.n == 1 or pix.alpha)
        
        if needs_conversion:
            # Create a white background canvas to flatten alpha
            # Use fitz.csRGB directly
            pix_clean = fitz.Pixmap(fitz.csRGB, pix) 
            if pix.alpha:
                pix_clean.set_rect(pix_clean.irect)
            image_bytes = pix_clean.tobytes("png")
            pix_clean = None # hint for GC
        else:
            image_bytes = pix.tobytes("png")

        return base64.b64encode(image_bytes).decode('utf-8')

    except Exception as e:
        logger.warning(f"Image optimization failed: {e}")
        return None

def _extract_page_content_interleaved(page: fitz.Page, is_ocr_enabled: bool = False) -> str:
    """
    Extracts text and images by manually discovering image locations 
    and merging them with text blocks based on vertical position.
    """
    # 1. Get Text Blocks ONLY
    raw_blocks = page.get_text("blocks")
    
    # Filter to keep ONLY Text blocks (type=0)
    final_blocks = [list(b) for b in raw_blocks if b[6] == 0]

    # 2. Manual Image Discovery
    if is_ocr_enabled:
        # get_images(full=True) is fast, returns metadata
        image_list = page.get_images(full=True)
        
        if image_list:
            for img in image_list:
                xref = img[0]
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    continue

                for bbox in rects:
                    # Filter out tiny elements early
                    if bbox.width < 50 or bbox.height < 50:
                        continue
                        
                    try:
                        # Use global constant ZOOM_MATRIX
                        pix = page.get_pixmap(clip=bbox, matrix=ZOOM_MATRIX)
                        
                        # Secondary Size Check
                        if pix.width < 100 or pix.height < 100:
                            continue

                        base64_img = _optimize_and_encode_image(pix)
                        if not base64_img: 
                            continue
                        
                        # [x0, y0, x1, y1, CONTENT, block_no, type=1]
                        final_blocks.append([bbox.x0, bbox.y0, bbox.x1, bbox.y1, base64_img, None, 1])

                    except Exception as e:
                        logger.warning(f"Image extraction failed on page {page.number}: {e}")

    # 3. Sort by Reading Order (Top -> Left)
    if not final_blocks:
        return ""
        
    final_blocks.sort(key=lambda b: (b[1], b[0]))

    # 4. Generate Output
    # Use list accumulation for speed
    page_content = []
    for b in final_blocks:
        if b[6] == 0: # Text
            text = b[4]
            if text.strip():
                page_content.append(text)
        elif b[6] == 1: # Image
            # Add newlines to ensure the tag is distinct
            page_content.append(f"\n<<IMAGE_OCR_DATA: {b[4]}>>\n")

    return "\n".join(page_content)

def try_load_pdf_chapters(doc: fitz.Document, filename: str, min_pages_for_split: int = 15, is_ocr_enabled: bool = False) -> Optional[List[Document]]:
    """
    Uses PyMuPDF's outline (TOC) to split PDFs by chapter.
    """
    try:
        if doc.page_count < min_pages_for_split:
            return None

        toc = doc.get_toc(simple=True) 
        if not toc:
            return None

        documents = []
        total_pages = doc.page_count
        cleaned_chapters = []

        # Deduplicate and flatten
        for entry in toc:
            lvl, title, page_num = entry[0], entry[1], entry[2]
            if 0 < page_num <= total_pages:
                if cleaned_chapters and cleaned_chapters[-1]['page'] == (page_num - 1):
                    cleaned_chapters[-1]['title'] += f" > {title}"
                else:
                    cleaned_chapters.append({
                        "page": page_num - 1, 
                        "title": title.strip()
                    })

        if len(cleaned_chapters) < 2:
            return None

        # Extract Text + Images per Chapter
        for i, chapter in enumerate(cleaned_chapters):
            start_page = chapter["page"]
            title = chapter["title"]
            
            # Determine end page
            if i + 1 < len(cleaned_chapters):
                end_page = cleaned_chapters[i+1]["page"]
            else:
                end_page = total_pages

            if start_page >= end_page: continue

            # Efficient Content Extraction
            chapter_parts = []
            for p_idx in range(start_page, end_page):
                chapter_parts.append(_extract_page_content_interleaved(doc[p_idx], is_ocr_enabled))
            
            # Join with double newlines
            chapter_content = "\n\n".join(chapter_parts)

            meta = {
                "chapter_title": title,
                "source_type": "book_chapter",
                "page_count": end_page - start_page,
                "source": filename
            }
            documents.append(Document(page_content=chapter_content, metadata=meta))
            
        return documents

    except Exception as e:
        logger.warning(f"PDF Chapter split failed for {filename}: {e}")
        return None

# ==========================================
# 4. MAIN LOADING FUNCTION
# ==========================================

def _process_single_file(file_path: Path, is_ocr_enabled: bool = False) -> List[Document]:
    """
    Loads a file, choosing the best strategy based on extension.
    """
    ext = file_path.suffix.lower()
    documents = []
    
    # 1. Calculate base metadata once
    file_meta = extract_metadata(str(file_path))

    try:
        # --- A. PDF Strategy ---
        if ext == '.pdf':
            with fitz.open(file_path) as doc:
                # 1. Try Chapter Split
                documents = try_load_pdf_chapters(doc, file_path.name, is_ocr_enabled=is_ocr_enabled)
                
                # 2. Fallback: Linear text extraction
                if not documents:
                    content_parts = []
                    for page in doc:
                        content_parts.append(_extract_page_content_interleaved(page, is_ocr_enabled))
                    
                    full_content = "\n\n".join(content_parts)
                    
                    # Check if we got anything
                    if full_content.strip() or "<<IMAGE_OCR_DATA" in full_content:
                        pdf_meta = doc.metadata
                        documents = [Document(page_content=full_content, metadata=pdf_meta)]
                    else:
                        if doc.page_count > 0:
                            logger.warning(f"⚠️  Empty content in {file_path.name}.")

        # --- B. Text/Code Strategy ---
        elif ext in {'.txt', '.md', '.markdown', '.log', '.json', '.py', '.js'}:
            loader = TextLoader(str(file_path), encoding='utf-8', autodetect_encoding=True)
            documents = loader.load()
            
        else:
            return []

        # --- C. Cleanup & Validation ---
        valid_docs = []
        for doc in documents:
            # 1. Preprocess Text
            doc.page_content = preprocess_text(doc.page_content)
            
            if validate_document(doc):
                # 2. Enforce File Metadata 
                doc.metadata.update(file_meta)
                
                # 3. Clean Garbage Keys efficiently
                for k in METADATA_KEYS_TO_REMOVE:
                    if k in doc.metadata:
                        del doc.metadata[k]
                    
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return []

def load_documents(
    directory: Optional[Path] = None, 
    db_dir: Optional[Path] = None, 
    ignore_processed: bool = False,
    is_ocr_enabled: bool = False,
    file_paths: Optional[List[Path]] = None
) -> List[Document]:
    """
    Main entry point.
    """
    # 1. Source Determination
    if file_paths is not None:
        all_files = [p for p in file_paths if p.is_file()]
        logger.info(f"Using {len(all_files)} explicitly provided file paths.")
    else:
        target_dir = directory or RAW_FILES_DIR
        if not target_dir.exists():
            logger.error(f"Data directory does not exist: {target_dir}")
            return []
        
        logger.info(f"Scanning for documents in: {target_dir}")
        supported_extensions = {'.pdf', '.txt', '.md', '.markdown'}
        all_files = [
            p for p in target_dir.rglob("*") 
            if p.is_file() and p.suffix.lower() in supported_extensions
        ]

    if not all_files:
        logger.warning("No valid files found for processing.")
        return []

    # 2. Init Registry
    registry = None
    if db_dir and not ignore_processed:
        registry = RegistryManager(db_dir)
    
    # 3. Filter via Registry
    # Use list comprehension for speed
    if registry:
        files_to_process = [f for f in all_files if not registry.is_processed(f)]
        
        skipped = len(all_files) - len(files_to_process)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already processed files.")
    else:
        files_to_process = all_files

    if not files_to_process:
        logger.info("All files are up to date.")
        return []

    # 4. Load Loop
    logger.info(f"Processing {len(files_to_process)} files...")
    all_documents = []
    SAVE_INTERVAL = 50 

    # Optimize progress bar: avoid updating manually inside loop if possible, 
    # but the logic requires manual updates for the logic flow.
    with tqdm(total=len(files_to_process), desc="Loading Documents", unit="file") as pbar:
        for i, file_path in enumerate(files_to_process):
            docs = _process_single_file(file_path, is_ocr_enabled)
            if docs:
                all_documents.extend(docs)
                if registry:
                    registry.mark_processed(file_path)
            
            # Note: Saving entire JSON every 50 files is expensive for large sets.
            # Kept as per requirements, but consider raising interval for production.
            if registry and i > 0 and i % SAVE_INTERVAL == 0:
                registry.save()
            pbar.update(1)
    
    if registry:
        registry.save()
    
    # 5. Summary
    # Efficient set creation
    sources = {d.metadata.get('source') for d in all_documents}
    logger.info(f"Successfully loaded {len(all_documents)} document chunks from {len(sources)} files.")
    
    return all_documents
