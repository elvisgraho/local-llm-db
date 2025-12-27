import logging
import json
import re
import fitz  # PyMuPDF
import tiktoken
import base64
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
    Clean and normalize text while preserving:
    1. Code blocks (```...```)
    2. Image OCR Data Tags (<<IMAGE_OCR_DATA:...>>)
    """
    if not text:
        return ""

    # Remove Null Bytes (Crucial for ChromaDB/SQLite stability)
    text = text.replace('\x00', '')

    # Split text by code blocks OR our specific image tags. 
    # This prevents the cleanup logic from mangling the Base64 strings.
    # We use non-greedy matching (.*?) to handle multiple tags/blocks.
    parts = re.split(r'(```.*?```|<<IMAGE_OCR_DATA:.*?>>)', text, flags=re.DOTALL) 

    for i in range(len(parts)):
        # Only normalize parts that are NOT special blocks (even indices)
        if i % 2 == 0:
            s = parts[i]
            # Normalize Windows/Mac line endings
            s = s.replace('\r\n', '\n').replace('\r', '\n')
            # Collapse horizontal whitespace (tabs/spaces) but keep newlines
            s = re.sub(r'[ \t]+', ' ', s)
            # Collapse excessive newlines (3+) into 2 (paragraph break)
            s = re.sub(r'\n\s*\n', '\n\n', s)
            parts[i] = s
        else:
            # OPTIONAL: Ensure special blocks have breathing room (newlines)
            # This helps the LLM/OCR pipeline distinguish the start/end clearly
            if not parts[i].startswith("\n"):
                parts[i] = "\n" + parts[i]
            if not parts[i].endswith("\n"):
                parts[i] = parts[i] + "\n"

    return "".join(parts).strip()

def validate_document(doc: Document) -> bool:
    """
    Validate document content and metadata.
    Returns False if content is just noise/empty.
    """
    if not doc.page_content:
        return False
        
    # [CRITICAL UPDATE] If doc contains Image Data, it is valid even if text is short
    if "<<IMAGE_OCR_DATA:" in doc.page_content:
        return True

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
# 3. PDF LOGIC (Layout Aware + In-Memory Image Optimization)
# ==========================================

def _optimize_and_encode_image(pix: fitz.Pixmap) -> Optional[str]:
    """
    Converts a PyMuPDF Pixmap into a Base64 encoded PNG string.
    Performs OCR optimization:
    1. Converts CMYK/Gray to RGB.
    2. Flattens Alpha channels (transparency) to White background.
    """
    try:
        # 1. OCR Optimization: Handle Color Spaces & Transparency
        # If CMYK (n=4) or Gray (n=1) or has Alpha (transparency)
        if pix.n >= 4 or pix.n == 1 or pix.alpha:
            # Create a white background canvas
            # This prevents transparent text from becoming black-on-black
            if pix.alpha:
                # Create RGB pixmap
                pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                # Composite onto white background (default is white for clear_with)
                # But PyMuPDF < 1.19 requires explicit background handling or simple drop
                # Simple approach: Drop alpha, but convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix) 
                pix.set_rect(pix.irect) # Fixes potential geometry issues after conversion
            else:
                # Just convert CMYK/Gray to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)

        # 2. Get bytes as PNG
        image_bytes = pix.tobytes("png")

        # 3. Filter Noise (REMOVED Byte-size check)
        # We now rely solely on the dimension check in the extraction function.
        # Allowing small file sizes handles simple charts/graphs correctly.
        
        # 4. Encode to Base64
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return base64_str

    except Exception as e:
        logger.warning(f"Image optimization failed: {e}")
        return None

def _extract_page_content_interleaved(page: fitz.Page) -> str:
    """
    Extracts text and images by manually discovering image locations 
    and merging them with text blocks based on vertical position.
    This fixes issues where get_text("blocks") misses images in complex PDFs.
    """
    # 1. Get Text Blocks ONLY
    # We use raw=False, sort=False because we will sort manually later.
    raw_blocks = page.get_text("blocks")
    
    # Filter to keep ONLY Text blocks (type=0)
    # We discard PyMuPDF's auto-detected images (type=1) because they are often incomplete.
    final_blocks = [list(b) for b in raw_blocks if b[6] == 0]

    # 2. Manual Image Discovery (The Root Cause Fix)
    # This finds images even if they are inside Form XObjects or masked containers
    image_list = page.get_images(full=True)
    
    for img in image_list:
        xref = img[0]
        try:
            # Find every location where this image appears on the page
            rects = page.get_image_rects(xref)
        except Exception:
            continue

        for bbox in rects:
            # --- NOISE FILTER ---
            # Filter out tiny elements (icons, lines, invisible spacers)
            if bbox.width < 50 or bbox.height < 50:
                continue
                
            try:
                # --- ZOOM FOR OCR (Critical for Code/Terminal) ---
                # 4x Zoom = ~288 DPI. Essential for distinguishing '.' from ','
                zoom_matrix = fitz.Matrix(4, 4)
                pix = page.get_pixmap(clip=bbox, matrix=zoom_matrix)
                
                # Secondary Size Check (post-zoom)
                # A 50x50 icon becomes 200x200. If it's still small, it's definitely noise.
                if pix.width < 100 or pix.height < 100:
                    continue

                # Optimize & Encode
                base64_img = _optimize_and_encode_image(pix)
                if not base64_img: 
                    continue
                
                # Create a "Block" that matches PyMuPDF's structure
                # Format: [x0, y0, x1, y1, CONTENT, block_no, type]
                # We use type=1 to indicate Image
                image_block = [bbox.x0, bbox.y0, bbox.x1, bbox.y1, base64_img, None, 1]
                final_blocks.append(image_block)

            except Exception as e:
                logger.warning(f"Image extraction failed on page {page.number}: {e}")

    # 3. Sort by Reading Order
    # Sort primarily by Top position (y0), then Left position (x0)
    # This interleaves the images correctly into the text flow.
    final_blocks.sort(key=lambda b: (b[1], b[0]))

    # 4. Generate Output
    page_content = []
    for b in final_blocks:
        if b[6] == 0: # Text Block
            text = b[4]
            if text.strip():
                page_content.append(text)
        elif b[6] == 1: # Image Block
            base64_str = b[4]
            # Add newlines to ensure the tag is distinct for the OCR pipeline
            tag = f"\n<<IMAGE_OCR_DATA: {base64_str}>>\n"
            page_content.append(tag)

    return "\n".join(page_content)

def try_load_pdf_chapters(doc: fitz.Document, filename: str, min_pages_for_split: int = 15) -> Optional[List[Document]]:
    """
    Uses PyMuPDF's outline (TOC) to split PDFs by chapter.
    Returns None if no usable outline is found.
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
            
            if i + 1 < len(cleaned_chapters):
                end_page = cleaned_chapters[i+1]["page"]
            else:
                end_page = total_pages

            if start_page >= end_page: continue

            # Extract content (Text + Base64 Images)
            chapter_content = ""
            for p_idx in range(start_page, end_page):
                page_text = _extract_page_content_interleaved(doc[p_idx])
                chapter_content += page_text + "\n\n"

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
                # 1. Try Chapter Split (with Image Embedding)
                documents = try_load_pdf_chapters(doc, file_path.name)
                
                # 2. Fallback: Linear text extraction (with Image Embedding)
                if not documents:
                    full_content = ""
                    for page in doc:
                        full_content += _extract_page_content_interleaved(page) + "\n\n"
                    
                    # Check if we got anything (Text OR Image Tags)
                    if full_content.strip() or "<<IMAGE_OCR_DATA" in full_content:
                        pdf_meta = doc.metadata
                        documents = [Document(page_content=full_content, metadata=pdf_meta)]
                    else:
                        # Scanned PDF with no OCR-able text and images failed filter
                        if doc.page_count > 0:
                            logger.warning(f"⚠️  Empty content in {file_path.name}. (Possibly Scanned PDF with poor quality images)")

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
            # Safe for Base64 tags due to updated Regex
            doc.page_content = preprocess_text(doc.page_content)
            
            if validate_document(doc):
                # 2. Enforce File Metadata 
                doc.metadata.update(file_meta)
                
                # 3. Clean Garbage Keys
                for k in ["producer", "creationDate", "modDate", "total_pages", "format", "encryption"]:
                    doc.metadata.pop(k, None)
                    
                valid_docs.append(doc)

        return valid_docs

    except Exception as e:
        logger.error(f"Error loading {file_path.name}: {e}")
        return []

def load_documents(
    directory: Optional[Path] = None, 
    db_dir: Optional[Path] = None, 
    ignore_processed: bool = False,
    file_paths: Optional[List[Path]] = None
) -> List[Document]:
    """
    Main entry point. Priority:
    1. If file_paths is provided, use them directly.
    2. Otherwise, scan directory (or RAW_FILES_DIR).
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

    # 4. Load Loop
    logger.info(f"Processing {len(files_to_process)} files...")
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
    
    # 5. Summary
    sources = {d.metadata.get('source') for d in all_documents if 'source' in d.metadata}
    logger.info(f"Successfully loaded {len(all_documents)} document chunks from {len(sources)} files.")
    
    return all_documents

def calculate_context_ceiling(documents: List[Document], system_prompt_len: int = 2000) -> List[Document]:
    if not documents:
        return []

    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        encoding = None

    # Sort by token count if possible, else fallback to character length
    if encoding:
        # Pre-calculating tokens prevents O(n log n) encoding calls during sort
        doc_data = []
        for d in documents:
            # IMPORTANT: For calculation, we likely don't want to count the raw Base64 
            # as massive tokens if we plan to strip it before LLM context context. 
            # However, for now, we count it as is.
            tokens = len(encoding.encode(d.page_content))
            doc_data.append((tokens, d))
        
        # Sort descending by token count
        doc_data.sort(key=lambda x: x[0], reverse=True)
        sorted_docs = [x[1] for x in doc_data]
        peak_tokens = doc_data[0][0]
    else:
        # Fallback to characters if tiktoken fails
        sorted_docs = sorted(documents, key=lambda d: len(d.page_content), reverse=True)
        peak_tokens = int(len(sorted_docs[0].page_content) / 2.3)

    # Calculate Ceiling using the verified peak_tokens
    sys_tokens = (len(encoding.encode("a" * system_prompt_len)) if encoding 
                  else int(system_prompt_len / 2.3))
    
    # 2560 buffer + 1.15x margin for KV cache overhead
    raw_ceiling = int((peak_tokens + sys_tokens + 2560) * 1.15)
    optimized_ceiling = 1 << (raw_ceiling - 1).bit_length()

    logger.info(f"Heaviest Source: {sorted_docs[0].metadata.get('source', 'unknown')}")
    logger.info(f"Peak Tokens: {peak_tokens}")
    logger.info(f"Allocated Context Ceiling: {optimized_ceiling}")
    
    return sorted_docs