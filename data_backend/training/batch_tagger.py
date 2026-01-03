import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.output_parsers import PydanticOutputParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.templates import DocumentMetadata
from training.processing_utils import calculate_context_ceiling, get_unique_path
from training.load_documents import load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    get_llm_response,
    clean_and_parse_json,
    extract_text_parts, 
    get_metadata_extraction_prompt
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metadata_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FileGenMetadata(DocumentMetadata):
    suggested_filename: str = Field(..., description="Snake_case filename (no extension, e.g., 'auth_bypass_v2'). Char limit: 50")
    release_date: str = Field(..., description="Release date in YYYY-MM-DD. Use 'Unknown' if not found.")

def process_content_llm(text: str) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=FileGenMetadata)
    try:
        prompt = get_metadata_extraction_prompt()
        fmt_instructions = parser.get_format_instructions()
        
        # Extract representative parts to stay within context window
        # (Assuming extract_text_parts handles large files intelligently)
        sampled_text = extract_text_parts(text)

        prompt_val = prompt.invoke({
            "text": sampled_text, 
            "format_instructions": fmt_instructions
        })
        
        raw_response = get_llm_response(prompt_val.to_string())
        return clean_and_parse_json(raw_response)
    except Exception as e:
        logger.error(f"LLM Processing Error: {e}")
        return {}

# -------------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Metadata extraction and document renaming.")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("output_dir", type=Path, nargs="?", 
                        default=Path(__file__).parent / "processed", help="Output folder")
    parser.add_argument("--force", action="store_true", help="Ignore history and process everything.")
    args = parser.parse_args()

    # 1. Environment Validation
    if not args.input_dir.exists():
        logger.error(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 2. History Initialization
    history_path = args.output_dir / "processing_history.json"
    if args.force and history_path.exists():
        history_path.unlink()
    history = ProcessingHistory(history_path)

    # 3. Dry Scan: Identify pending files without loading content
    print("Scanning directory for pending files...")
    supported_ext = {'.pdf', '.txt', '.md', '.markdown'}
    
    # Generate resolved absolute paths to prevent Windows pathing mismatches
    all_paths = (p.resolve() for p in args.input_dir.rglob("*") if p.is_file())
    
    pending_paths = [
        p for p in all_paths 
        if p.suffix.lower() in supported_ext and history.should_process(p)
    ]

    skipped_count = 0  # We don't have a total count yet without scanning all, but logic follows
    if not pending_paths:
        logger.info("All documents already processed. Use --force to override.")
        sys.exit(0)

    # 4. Targeted Loading: Use the file_paths parameter to load ONLY what is needed
    logger.info(f"Loading content for {len(pending_paths)} pending files...")
    try:
        filtered_documents: List[Document] = load_documents(file_paths=pending_paths)
    except Exception as e:
        logger.error(f"Loader failed: {e}")
        sys.exit(1)

    if not filtered_documents:
        logger.info("No documents were successfully loaded.")
        sys.exit(0)

    # 5. Resource Calculation (Flash-Attention Optimized)
    filtered_documents = calculate_context_ceiling(filtered_documents)

    print(f"--- Configuration ---")
    print(f"Pending Documents: {len(filtered_documents)}")
    print(f"Target: {args.output_dir}")
    print(f"Starting extraction...\n")

    # 6. Processing Loop
    stats = {"processed": 0, "errors": 0}
    try:
        for i, doc in enumerate(filtered_documents, 1):
            source_path = Path(doc.metadata.get('source', '')).resolve()
            content = doc.page_content

            # --- Technical Cleaning ---
            existing_date = None
            date_match = re.search(r'^Released:\s*(.+)$', content, flags=re.MULTILINE | re.IGNORECASE)
            if date_match:
                existing_date = date_match.group(1).strip()
                content = re.sub(r'^Released:.*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE)

            content = re.sub(r'^Tags:\s*\{.*?\}\s*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE).strip()

            if not content:
                logger.warning(f"Rejecting {source_path.name}: Empty after cleaning.")
                stats["errors"] += 1
                continue

            # --- LLM Extraction ---
            sys.stdout.write(f"[{i}/{len(filtered_documents)}] Processing: {source_path.name}\r")
            meta = process_content_llm(content)
            
            if not meta:
                logger.warning(f"Fail: Metadata extraction error for {source_path.name}")
                stats["errors"] += 1
                continue

            if not meta.pop('is_technical_content', True):
                continue

            # Date Priority Logic
            llm_date = meta.pop('release_date', None)
            rdate = existing_date or (llm_date if str(llm_date).lower() != 'unknown' else None)
            
            fname = meta.pop('suggested_filename', source_path.stem)
            date_line = f"Released: {rdate}\n" if rdate else ""
            final_data = f"{date_line}Tags: {json.dumps(meta, ensure_ascii=False)}\n\n{content}"

            # --- IO Persistence ---
            out_path = get_unique_path(args.output_dir, f"{fname}.md")
            try:
                out_path.write_text(final_data, encoding='utf-8')
                history.record_processing(source_path, output_file=str(out_path.resolve()))
                history.save()
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"Write Error: {out_path.name}: {e}")
                stats["errors"] += 1

            sys.stdout.write("\033[K")

    except KeyboardInterrupt:
        print("\n\n[!] Interrupt detected. State synchronized.")
    finally:
        history.save()

    print(f"\n--- Summary ---")
    print(f"Successfully Extracted: {stats['processed']}")
    print(f"Failures:              {stats['errors']}")

if __name__ == "__main__":
    main()