import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.processing_utils import get_unique_path
from training.load_documents import load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    get_llm_response,
    clean_and_parse_json,
    extract_text_parts, 
    get_metadata_extraction_prompt,
    PydanticOutputParser,
    DocumentMetadata,
    Field
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
    parser = argparse.ArgumentParser(description="Parse folders, extract metadata, and rename.")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path(__file__).parent / "processed", help="Output folder")
    parser.add_argument("--force", action="store_true", help="Ignore history and process everything.")
    
    args = parser.parse_args()
    
    # 1. Prepare Directories
    if not args.input_dir.exists():
        logger.error(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Initialize History
    history_path = args.output_dir / "processing_history.json"
    if args.force and history_path.exists():
        history_path.unlink()

    history = ProcessingHistory(history_path)

    print(f"--- Configuration ---")
    print(f"Source: {args.input_dir}")
    print(f"Target: {args.output_dir}")
    print(f"History: {history_path}")

    # 3. Load Documents
    logger.info("Loading document index...")
    # NOTE: We set ignore_processed=False because we handle history ourselves logic 
    # (to detect modifications, which standard loaders might miss)
    try:
        documents = load_documents(directory=args.input_dir, ignore_processed=False)
    except Exception as e:
        logger.error(f"Loader failed: {e}")
        sys.exit(1)

    if not documents:
        logger.info("No documents found.")
        sys.exit(0)

    total_docs = len(documents)
    logger.info(f"Processing {total_docs} documents...")

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    try:
        for i, doc in enumerate(documents, 1):
            source_path_str = doc.metadata.get('source', '')
            if not source_path_str:
                continue
            
            file_path = Path(source_path_str).resolve()

            # --- Check History ---
            if not history.should_process(file_path):
                stats["skipped"] += 1
                # ... [logging] ...
                continue

            print(f"[{i}/{total_docs}] Processing: {file_path.name} ...", end="\r")

            # --- Clean Content ---
            content = doc.page_content
            
            # Extract existing Released date if present (Regex improved)
            existing_date = None
            date_match = re.search(r'^Released:\s*(.+)$', content, flags=re.MULTILINE | re.IGNORECASE)
            if date_match:
                existing_date = date_match.group(1).strip()
                # Remove line
                content = re.sub(r'^Released:.*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE)

            # Remove existing Tags line
            content = re.sub(r'^Tags:\s*\{.*?\}\s*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE).strip()

            if not content:
                logger.warning(f"  [Empty] Content empty after cleaning: {file_path.name}")
                stats["errors"] += 1
                continue
            

            # --- LLM Extraction ---
            meta = process_content_llm(content)
            
            if not meta:
                logger.warning(f"  [Fail] Metadata extraction failed: {file_path.name}")
                stats["errors"] += 1
                continue

            # --- Prepare Output Data ---
            fname = meta.pop('suggested_filename', 'untitled_doc')
            llm_date = meta.pop('release_date', None)

            is_valid = meta.pop('is_technical_content', True)
            if is_valid is False:
                chapter = doc.metadata.get('chapter_title', '')
                src = doc.metadata.get('source')
                print(f"🚫 Skipping Chunk: {src} {f'[{chapter}]' if chapter else ''} (Flagged as non-technical)", flush=True)
                continue

            # Determine date logic
            if existing_date:
                rdate = existing_date
            elif llm_date and str(llm_date).lower() != 'unknown':
                rdate = llm_date
            else:
                rdate = None 

            date_line = f"Released: {rdate}\n" if rdate else ""
            
            final_data = (
                f"{date_line}"
                f"Tags: {json.dumps(meta, ensure_ascii=False)}\n\n"
                f"{content}"
            )

            # --- Write New Output ---
            out_path = get_unique_path(args.output_dir, fname)
            try:
                out_path.write_text(final_data, encoding='utf-8')
                logger.info(f"  [Saved] {out_path.name} (Source: {file_path.name})")
                # Update History
                history.record_processing(file_path, output_file=str(out_path.resolve()))
                stats["processed"] += 1
            except Exception as e:
                logger.error(f"  [Write Error] {out_path.name}: {e}")
                stats["errors"] += 1
            
            # Clean console line
            history.save()
            sys.stdout.write("\033[K")

    except KeyboardInterrupt:
        print("\n\n[!] Processing stopped by user. Saving history...")
    finally:
        history.save()

    print(f"\n--- Summary ---")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped:   {stats['skipped']}")
    print(f"Errors:    {stats['errors']}")
    print(f"Log:       metadata_processing.log")

if __name__ == "__main__":
    main()