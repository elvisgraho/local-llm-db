import re
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict
from query.database_paths import DATABASE_DIR
from training.load_documents import load_documents
from training.extract_metadata_llm import (
        DocumentMetadata,
        _get_llm_response,
        _clean_and_parse_json,
        PydanticOutputParser,
        DocumentMetadata,
        extract_text_parts, 
        get_metadata_extraction_prompt,
        Field
    )

# if python is confused about imports
# $env:PYTHONPATH = "C:\Users\user\Desktop\AI\local-llm-db\data_backend"

def main():
    parser = argparse.ArgumentParser(description="Parse folders using LangChain loader and tag files.")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("output_dir", type=Path, nargs="?", default=Path(__file__).parent / "processed", help="Output folder")
    
    args = parser.parse_args()
    
    # 1. Prepare Directories
    if not args.input_dir.exists():
        print(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Source: {args.input_dir}")
    print(f"Target: {args.output_dir}\n")

    history_file = args.output_dir / "processed_history.json"
    processed_paths = set()
    
    if history_file.exists():
        try:
            processed_paths = set(json.loads(history_file.read_text(encoding='utf-8')))
            print(f"Resuming: Found {len(processed_paths)} previously processed files.")
        except json.JSONDecodeError:
            print("WARNING: Corrupt history file. Starting fresh.")

    # ignore_processed=True forces re-processing of all files found
    try:
        documents = load_documents(directory=args.input_dir, ignore_processed=True)
    except Exception as e:
        print(f"CRITICAL: Loader failed: {e}")
        sys.exit(1)

    if not documents:
        print("No documents found or loaded.")
        sys.exit(0)

    print(f"\nProcessing {len(documents)} documents with LLM...")

    # 3. Process & Write
    try:
        for doc in documents:
            original_source = str(Path(doc.metadata.get('source', '')).resolve())
            # CHECKPOINT: Skip if already in history
            if original_source in processed_paths:
                continue

            content = doc.page_content
            original_source = doc.metadata.get('source', 'unknown')
            
            # 1. Extract and preserve existing Released date if present
            existing_date = None
            date_match = re.search(r'^Released:\s*(.+)$', content, flags=re.MULTILINE)
            if date_match:
                existing_date = date_match.group(1).strip()
                # Remove the Released line from content to avoid duplication/LLM bias
                content = re.sub(r'^Released:.*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE)

            # 2. Remove existing Tags line
            content = re.sub(r'^Tags:\s*\{.*?\}\s*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE).strip()

            if not content:
                continue

            # LLM Extraction
            meta = process_content_llm(content)
            
            if not meta:
                print(f"  [Skip] Metadata extraction failed for {Path(original_source).name}")
                continue

            # Extract Header Fields
            fname = meta.pop('suggested_filename', 'untitled_doc')
            llm_date = meta.pop('release_date', None)

            # 2. Determine final displayed date based on priority and filtering 'Unknown'.
            if existing_date:
                rdate = existing_date
            elif llm_date and llm_date != 'Unknown':
                rdate = llm_date
            else:
                rdate = None 

            # Construct Content
            # Only include "Released:" line if rdate is set to a valid, non-None value.
            date_line = f"Released: {rdate}\n" if rdate else ""

            final_data = (
                f"{date_line}"
                f"Tags: {json.dumps(meta)}\n\n"
                f"{content}"
            )

            # Write to Output
            out_path = get_unique_path(args.output_dir, fname)
            try:
                out_path.write_text(final_data, encoding='utf-8')
                print(f"  [Saved] {out_path.absolute()}")
            except Exception as e:
                print(f"  [Error] Writing {out_path.name}: {e}")

            processed_paths.add(original_source)
            try:
                # Write to disk every time to survive interrupts
                history_file.write_text(json.dumps(list(processed_paths), indent=2), encoding='utf-8')
            except Exception as e:
                print(f"  [Warning] Failed to update history file: {e}")
    except KeyboardInterrupt:                       
        print("\n\n[!] Processing stopped by user.")
        sys.exit(0)

class FileGenMetadata(DocumentMetadata):
    suggested_filename: str = Field(..., description="Snake_case filename (no extension, e.g., 'auth_bypass_v2').")
    release_date: str = Field(..., description="Release date in YYYY-MM-DD. Use 'Unknown' if not found.")

def process_content_llm(text: str) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=FileGenMetadata)
    try:
        prompt = get_metadata_extraction_prompt()
        fmt_instructions = parser.get_format_instructions()
        
        # Limit context to first 3500 chars
        prompt_val = prompt.invoke({
            "text": extract_text_parts(text), 
            "format_instructions": fmt_instructions
        })
        
        raw_response = _get_llm_response(prompt_val.to_string())
        return _clean_and_parse_json(raw_response)
    except Exception as e:
        print(f"Error during LLM processing: {e}")
        return {}
    
def get_unique_path(out_dir: Path, filename: str) -> Path:
    """Guarantees no overwrites by appending a counter."""
    safe_name = re.sub(r'[^a-z0-9_]', '', filename.lower())
    # Fallback if LLM returns empty string
    if not safe_name: safe_name = "untitled_doc"
    
    candidate = out_dir / f"{safe_name}.txt"
    counter = 1
    while candidate.exists():
        candidate = out_dir / f"{safe_name}_{counter}.txt"
        counter += 1
    return candidate

if __name__ == "__main__":
    main()