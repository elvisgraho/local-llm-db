import re
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict
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

    # 2. Load Documents (Leveraging your repo_loader functions)
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
    for doc in documents:
        content = doc.page_content
        original_source = doc.metadata.get('source', 'unknown')
        
        # Skip if empty after cleaning
        if not content.strip():
            continue

        # LLM Extraction
        meta = process_content_llm(content)
        
        if not meta:
            print(f"  [Skip] Metadata extraction failed for {Path(original_source).name}")
            continue

        # Extract Header Fields
        fname = meta.pop('suggested_filename', 'untitled_doc')
        rdate = meta.pop('release_date', 'Unknown')

        # Construct Content
        final_data = (
            f"Released: {rdate}\n"
            f"Tags: {json.dumps(meta)}\n\n"
            f"{content}"
        )

        # Write to Output (Safe Path)
        out_path = get_unique_path(args.output_dir, fname)
        try:
            out_path.write_text(final_data, encoding='utf-8')
            print(f"  [Saved] {out_path.name}")
        except Exception as e:
            print(f"  [Error] Writing {out_path.name}: {e}")

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