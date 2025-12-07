import os
import json
import logging
import signal
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

# Updated Library
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

# Local Imports
from global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Global State ---
processed_files: Set[str] = set()
should_exit = False

# --- Pre-compiled Regex ---
# Matches <thinking> tags and common variations
THINKING_PATTERN = re.compile(
    r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>|'
    r'\s*\[/?(?:thinking|thought|reasoning|think)\b[^\]]*\]\s*|'
    r'\s*\((?:thinking|thought|reasoning|think)\b[^)]*\)\s*',
    flags=re.DOTALL | re.IGNORECASE
)
# Matches markdown code blocks often returned by LLMs
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global should_exit
    logger.info("\nReceived interrupt signal. Finishing current file and exiting...")
    should_exit = True

def save_progress(progress_file: Path):
    """Save the list of processed files to a text file."""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            for file_path in sorted(processed_files):
                f.write(f"{file_path}\n")
        logger.info(f"Progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

def load_progress(progress_file: Path) -> Set[str]:
    """Load the list of previously processed files."""
    processed = set()
    try:
        if progress_file.exists():
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed.add(line)
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
    return processed

def read_pdf(file_path: Path) -> str:
    """Read and extract text from a PDF file using pypdf."""
    try:
        # pypdf allows reading directly from path string or file object
        reader = PdfReader(str(file_path))
        
        # Optimization: Use list join instead of string concatenation
        text_parts = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text_parts.append(extracted)
                
        return "\n".join(text_parts)
    except PdfReadError as e:
        logger.error(f"Corrupted or invalid PDF {file_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_llm_response(prompt: str) -> str:
    """Get response from local LLM with retries."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        # Use a timeout to prevent hanging indefinitely
        response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        response_data = response.json()
        
        choices = response_data.get("choices")
        if not choices:
            return ""
            
        return choices[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API Connection Error: {e}")
        raise # Allow tenacity to retry

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extracts JSON object from raw LLM output."""
    text = text.strip()
    
    # 1. Remove Thinking Tags
    text = THINKING_PATTERN.sub('', text).strip()
    
    # 2. Check for Markdown Code Blocks
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        text = match.group(1)

    # 3. Clean common issues
    text = text.replace('█', '')
    text = text.replace('…', '...')
    # Fix trailing commas in lists/objects which are invalid JSON but common in LLM output
    text = re.sub(r',(\s*[\]}])', r'\1', text)

    # 4. Attempt Parsing
    try:
        # Find outer brackets if text contains extra chatter
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            json_str = text[start : end + 1]
            return json.loads(json_str)
        
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def process_pdf_content(text: str, search_type: str) -> Dict[str, Any]:
    """Process PDF content using LLM to extract specific information."""
    if not text.strip():
        return {"requested_data": []}

    prompt = f"""
    You are a JSON extraction assistant. Analyze the text and extract: {search_type}.
    
    Rules:
    1. Output ONLY valid JSON.
    2. Structure: {{ "requested_data": ["item1", "item2"] }}
    3. If nothing found, return {{ "requested_data": [] }}
    4. No markdown, no explanations.
    5. Clean extracted strings (remove whitespace).

    Text:
    {text[:20000]} 
    """ 
    # Truncated text slightly to avoid overflowing generic context windows if PDF is huge
    
    try:
        raw_response = get_llm_response(prompt)
    except Exception:
        return {"requested_data": []}
    
    if not raw_response:
        return {"requested_data": []}
    
    result = extract_json_from_text(raw_response)
    
    # Validation
    if not result or "requested_data" not in result or not isinstance(result["requested_data"], list):
        return {"requested_data": []}
        
    # Post-processing cleaning (Specific to user requirements)
    cleaned_items = []
    for item in result["requested_data"]:
        if isinstance(item, str):
            item = item.strip()
            # Remove spaces in path-like strings (legacy behavior preserved)
            item = item.replace(" ", "")
            # Keep printable non-space chars
            item = ''.join(c for c in item if c.isprintable() and not c.isspace())
            if item:
                cleaned_items.append(item)
    
    result["requested_data"] = cleaned_items
    return result

def process_directory(
    directory: str, 
    search_type: str, 
    output_file: str, 
    progress_file: str, 
    debug: bool = False
):
    """Process all PDFs in directory and subdirectories."""
    global processed_files
    
    if debug:
        logger.setLevel(logging.DEBUG)
    
    source_dir = Path(directory)
    output_path = Path(output_file)
    progress_path = Path(progress_file)
    
    if not source_dir.exists():
        logger.error(f"Directory not found: {directory}")
        return

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load state
    processed_files = load_progress(progress_path)
    logger.info(f"Loaded {len(processed_files)} processed files.")
    
    # Find PDFs using pathlib (recursive glob)
    # Using rglob('*') and checking suffix is case-insensitive safe
    all_files = sorted(source_dir.rglob('*'))
    pdf_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.pdf']
    
    # Filter processed
    # Normalize paths to strings for set comparison
    files_to_process = [p for p in pdf_files if str(p) not in processed_files]
    
    if not files_to_process:
        logger.info("No new PDF files to process.")
        return
    
    logger.info(f"Found {len(files_to_process)} new PDF files.")
    
    processed_count = 0
    
    # Use 'a' (append) to add to results
    with open(output_path, 'a', encoding='utf-8') as f_out:
        for pdf_path in tqdm(files_to_process, desc="Processing PDFs", disable=debug):
            if should_exit:
                break
                
            str_path = str(pdf_path)
            if debug:
                logger.debug(f"Processing: {str_path}")
            
            content = read_pdf(pdf_path)
            
            # Even if content is empty, we mark as processed to skip next time
            # (It's a valid file, just unreadable or empty)
            if content:
                results = process_pdf_content(content, search_type)
                
                if results and results.get("requested_data"):
                    try:
                        processed_count += 1
                        for item in results["requested_data"]:
                            f_out.write(f"{item}\n")
                        f_out.flush()
                    except Exception as e:
                        logger.error(f"Error writing results for {str_path}: {e}")
                        continue
            
            processed_files.add(str_path)
            save_progress(progress_path)
            
    if should_exit:
        logger.info("Processing interrupted by user.")
    else:
        logger.info(f"Complete. Extracted data from {processed_count} files to {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDFs to extract specific information using LLM')
    parser.add_argument('directory', help='Directory containing PDF files')
    parser.add_argument('search_type', help='Information to search for (e.g., "SQL payloads")')
    parser.add_argument('output_file', help='Output file to write results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    progress_file = f"{args.output_file}.progress.txt"
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        process_directory(
            args.directory, 
            args.search_type, 
            args.output_file, 
            progress_file, 
            args.debug
        )
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Exiting...")
        save_progress(Path(progress_file))
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()