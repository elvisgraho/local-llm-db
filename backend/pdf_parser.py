import os
import json
import requests
import logging
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import PyPDF2
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for graceful shutdown
processed_files: Set[str] = set()
should_exit = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    global should_exit
    logger.info("\nReceived interrupt signal. Saving progress and exiting gracefully...")
    should_exit = True

def save_progress(progress_file: str):
    """Save the list of processed files to a text file, one per line."""
    try:
        with open(progress_file, 'w', encoding='utf-8') as f:
            for file_path in sorted(processed_files):
                f.write(f"{file_path}\n")
        logger.info(f"Progress saved to {progress_file}")
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

def load_progress(progress_file: str) -> Set[str]:
    """Load the list of previously processed files from a text file."""
    processed = set()
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        processed.add(line)
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
    return processed

def read_pdf(file_path: str) -> str:
    """Read and extract text from a PDF file."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error reading PDF {file_path}: {e}")
        return ""

def get_llm_response(prompt: str) -> str:
    """Get response from local LLM."""
    api_url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1-distill-qwen-14b-uncensored",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        return response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling LLM API: {e}")
        return ""

def process_pdf_content(text: str, search_type: str) -> Dict[str, Any]:
    """Process PDF content using LLM to extract specific information."""
    prompt = f"""
    You are a JSON extraction assistant. Your task is to analyze the following text and extract {search_type}.
    
    Rules:
    1. Return ONLY a valid JSON object with the following structure:
       {{
         "requested_data": [
           "item1",
           "item2",
           "item3"
         ]
       }}
    2. Do not include any explanations, thinking process, or markdown formatting
    3. If nothing is found, return {{"requested_data": []}}
    4. Do not wrap the response in code blocks or markdown
    5. Use standard ASCII characters only
    6. Always maintain the exact JSON structure shown above
    7. Each item should be a simple string value

    Text to analyze:
    {text}
    """
    
    response = get_llm_response(prompt)
    if not response:
        return {"requested_data": []}
    
    # Clean the response to ensure it's valid JSON
    response = response.strip()
    
    # First try to find a JSON object in the response
    try:
        # Look for the start of a JSON object
        start_idx = response.find('{')
        if start_idx == -1:
            return {"requested_data": []}
            
        # Find the matching closing brace
        brace_count = 1
        end_idx = start_idx + 1
        while brace_count > 0 and end_idx < len(response):
            if response[end_idx] == '{':
                brace_count += 1
            elif response[end_idx] == '}':
                brace_count -= 1
            end_idx += 1
            
        if brace_count == 0:
            # Extract just the JSON part
            json_str = response[start_idx:end_idx]
            result = json.loads(json_str)
        else:
            # If we couldn't find a complete JSON object, try cleaning the whole response
            result = json.loads(response)
    except json.JSONDecodeError:
        # If that failed, try cleaning the response
        try:
            # Remove any markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1]
            elif "```" in response:
                response = response.split("```")[1]
            
            # Remove any thinking process or explanations
            if "<think>" in response:
                response = response.split("</think>")[-1]
            
            # Clean up any remaining whitespace and newlines
            response = response.strip()
            
            # Clean up any problematic characters
            response = response.replace('█', '')  # Remove block characters
            response = response.replace('…', '...')  # Replace ellipsis
            response = response.replace('…', '...')  # Replace Unicode ellipsis
            response = response.replace('…', '...')  # Replace other ellipsis variants
            
            # Remove any trailing commas in arrays
            response = response.replace(',]', ']')
            response = response.replace(',}', '}')
            
            # Remove any BOM or hidden characters
            response = response.encode('ascii', 'ignore').decode('ascii')
            
            # Try to find the last complete JSON object
            if "}" in response:
                response = response[:response.rindex("}") + 1]
            
            result = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {response[:200]}...")  # Log first 200 chars for debugging
            logger.debug(f"JSON Error details: {str(e)}")  # Log detailed error in debug mode
            return {"requested_data": []}
    
    # Validate and fix structure if needed
    if not isinstance(result, dict):
        result = {"requested_data": []}
    elif "requested_data" not in result:
        result = {"requested_data": []}
    elif not isinstance(result["requested_data"], list):
        result["requested_data"] = []
        
    # Clean up each item
    cleaned_items = []
    for item in result["requested_data"]:
        if isinstance(item, str):
            # Clean up the string
            item = item.strip()
            # Remove any spaces in the path
            item = item.replace(" ", "")
            # Remove any special characters
            item = ''.join(c for c in item if c.isprintable() and not c.isspace())
            if item:  # Only add non-empty items
                cleaned_items.append(item)
    
    result["requested_data"] = cleaned_items
    return result

def process_directory(directory: str, search_type: str, output_file: str, progress_file: str, debug: bool = False):
    """Process all PDFs in directory and subdirectories."""
    global processed_files, should_exit
    
    if debug:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory if it exists in the path
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load previously processed files
    processed_files = load_progress(progress_file)
    logger.info(f"Loaded {len(processed_files)} previously processed files")
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                if pdf_path not in processed_files:
                    pdf_files.append(pdf_path)
    
    if not pdf_files:
        logger.info("No new PDF files to process")
        return
    
    logger.info(f"Found {len(pdf_files)} new PDF files to process")
    
    # Process each PDF and write results
    processed_count = 0
    with open(output_file, 'a', encoding='utf-8') as f:  # Changed to append mode
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs", disable=debug):
            if should_exit:
                break
                
            if debug:
                logger.debug(f"Processing: {pdf_path}")
            
            # Read PDF content
            content = read_pdf(pdf_path)
            if not content:
                continue
            
            # Process content with LLM
            results = process_pdf_content(content, search_type)
            
            # Write results to file if we have any valid results
            if results and results["requested_data"]:  # Only write if we have actual data
                try:
                    processed_count += 1
                    # Write each path on a new line
                    for path in results["requested_data"]:
                        f.write(f"{path}\n")
                    f.flush()  # Ensure results are written immediately
                except Exception as e:
                    logger.error(f"Error writing results for {pdf_path}: {e}")
                    continue
            
            # Mark file as processed and save progress
            processed_files.add(pdf_path)
            save_progress(progress_file)
    
    if should_exit:
        logger.info("Processing interrupted. Progress saved.")
    else:
        logger.info(f"Processing complete. Found results in {processed_count} files. Output written to {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process PDFs to extract specific information using LLM')
    parser.add_argument('directory', help='Directory containing PDF files')
    parser.add_argument('search_type', help='Type of information to search for (e.g., "endpoints for fuzzing", "sql payloads")')
    parser.add_argument('output_file', help='Output file to write results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set up progress file path
    progress_file = f"{args.output_file}.progress.txt"
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        process_directory(args.directory, args.search_type, args.output_file, progress_file, args.debug)
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Saving progress and exiting...")
        save_progress(progress_file)
        sys.exit(0)

if __name__ == "__main__":
    main() 