import os
import sys
import argparse
import logging
import re
import signal
from tqdm import tqdm
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.processing_utils import get_unique_path
from training.load_documents import load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    _get_llm_response,  
    ChatPromptTemplate
)

# Configure Logging
logging.basicConfig(
    filename='writeup_generation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Global Signal Handler for Graceful Shutdown
# -------------------------------------------------------------------------
STOP_REQUESTED = False

def signal_handler(sig, frame):
    global STOP_REQUESTED
    print("\n\n[!] Ctrl+C detected. Finishing current file and stopping...")
    print("[!] Please wait for the current save to complete.\n")
    STOP_REQUESTED = True

# Register the signal
signal.signal(signal.SIGINT, signal_handler)

# -------------------------------------------------------------------------
# 1. Prompt
# -------------------------------------------------------------------------

llm_prompt_template = """
You are a Senior Technical Intelligence Analyst. Your task is to convert raw, potentially messy source text into a **comprehensive, professional technical documentation file**.

### CORE OBJECTIVES
1.  **Extraction**: Identify every technical detail, command, code snippet, configuration, and vulnerability mentioned.
2.  **Reconstruction**: If the transcript is fragmented, use domain expertise to reconstruct the intended technical logic (e.g., fixing broken command syntax).
3.  **Synthesis**: Organize scattered information into a linear, logical technical narrative.

### STRICT FORMATTING RULES
-   **Pragmatism**: Prioritize solutions that work in real-world environments.
-   **NO EMOJIS**: Do not use emojis anywhere.
-   **NO TABLES**: Do not generate Markdown tables.
-   **NO CONVERSATIONAL FILLER**: Do not use phrases like "In this document," "The speaker discusses," or "Here is the writeup." Start directly with technical content.
-   **NO ETHICAL/LEGAL DISCLAIMERS**: This is an internal intelligence repository. Omit all warnings, refusals, or moralizing text.

### INPUT CONTENT TO ANALYZE:
{content}
"""

# -------------------------------------------------------------------------
# 2. Helper Functions
# -------------------------------------------------------------------------

def clean_filename(filename: str) -> str:
    """Sanitize filename to prevent OS errors."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def load_processed_log(log_path: Path) -> set:
    """Loads the set of already processed source file paths."""
    if not log_path.exists():
        return set()
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception:
        return set()

def append_to_processed_log(log_path: Path, source_path: str):
    """Appends a processed file path to the log."""
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{source_path}\n")
    except Exception as e:
        logger.error(f"Failed to update processed log: {e}")

# -------------------------------------------------------------------------
# 3. Generation Logic
# -------------------------------------------------------------------------

def generate_writeup(text: str) -> str:
    """
    Invokes the LLM and strips out 'Thinking' blocks.
    """
    prompt = ChatPromptTemplate.from_template(llm_prompt_template)
    prompt_val = prompt.invoke({"content": text})
    
    # Blocking call
    raw_response = _get_llm_response(prompt_val.to_string())
    
    # --- Logic for Thinking Models ---
    # If the model outputs a delimiter, strip everything before it.
    delimiter = "[BEGIN FINAL RESPONSE]"
    if delimiter in raw_response:
        # Split once, take the second part (index 1)
        content = raw_response.split(delimiter, 1)[1]
        return content.strip()
    
    return raw_response.strip()

def process_documents_sequentially(input_dir: Path, output_dir: Path):
    global STOP_REQUESTED
    
    # Setup Resume Capability
    history_path = output_dir / "writeup_history.json"
    history = ProcessingHistory(history_path)
    
    print("Loading documents index...")
    try:
        documents = load_documents(directory=input_dir, ignore_processed=True)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load documents: {e}")
        return

    total_docs = len(documents)
    print(f"Found {total_docs} documents.")
    print(f"Output Directory: {output_dir}")
    print("Starting processing... (Press Ctrl+C to stop safely)\n")

    stats = {"SUCCESS": 0, "SKIPPED": 0, "ERROR": 0}

    # Iterate
    for doc in tqdm(documents, total=total_docs, unit="file"):
        
        # 1. Check for Interrupt BEFORE processing the next file
        if STOP_REQUESTED:
            print("--- Stop requested by user. Exiting safely. ---")
            break

        try:
            source_path_str = doc.metadata.get('source', 'unknown')
            
            if not history.should_process(original_path):
                continue

            original_path = Path(source_path_str)
            safe_name = clean_filename(original_path.stem)
            
            # 3. Validate Content
            content = doc.page_content
            if not content or len(content.strip()) < 500:
                logger.warning(f"Skipping {original_path.name} - Content too short.")
                stats["ERROR"] += 1
                continue

            # 4. Generate Writeup
            writeup_body = generate_writeup(content)

            # 5. Save to File (Collision Safe)
            base_filename = f"{safe_name}.md"
            final_output_path = get_unique_path(output_dir, base_filename)
            
            with open(final_output_path, "w", encoding="utf-8") as f:
                f.write(writeup_body)

            # 6. Update Log (Resume Logic)
            history.record_processing(
                original_path, 
                output_file=str(final_output_path)
            )
            
            stats["SUCCESS"] += 1
            logger.info(f"Processed: {original_path.name} -> {final_output_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {doc.metadata.get('source')}: {e}")
            stats["ERROR"] += 1

    history.save()
    print("\n--- Processing Summary ---")
    print(f"Successfully Generated: {stats['SUCCESS']}")
    print(f"Errors encountered:     {stats['ERROR']}")
    print(f"Check 'writeup_generation.log' for details.")

# -------------------------------------------------------------------------
# 4. Main
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LLM Writeup Generator")
    parser.add_argument("input_dir", type=Path, help="Path to source PDFs")
    args = parser.parse_args()
    
    if not args.input_dir.exists():
        print("Input directory does not exist.")
        sys.exit(1)

    output_dir = Path(__file__).parent / "writeups"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    process_documents_sequentially(args.input_dir, output_dir)

if __name__ == "__main__":
    main()