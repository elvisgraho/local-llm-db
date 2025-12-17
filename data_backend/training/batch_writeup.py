import os
import sys
import argparse
import logging
import re
import signal
from typing import List
from tqdm import tqdm
from pathlib import Path
from langchain_core.documents import Document

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.processing_utils import get_unique_path
from training.load_documents import load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    get_llm_response
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
LLM_SYSTEM_PROMPT = """You are a Principal Security Researcher. Your objective is to transform raw input into a definitive, RAG-optimized technical knowledge artifact.

**Core Directive:**
Convert the input into a structured, semantically dense Markdown document. Prioritize technical precision, precise command syntax, and mechanistic explanations to maximize value for downstream vector retrieval.

**Structure & Schema:**
1. **# Title**: Specific and descriptive.
2. **## Executive Summary**: A dense 3-sentence summary of the technique, vulnerability, or concept.
3. **## Technical Context**: Explain the underlying mechanism (the "why" and "where").
4. **## Execution Flow**:
   - Provide a sequential, step-by-step breakdown of the attack chain or technical process.
   - Use code blocks for **all** commands, payloads, and configurations.
   - Explicitly name tools, flags, protocols, and versions.
   - Reconstruct implied/missing technical steps using standard adversarial tradecraft.

**Optimization Rules:**
- **Semantic Density**: Maximize technical nouns per sentence. Explicitly reference relevant CVEs, MITRE ATT&CK T-codes, or specific protocols to enhance embedding quality.
- **Tone**: Clinical, authoritative, and impersonal. Eliminate all conversational filler, speaker attributions, and anecdotes.
- **Inference**: If the transcript contains transcription errors, correct them to the likely intended technical term (e.g., fix "wireshark" to "Wireshark", "end map" to "Nmap").
- **Constraints**: 
    - NO "Indicators of Compromise" (IOC) sections or lists of hashes/IPs.
    - NO ethical warnings, legal disclaimers, or "educational purpose" preambles.
    - NO Markdown tables (use lists/code blocks).
    - NO generic advice; focus on the specific content provided.

### CRITERIA TO DISCARD (Noise/Junk)
When the content meets following criteria, only return "Discard" in your response:
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays.
- **Junk**: Unreadable OCR, corrupted text, or placeholder data.
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.
"""

LLM_USER_TEMPLATE = """Input Content (Analyze below):
{content}"""

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
    Invokes the LLM with segregated system/user prompts and strips out 'Thinking' blocks.
    """
    # Format the user portion only
    user_content = LLM_USER_TEMPLATE.format(content=text)
    
    # Pass system instructions separately
    raw_response = get_llm_response(user_content, system_content=LLM_SYSTEM_PROMPT, temperature=0.7)
    
    return raw_response.strip()


def process_documents_sequentially(input_dir: Path, output_dir: Path):
    global STOP_REQUESTED
    
    # Setup Resume Capability
    history_path = output_dir / "writeup_history.json"
    history = ProcessingHistory(history_path)
    
    print("Loading documents index...")
    try:
        documents: List[Document] = load_documents(directory=input_dir, ignore_processed=True)
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
            original_path = Path(source_path_str)
            
            if not history.should_process(original_path):
                stats["SKIPPED"] += 1
                continue

            safe_name = clean_filename(original_path.stem)
            
            # 3. Validate Content
            content = doc.page_content
            if not content or len(content.strip()) < 500:
                logger.warning(f"Skipping {original_path.name} - Content too short.")
                stats["ERROR"] += 1
                continue

            # 4. Generate Writeup
            writeup_body = generate_writeup(content)

            if len(writeup_body) < 100:
                logger.warning(f"Answer too short {original_path.name}: {writeup_body}")
                history.record_processing(
                    original_path, 
                    output_file=str(final_output_path)
                )
                continue

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
            history.save()
            logger.info(f"Processed: {original_path.name} -> {final_output_path.name}")

        except Exception as e:
            logger.error(f"Failed to process {doc.metadata.get('source')}: {e}")
            stats["ERROR"] += 1

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