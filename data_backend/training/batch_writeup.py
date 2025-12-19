import os
import sys
import argparse
import logging
import re
import signal
from tqdm import tqdm
from pathlib import Path
from typing import List
from langchain_core.documents import Document

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.processing_utils import get_unique_path
from training.load_documents import calculate_context_ceiling, load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    get_llm_response
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("writeup_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
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
    
    # 1. Initialize State
    history_path = output_dir / "writeup_history.json"
    history = ProcessingHistory(history_path)
    
    # 2. Dry Scan (Paths Only - No Memory Load)
    print("Scanning directory for pending files...")
    supported_ext = {'.pdf', '.txt', '.md', '.markdown'}
    
    # Generate resolved absolute paths for comparison
    all_paths = (p.resolve() for p in input_dir.rglob("*") if p.is_file())
    
    pending_paths = [
        p for p in all_paths 
        if p.suffix.lower() in supported_ext and history.should_process(p)
    ]

    if not pending_paths:
        print("No new documents to process.")
        return

    # 3. Targeted Load (Using your updated function)
    # This specifically passes only the filtered list to load_documents
    print(f"Loading content for {len(pending_paths)} files...")
    documents: List[Document] = load_documents(file_paths=pending_paths, ignore_processed=True)

    if not documents:
        print("Error: Files found but content loading failed.")
        return

    # 4. Resource Allocation
    documents = calculate_context_ceiling(documents)

    # 5. Execution Loop
    stats = {"SUCCESS": 0, "ERROR": 0}
    print(f"Starting processing for {len(documents)} document segments...\n")

    for doc in tqdm(documents, desc="Generating Writeups", unit="file"):
        if STOP_REQUESTED:
            break

        try:
            source_path = Path(doc.metadata.get('source', 'unknown')).resolve()
            content = doc.page_content.strip()

            # Technical Validation
            if len(content) < 500:
                stats["ERROR"] += 1
                continue

            # LLM Inference
            writeup_body = generate_writeup(content)

            if not writeup_body or len(writeup_body) < 100:
                stats["ERROR"] += 1
                print(f"\nSmall Response: '{writeup_body}' for {source_path}.")
                continue

            # File Persistence
            safe_name = clean_filename(source_path.stem)
            final_path = get_unique_path(output_dir, f"{safe_name}.md")
            final_path.write_text(writeup_body, encoding="utf-8")

            # Update History (Immediate synchronization)
            history.record_processing(source_path, output_file=str(final_path))
            history.save()
            
            stats["SUCCESS"] += 1

        except Exception as e:
            logger.error(f"Inference failure: {e}")
            stats["ERROR"] += 1

    print(f"\n--- Final Status: {stats['SUCCESS']} Success | {stats['ERROR']} Error ---")

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