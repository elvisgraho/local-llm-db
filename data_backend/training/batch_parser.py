import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.load_documents import calculate_context_ceiling, load_documents
from training.history_manager import ProcessingHistory
from training.extract_metadata_llm import (
    get_llm_response,
    clean_and_parse_json,
    extract_text_parts,      
    PydanticOutputParser,
    Field,
    BaseModel, 
    ChatPromptTemplate
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redteam_filter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RedTeamFilter(BaseModel):
    decision: str = Field(..., description="Must be strictly 'KEEP' or 'DELETE'.")
    reasoning: str = Field(..., description="Concise technical justification.")

def get_filter_prompt() -> ChatPromptTemplate:
    template_str = """You are a Senior IT Engineer and Knowledge Base Curator.
Your task: Decide if the provided document text is valuable for a penetration testing library.

### CRITERIA TO 'KEEP' (Technical Value)
- Contains **actionable** content: code snippets, exploit payloads, command-line usage.
- Explains specific vulnerabilities (CVEs), architectural internals, or bypass techniques.
- Technical manuals, whitepapers, or detailed tutorials.

### CRITERIA TO 'DELETE' (Noise/Junk)
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays.
- **Junk**: Unreadable OCR, corrupted text, or placeholder data.
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.

### INPUT TEXT (Sample):
{text}

### INSTRUCTIONS
1. If in doubt, **KEEP** it. Only Delete if it is clearly marketing or junk.
2. Return ONLY valid JSON matching the schema below.

{format_instructions}
"""
    return ChatPromptTemplate.from_template(template_str)

# -------------------------------------------------------------------------
# 3. Logic & Operations
# -------------------------------------------------------------------------

def evaluate_document_content(text: str) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=RedTeamFilter)
    
    # Extract representative parts to avoid token limits while maintaining context
    sampled_text = extract_text_parts(text)
    
    try:
        prompt = get_filter_prompt()
        prompt_val = prompt.invoke({
            "text": sampled_text, 
            "format_instructions": parser.get_format_instructions()
        })
        
        raw_response = get_llm_response(prompt_val.to_string())
        result = clean_and_parse_json(raw_response)
        
        if not result or 'decision' not in result:
            return {"decision": "KEEP", "reasoning": "LLM returned invalid format"}
            
        return result

    except Exception as e:
        logger.error(f"  [LLM Error] {e}")
        return {"decision": "KEEP", "reasoning": f"Exception: {str(e)}"}

def safe_move_to_delete(file_path: Path, delete_dir: Path) -> bool:
    try:
        if not file_path.exists():
            return False

        destination = delete_dir / file_path.name
        
        # Handle Filename Collisions in Trash
        if destination.exists():
            stem = file_path.stem
            suffix = file_path.suffix
            counter = 1
            while destination.exists():
                destination = delete_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        
        shutil.move(str(file_path), str(destination))
        return True
    except Exception as e:
        logger.error(f"  [ERROR] Failed to move {file_path.name}: {e}")
        return False

# -------------------------------------------------------------------------
# 4. Main Execution
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Red Team Doc Filter (Sequential/Local LLM).")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("--reset", action="store_true", help="Ignore history and re-process all files")
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    # 1. Setup Directories
    delete_dir = Path(__file__).parent / "to_delete"
    delete_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup History
    history_path = args.input_dir / "filter_history.json"
    if args.reset and history_path.exists():
        history_path.unlink()
    history = ProcessingHistory(history_path)

    # 3. Dry Scan: Filter paths without loading content
    print("\nScanning directory for pending files...")
    supported_ext = {'.pdf', '.txt', '.md', '.markdown', '.log', '.json', '.py'}
    
    # Generator to minimize memory pressure during rglob
    all_paths = (p.resolve() for p in args.input_dir.rglob("*") if p.is_file())
    
    pending_paths = [
        p for p in all_paths 
        if p.suffix.lower() in supported_ext and history.should_process(p)
    ]

    if not pending_paths:
        print(f"All files up to date.")
        sys.exit(0)

    # 4. Targeted Loading: Use the file_paths parameter
    print(f"Loading content for {len(pending_paths)} pending files...")
    try:
        filtered_documents: List[Document] = load_documents(file_paths=pending_paths)
    except Exception as e:
        logger.error(f"Loader failed: {e}")
        sys.exit(1)

    if not filtered_documents:
        print("No documents were successfully loaded.")
        sys.exit(0)

    # 5. Dynamic Resource Calculation
    filtered_documents = calculate_context_ceiling(filtered_documents)

    stats = {"KEPT": 0, "DELETED": 0, "SKIPPED_ERROR": 0}

    print(f"--- Configuration ---")
    print(f"Pending Documents: {len(filtered_documents)}")
    print(f"Trash:   {delete_dir}\n")

    # 6. Optimized Execution Loop
    try:
        for i, doc in enumerate(filtered_documents, 1):
            file_path = Path(doc.metadata.get('source', '')).resolve()
            
            # Atomic mtime capture before potential deletion/move
            current_mtime = file_path.stat().st_mtime
            content = doc.page_content

            # Fast Fail: Entropy check
            if not content or len(content.strip()) < 500:
                logger.info(f"[{i}/{len(filtered_documents)}] [SHORT] {file_path.name}")
                if safe_move_to_delete(file_path, delete_dir):
                    stats["DELETED"] += 1
                    history.record_processing(file_path, mtime=current_mtime, decision="DELETE", reason="insufficient_length")
                    history.save()
                continue

            # LLM Evaluation
            sys.stdout.write(f"[{i}/{len(filtered_documents)}] Analyzing: {file_path.name}\r")
            result = evaluate_document_content(content)
            decision = result.get('decision', 'KEEP').upper()
            reason = result.get('reasoning', 'no_reason')

            sys.stdout.write("\033[K") # Clear line

            if decision == 'DELETE':
                if safe_move_to_delete(file_path, delete_dir):
                    logger.info(f"[DELETED] {file_path.name}")
                    stats["DELETED"] += 1
                else:
                    stats["SKIPPED_ERROR"] += 1
                    continue
            else:
                logger.info(f"[KEPT] {file_path.name}")
                stats["KEPT"] += 1

            # Update History
            history.record_processing(file_path, mtime=current_mtime, decision=decision, reason=reason)
            history.save()

    except KeyboardInterrupt:
        print("\n\n[!] Interrupt detected. History synchronized.")
    finally:
        history.save()

    print(f"\n--- Final Summary ---")
    print(f"New Kept:    {stats['KEPT']}")
    print(f"New Deleted: {stats['DELETED']}")
    print(f"Errors:      {stats['SKIPPED_ERROR']}")

if __name__ == "__main__":
    main()