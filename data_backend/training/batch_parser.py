import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.load_documents import load_documents
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
    # REVERTED: Using location relative to script as requested
    delete_dir = Path(__file__).parent / "to_delete"
    delete_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Setup History (stored in the input folder so it stays with the data)
    history_path = args.input_dir / "filter_history.json"
    if args.reset and history_path.exists():
        print("Resetting history...")
        history_path.unlink()
        
    history = ProcessingHistory(history_path)
    
    print(f"--- Configuration ---")
    print(f"Source:  {args.input_dir}")
    print(f"Trash:   {delete_dir}")
    print(f"History: {history_path}")

    # 3. Load Documents
    print("\nLoading documents index...")
    documents = load_documents(directory=args.input_dir)

    if not documents:
        print("No documents found in loader.")
        sys.exit(0)

    total_docs = len(documents)
    stats = {"KEPT": 0, "DELETED": 0, "ALREADY_DONE": 0, "SKIPPED_ERROR": 0}
    
    print(f"Found {total_docs} candidates. Starting process...\n")

    try:
        for i, doc in enumerate(documents, 1):
            source_path_str = doc.metadata.get('source', '')
            if not source_path_str:
                continue
            
            file_path = Path(source_path_str).resolve()
            
            # --- Check 1: Existence & History ---
            # We must check existence first, otherwise stat() fails
            if not file_path.exists():
                continue

            # Capture mtime NOW, before we potentially delete the file
            current_mtime = file_path.stat().st_mtime
            if not history.should_process(file_path):
                stats["ALREADY_DONE"] += 1
                continue

            content = doc.page_content
            
            # --- Check 2: Empty Content (Fast Fail) ---
            if not content or len(content.strip()) < 500:
                logger.info(f"[{i}/{total_docs}] [TOO SHORT] {file_path.name}")
                if safe_move_to_delete(file_path, delete_dir):
                    stats["DELETED"] += 1
                    # Update history using the captured mtime
                    history.record_processing(
                        file_path, 
                        mtime=current_mtime, 
                        decision="DELETE", 
                        reason=reason
                    )
                continue

            # --- Check 3: LLM Evaluation ---
            print(f"[{i}/{total_docs}] Analyzing: {file_path.name} ...", end="\r")
            
            result = evaluate_document_content(content)
            decision = result.get('decision', 'KEEP').upper()
            reason = result.get('reasoning', 'No reason provided')

            # Clear console line
            sys.stdout.write("\033[K") 

            if decision == 'DELETE':
                logger.info(f"[{i}/{total_docs}] [DELETE] {file_path.name}")
                
                if safe_move_to_delete(file_path, delete_dir):
                    stats["DELETED"] += 1
                    # Update history using the captured mtime (file is now gone!)
                    history.record_processing(
                        file_path, 
                        mtime=current_mtime, 
                        decision="DELETE", 
                        reason=reason
                    )
                else:
                    stats["SKIPPED_ERROR"] += 1
            else:
                logger.info(f"[{i}/{total_docs}] [KEEP  ] {file_path.name}")
                stats["KEPT"] += 1
                # Update history
                history.record_processing(
                    file_path, 
                    mtime=current_mtime, 
                    decision="KEEP", 
                    reason=reason
                )

    except KeyboardInterrupt:
        print("\n\nStopping early... Saving history...")
    finally:
        # Ensure history is saved on exit
        history.save()

    print(f"\n--- Summary ---")
    print(f"Analyzed: {stats['KEPT'] + stats['DELETED']}")
    print(f"Skipped:  {stats['ALREADY_DONE']} (Up to date)")
    print(f"Deleted:  {stats['DELETED']}")
    print(f"Errors:   {stats['SKIPPED_ERROR']}")
    print(f"Trash Dir: {delete_dir}")

if __name__ == "__main__":
    main()