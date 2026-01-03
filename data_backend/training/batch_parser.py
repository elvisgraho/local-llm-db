"""
Red Team document filter using batch processor base.
"""
import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.templates import RedTeamFilter, get_filter_prompt
from training.llm_client import get_llm_response, clean_and_parse_json, extract_text_parts
from training.batch_processor_base import BatchProcessorBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("redteam_filter.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RedTeamFilter(BatchProcessorBase):
    """Batch processor for filtering documents based on technical content."""

    def __init__(self, input_dir: Path, output_dir: Path, force: bool = False):
        super().__init__(input_dir, output_dir, force)
        self.delete_dir = Path(__file__).parent / "to_delete"
        self.delete_dir.mkdir(parents=True, exist_ok=True)
        self.stats["deleted"] = 0
        self.stats["kept"] = 0

    def get_history_filename(self) -> str:
        return "filter_history.json"

    def process_document(self, doc: Document, index: int, total: int) -> Optional[Dict[str, Any]]:
        """Evaluate and filter document based on content."""
        source_path = Path(doc.metadata.get('source', '')).resolve()
        content = doc.page_content

        if not content or len(content.strip()) < 500:
            logger.info(f"[{index}/{total}] [SHORT] {source_path.name}")
            if self._safe_move_to_delete(source_path):
                self.stats["deleted"] += 1
                return {"decision": "DELETE", "reason": "insufficient_length"}
            return None

        sys.stdout.write(f"[{index}/{total}] Analyzing: {source_path.name}\r")
        result = self._evaluate_document_content(content)
        decision = result.get('decision', 'KEEP').upper()
        reason = result.get('reasoning', 'no_reason')

        sys.stdout.write("\033[K")

        if decision == 'DELETE':
            if self._safe_move_to_delete(source_path):
                logger.info(f"[DELETED] {source_path.name}")
                self.stats["deleted"] += 1
                return {"decision": decision, "reason": reason}
            else:
                return None
        else:
            logger.info(f"[KEPT] {source_path.name}")
            self.stats["kept"] += 1
            return {"decision": decision, "reason": reason}

    def _evaluate_document_content(self, text: str) -> Dict[str, Any]:
        """Evaluate document content using LLM."""
        from training.templates import RedTeamFilter as RTF
        parser = PydanticOutputParser(pydantic_object=RTF)

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

    def _safe_move_to_delete(self, file_path: Path) -> bool:
        """Move file to delete directory."""
        try:
            if not file_path.exists():
                return False

            destination = self.delete_dir / file_path.name

            if destination.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                counter = 1
                while destination.exists():
                    destination = self.delete_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.move(str(file_path), str(destination))
            return True
        except Exception as e:
            logger.error(f"  [ERROR] Failed to move {file_path.name}: {e}")
            return False

    def _print_summary(self):
        """Print processing summary with filter-specific stats."""
        logger.info(f"\n--- Final Summary ---")
        logger.info(f"New Kept:    {self.stats['kept']}")
        logger.info(f"New Deleted: {self.stats['deleted']}")
        logger.info(f"Errors:      {self.stats['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Red Team Doc Filter (Sequential/Local LLM).")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("--reset", action="store_true", help="Ignore history and re-process all files")
    args = parser.parse_args()

    if not args.input_dir.exists():
        logger.error(f"CRITICAL: Input directory '{args.input_dir}' does not exist.")
        sys.exit(1)

    processor = RedTeamFilter(
        input_dir=args.input_dir,
        output_dir=args.input_dir,
        force=args.reset
    )
    processor.run()


if __name__ == "__main__":
    main()
