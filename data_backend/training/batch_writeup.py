"""
LLM writeup generator using batch processor base.
"""
import os
import sys
import argparse
import logging
import re
import signal
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.documents import Document

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.templates import LLM_WRITEUP_SYSTEM_PROMPT, LLM_WRITEUP_USER_TEMPLATE
from training.processing_utils import get_unique_path
from training.llm_client import get_llm_response
from training.batch_processor_base import BatchProcessorBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("writeup_generation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

STOP_REQUESTED = False


def signal_handler(sig, frame):
    global STOP_REQUESTED
    print("\n\n[!] Ctrl+C detected. Finishing current file and stopping...")
    print("[!] Please wait for the current save to complete.\n")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)


class WriteupGenerator(BatchProcessorBase):
    """Batch processor for generating technical writeups from documents."""

    def get_history_filename(self) -> str:
        return "writeup_history.json"

    def _should_stop(self) -> bool:
        return STOP_REQUESTED

    def process_document(self, doc: Document, index: int, total: int) -> Optional[Dict[str, Any]]:
        """Generate writeup for a document."""
        source_path = Path(doc.metadata.get('source', '')).resolve()
        content = doc.page_content.strip()

        if len(content) < 500:
            return None

        writeup_body = self._generate_writeup(content)
        safe_name = self._clean_filename(source_path.stem)
        final_path = get_unique_path(self.output_dir, f"{safe_name}.md")

        if not writeup_body or len(writeup_body) < 100:
            if "DELETE" in writeup_body.upper():
                logger.info(f"\nJUNK: {source_path}.")
                return {"output_path": final_path, "status": "junk"}
            else:
                logger.info(f"\nSmall Response: '{writeup_body}' for {source_path}.")
                return None

        final_path.write_text(writeup_body, encoding="utf-8")
        return {"output_path": final_path}

    def _generate_writeup(self, text: str) -> str:
        """Invoke LLM to generate writeup."""
        user_content = LLM_WRITEUP_USER_TEMPLATE.format(content=text)
        raw_response = get_llm_response(
            user_content,
            system_content=LLM_WRITEUP_SYSTEM_PROMPT,
            temperature=0.7
        )
        return raw_response.strip()

    @staticmethod
    def _clean_filename(filename: str) -> str:
        """Sanitize filename to prevent OS errors."""
        return re.sub(r'[\\/*?:"<>|]', "", filename)


def main():
    parser = argparse.ArgumentParser(description="LLM Writeup Generator")
    parser.add_argument("input_dir", type=Path, help="Path to source PDFs")
    args = parser.parse_args()

    if not args.input_dir.exists():
        print("Input directory does not exist.")
        sys.exit(1)

    output_dir = Path(__file__).parent / "writeups"

    processor = WriteupGenerator(
        input_dir=args.input_dir,
        output_dir=output_dir,
        force=False
    )
    processor.run()


if __name__ == "__main__":
    main()
