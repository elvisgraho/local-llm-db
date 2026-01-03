"""
Metadata extraction and document renaming using batch processor base.
"""
import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from langchain_core.documents import Document
from pydantic import Field
from langchain_core.output_parsers import PydanticOutputParser

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from training.templates import DocumentMetadata
from training.processing_utils import get_unique_path
from training.llm_client import get_llm_response, clean_and_parse_json, extract_text_parts
from training.batch_processor_base import BatchProcessorBase

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metadata_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FileGenMetadata(DocumentMetadata):
    suggested_filename: str = Field(..., description="Snake_case filename (no extension, e.g., 'auth_bypass_v2'). Char limit: 50")
    release_date: str = Field(..., description="Release date in YYYY-MM-DD. Use 'Unknown' if not found.")


def get_metadata_extraction_prompt():
    from training.templates import get_metadata_extraction_prompt as gmp
    return gmp()


class MetadataExtractor(BatchProcessorBase):
    """Batch processor for metadata extraction and document tagging."""

    def get_history_filename(self) -> str:
        return "processing_history.json"

    def process_document(self, doc: Document, index: int, total: int) -> Optional[Dict[str, Any]]:
        """Extract metadata and save document with tags."""
        source_path = Path(doc.metadata.get('source', '')).resolve()
        content = doc.page_content

        existing_date = None
        date_match = re.search(r'^Released:\s*(.+)$', content, flags=re.MULTILINE | re.IGNORECASE)
        if date_match:
            existing_date = date_match.group(1).strip()
            content = re.sub(r'^Released:.*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE)

        content = re.sub(r'^Tags:\s*\{.*?\}\s*(\n|\r\n)?', '', content, count=1, flags=re.MULTILINE | re.IGNORECASE).strip()

        if not content:
            logger.warning(f"Rejecting {source_path.name}: Empty after cleaning.")
            return None

        sys.stdout.write(f"[{index}/{total}] Processing: {source_path.name}\r")
        meta = self._process_content_llm(content)

        if not meta:
            logger.warning(f"Fail: Metadata extraction error for {source_path.name}")
            return None

        if not meta.pop('is_technical_content', True):
            return None

        llm_date = meta.pop('release_date', None)
        rdate = existing_date or (llm_date if str(llm_date).lower() != 'unknown' else None)

        fname = meta.pop('suggested_filename', source_path.stem)
        date_line = f"Released: {rdate}\n" if rdate else ""
        final_data = f"{date_line}Tags: {json.dumps(meta, ensure_ascii=False)}\n\n{content}"

        out_path = get_unique_path(self.output_dir, f"{fname}.md")
        try:
            out_path.write_text(final_data, encoding='utf-8')
            sys.stdout.write("\033[K")
            return {"output_path": out_path}
        except Exception as e:
            logger.error(f"Write Error: {out_path.name}: {e}")
            return None

    def _process_content_llm(self, text: str) -> Dict[str, Any]:
        """Process content with LLM to extract metadata."""
        parser = PydanticOutputParser(pydantic_object=FileGenMetadata)
        try:
            prompt = get_metadata_extraction_prompt()
            fmt_instructions = parser.get_format_instructions()

            sampled_text = extract_text_parts(text)

            prompt_val = prompt.invoke({
                "text": sampled_text,
                "format_instructions": fmt_instructions
            })

            raw_response = get_llm_response(prompt_val.to_string())
            return clean_and_parse_json(raw_response)
        except Exception as e:
            logger.error(f"LLM Processing Error: {e}")
            return {}


def main():
    parser = argparse.ArgumentParser(description="Metadata extraction and document renaming.")
    parser.add_argument("input_dir", type=Path, help="Path to source folder")
    parser.add_argument("output_dir", type=Path, nargs="?",
                        default=Path(__file__).parent / "processed", help="Output folder")
    parser.add_argument("--force", action="store_true", help="Ignore history and process everything.")
    args = parser.parse_args()

    processor = MetadataExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        force=args.force
    )
    processor.run()


if __name__ == "__main__":
    main()
