import re
import sys
import logging
from typing import Dict, Any, Optional, Tuple

# --- Modern LangChain & Pydantic Imports ---
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

# --- Local Imports ---
from training.templates import OCR_SYSTEM_PROMPT, DocumentMetadata, get_metadata_extraction_prompt
from training.llm_client import get_llm_response, clean_and_parse_json, extract_text_parts, get_llm_client

try:
    from common.config import config
except ImportError:
    from data_backend.common.config import config

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logger = logging.getLogger(__name__)

# Matches "Tags: tag1, tag2, T1059" at start of file (Case Insensitive)
MANUAL_TAGS_PATTERN = re.compile(r'^Tags:\s*(.+)$', re.MULTILINE | re.IGNORECASE)
MITRE_ID_PATTERN = re.compile(r'^T\d{4}(\.\d{3})?$')

def _extract_tags_from_content(content: str) -> Tuple[str | Any, str]:
    """Finds 'Tags: ...', parses, and STRIPS it from content."""
    match = MANUAL_TAGS_PATTERN.search(content)
    if match:
        tag_str = match.group(1)
        try:
            # Remove the line from content
            start, end = match.span()
            new_content = content[:start] + content[end:].lstrip()
            return tag_str, new_content
        except Exception:
            pass
    return None, content

# --- 6. Main Exported Function ---

def add_metadata_to_document(doc: Document, add_tags_llm: bool) -> Optional[Document]:
    """
    Hybrid Strategy:
    1. Extract Manual Tags (if present) & strip header.
    2. If LLM requested, run extraction.
    3. Merge results.
    """
    
    # 1. Manual Extraction
    manual_meta, new_content = _extract_tags_from_content(doc.page_content)
    if new_content != doc.page_content:
        doc.page_content = new_content

    # 2. LLM Extraction
    llm_meta = {}
    if add_tags_llm:
        try:
            logger.debug(f"DEBUG: Triggering LLM extraction for {doc.metadata.get('source')}...")
            prompt = get_metadata_extraction_prompt()
            parser = PydanticOutputParser(pydantic_object=DocumentMetadata)
            
            prompt_val = prompt.invoke({
                "text": extract_text_parts(doc.page_content),
                "format_instructions": parser.get_format_instructions()
            })
            
            raw = get_llm_response(prompt_val.to_string())
            parsed_json = clean_and_parse_json(raw)

            if parsed_json:
                validated = DocumentMetadata.model_validate(parsed_json)
                llm_meta = validated.model_dump(exclude_none=True)
                
                is_valid = llm_meta.pop('is_technical_content', True)
                if is_valid is False:
                    chapter = doc.metadata.get('chapter_title', '')
                    src = doc.metadata.get('source')
                    logger.info(f"🚫 Skipping Chunk: {src} {f'[{chapter}]' if chapter else ''} (Flagged as non-technical)")
                    return None

                
        except Exception as e:
            logger.error(f"LLM Extraction failed for {doc.metadata.get('source')}: {e}")

    # 3. Merge Strategy (Manual > LLM)
    final_meta = llm_meta.copy()

    # 4. Update Document
    if final_meta:
        doc.metadata.update(final_meta)
        logger.debug(f"DEBUG: Successfully tagged {doc.metadata.get('source')}")
        
    return doc

def process_image_tag(match: re.Match) -> str:
    """
    Standalone callback for re.sub.
    Takes a regex match, extracts Base64, calls Vision LLM via get_llm_response,
    and returns a formatted text description.
    """
    base64_data = match.group(1).strip()
    if not base64_data:
        return ""

    try:
        user_prompt = f"data:image/png;base64,{base64_data}\n\nTranscribe the text and layout from this image exactly."

        client = get_llm_client()
        response = client.get_response(
            prompt_text=user_prompt,
            system_content=OCR_SYSTEM_PROMPT,
            temperature=0.1,
            model_name=client.ocr_model_name
        )

        if not response:
            return ""

        cleaned_response = response.strip()
        if cleaned_response.startswith("```markdown"):
            cleaned_response = cleaned_response.replace("```markdown", "").replace("```", "")
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.replace("```", "")

        return f"\n[IMAGE CONTENT START]\n{cleaned_response.strip()}\n[IMAGE CONTENT END]\n"

    except Exception as e:
        logger.error(f"Vision LLM (Gliese-OCR) failed: {e}")
        return "\n[IMAGE PROCESSING ERROR: Content could not be extracted]\n"
