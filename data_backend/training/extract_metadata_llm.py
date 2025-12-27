import json
import re
import sys
import logging
import unicodedata
import requests
from typing import Dict, Any, Optional, Tuple

# --- Modern LangChain & Pydantic Imports ---
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser

# --- Local Imports ---
# Ensure query.global_vars exists in your project structure
from training.templates import OCR_SYSTEM_PROMPT, DocumentMetadata, get_metadata_extraction_prompt
from query.global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL, LOCAL_OCR_MODEL

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    pass

logger = logging.getLogger(__name__)

# --- 1. Regex Patterns ---
THINKING_PATTERN = re.compile(
    r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>|'
    r'\s*\[/?(?:thinking|thought|reasoning|think)\b[^\]]*\]\s*|'
    r'\s*\((?:thinking|thought|reasoning|think)\b[^)]*\)\s*',
    flags=re.DOTALL | re.IGNORECASE
)
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)

# Matches "Tags: tag1, tag2, T1059" at start of file (Case Insensitive)
MANUAL_TAGS_PATTERN = re.compile(r'^Tags:\s*(.+)$', re.MULTILINE | re.IGNORECASE)
MITRE_ID_PATTERN = re.compile(r'^T\d{4}(\.\d{3})?$') 

# --- 4. Helper Functions ---

# Use session to prevent open socket accumulation
session = requests.Session()

def get_llm_response(prompt_text: str, system_content: Optional[str] = None, temperature: float = 0.3, model_name: str = LOCAL_MAIN_MODEL) -> str:
    headers = {"Content-Type": "application/json"}
    
    # Handle different API base formatting
    base = LOCAL_LLM_API_URL.rstrip('/')
    if not base.endswith("/v1/chat/completions"):
        url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
    else:
        url = base
    
    # Construct messages preserving backward compatibility
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": prompt_text})

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        response = session.post(url, json=payload, headers=headers, timeout=240)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        logger.error(f"LLM Request FAILED: {e}")
        return ""
    

def clean_and_parse_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}

    # 1. Normalize unicode (NFKC) to convert non-standard chars (like \u2010)
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Explicitly fix common LLM-escaped unicode that NFKC might miss
    text = text.replace("\\u2010", "-").replace("\\u2011", "-") \
               .replace("\\u2013", "-").replace("\\u2014", "-") \
               .replace("\\u00a0", " ") 
    
    # 3. Remove "Thinking" blocks
    if 'THINKING_PATTERN' in globals():
        text = THINKING_PATTERN.sub('', text).strip()
    
    json_str = ""
    # 4. Extract JSON from Markdown block
    if 'JSON_BLOCK_PATTERN' in globals():
        match = JSON_BLOCK_PATTERN.search(text)
        if match:
            json_str = match.group(1)

    if not json_str:
        # Fallback: Find the outermost JSON structure (object or array)
        start_match = re.search(r'({|\[)', text)
        end_match = re.search(r'(}|\])', text[::-1]) # Search reversed for last match

        if start_match and end_match:
            start_pos = start_match.start(1)
            # end_match.end(1) is the position from the *end* of the reversed string
            # Convert to forward index: len(text) - reversed_end_pos
            end_pos = len(text) - end_match.end(1)
            
            # Use max(end_pos, start_pos + 1) to ensure at least one char is included
            # and to handle cases where the text is very short
            if end_pos > start_pos:
                json_str = text[start_pos : end_pos + 1]
            else:
                return {}
        else:
            return {}
            
    # 5. Clean trailing commas (e.g., {"key": "value",})
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
    
    try:
        return json.loads(json_str)
    except Exception:
        return {}
    
def extract_text_parts(text: str, part_size: int = 2000, part_count: int = 20) -> str:
    """
    Picks n uniformly spaced parts of size part_size from the text.
    Optimized to prevent ZeroDivisionError and logical indexing errors.
    """
    L = len(text)
    
    # If text is shorter than total requested size, return all of it
    if L <= part_size * part_count:
        return text

    if part_count <= 1:
        return text[:part_size]

    max_start = L - part_size
    parts = []

    divisor = part_count - 1 if part_count > 1 else 1

    for i in range(part_count):
        start = int(i * max_start / divisor)
        parts.append(text[start : start + part_size])

    return "".join(parts)

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
        # Construct a prompt that includes the image data.
        user_prompt = f"data:image/png;base64,{base64_data}\n\nTranscribe the text and layout from this image exactly."

        # Call the LLM with OCR-optimized settings
        # Temperature 0.1 reduces hallucinations, essential for OCR.
        response = get_llm_response(
            prompt_text=user_prompt,
            system_content=OCR_SYSTEM_PROMPT,
            temperature=0.1,
            model_name=LOCAL_OCR_MODEL
        )

        if not response:
            return ""

        # Clean up response (some models might wrap output in markdown blocks)
        cleaned_response = response.strip()
        if cleaned_response.startswith("```markdown"):
            cleaned_response = cleaned_response.replace("```markdown", "").replace("```", "")
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response.replace("```", "")

        # Return format: brackets help the embedding model / chunker distinguish it
        return f"\n[IMAGE CONTENT START]\n{cleaned_response.strip()}\n[IMAGE CONTENT END]\n"
        
    except Exception as e:
        logger.error(f"Vision LLM (Gliese-OCR) failed: {e}")
        # Return a neutral placeholder so the pipeline doesn't break
        return "\n[IMAGE PROCESSING ERROR: Content could not be extracted]\n"
