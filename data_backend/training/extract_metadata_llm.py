import json
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Modern LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Pydantic V2 Imports ---
from pydantic import BaseModel, Field, ValidationError

# Add parent directory to Python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir.parent) not in sys.path:
    sys.path.insert(0, str(current_dir.parent))

from query.global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL

logger = logging.getLogger(__name__)

# --- Pre-compiled Regex ---
THINKING_PATTERN = re.compile(
    r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>|'
    r'\s*\[/?(?:thinking|thought|reasoning|think)\b[^\]]*\]\s*|'
    r'\s*\((?:thinking|thought|reasoning|think)\b[^)]*\)\s*',
    flags=re.DOTALL | re.IGNORECASE
)
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)
TAGS_LINE_PATTERN = re.compile(r"^\s*Tags:\s*(\{.*\})\s*$", re.IGNORECASE | re.MULTILINE)

class DocumentMetadata(BaseModel):
    """Schema for document metadata extracted by LLM (Pydantic V2)."""
    content_type: Optional[str] = Field(None, description="Type of content (e.g., 'technical-doc', 'code-example')")
    main_topic: Optional[str] = Field(None, description="Main topic or subject (1-3 words)")
    key_concepts: Optional[str] = Field(None, description="Comma-separated list of concepts")
    has_code: bool = Field(False, description="Contains code examples")
    has_instructions: bool = Field(False, description="Contains steps/commands")
    is_tutorial: bool = Field(False, description="Is tutorial-like")
    section_type: Optional[str] = Field(None, description="Graph section type (scenario/mitigation/impact)")
def get_metadata_extraction_prompt() -> ChatPromptTemplate:
    template_str = """You are a metadata extraction specialist. Analyze the text and return a JSON object.

Text:
{text}

Instructions:
1. Return ONLY valid JSON.
2. No explanations or markdown formatting outside the JSON block.
3. Use the following schema:
{format_instructions}
Example output:
{{
    "content_type": "technical-doc",
    "main_topic": "async-python",
    "key_concepts": "coroutines, await",
    "has_code": true,
    "has_instructions": false,
    "is_tutorial": true,
    "section_type": "explanation"
}}"""
    return ChatPromptTemplate.from_template(template_str)

def _get_llm_response(prompt_text: str) -> str:
    headers = {"Content-Type": "application/json"}
    
    # --- URL FIX: Construct correct Endpoint ---
    base = LOCAL_LLM_API_URL.rstrip('/')
    if not base.endswith("/v1/chat/completions"):
        if base.endswith("/v1"):
            url = f"{base}/chat/completions"
        else:
            url = f"{base}/v1/chat/completions"
    else:
        url = base
        
    print(f"DEBUG: Tagging via {url} (Model: {LOCAL_MAIN_MODEL})", flush=True)

    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return content.strip()
    except Exception as e:
        print(f"DEBUG: LLM Request FAILED: {e}", flush=True)
        return ""

def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    if not text: return {}
    text = THINKING_PATTERN.sub('', text).strip()
    match = JSON_BLOCK_PATTERN.search(text)
    if match: text = match.group(1)
    else:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1: text = text[start : end + 1]
    try:
        return json.loads(text)
    except Exception:
        # Attempt simple fix for trailing commas
        try:
            fixed_text = re.sub(r',(\s*[\]}])', r'\1', text)
            return json.loads(fixed_text)
        except Exception:
            return {}
def validate_metadata_field(field_name: str, value: Any) -> Any:
    """Validate a single metadata field against the Pydantic model type."""
    try:
        # Pydantic v2: use model_fields
        field_info = DocumentMetadata.model_fields.get(field_name)
        if not field_info:
            return value

        # Get type annotation
        target_type = field_info.annotation

        if target_type == str or target_type == Optional[str]:
            if isinstance(value, list):
                return ", ".join(str(v) for v in value)
            return str(value)
        
        if target_type == bool or target_type == Optional[bool]:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
            
        return value
    except Exception:
        # Safe defaults
        return "unknown" if "type" in field_name or "topic" in field_name else False

def extract_metadata_llm(text: str) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=DocumentMetadata)
    try:
        prompt_template = get_metadata_extraction_prompt()
        format_instructions = parser.get_format_instructions()
        
        prompt_value = prompt_template.invoke({
            "text": text[:3000], 
            "format_instructions": format_instructions
        })
        final_prompt_str = prompt_value.to_string()
        
        response_text = _get_llm_response(final_prompt_str)
        metadata_dict = _clean_and_parse_json(response_text)
        
        if not metadata_dict:
            return {}

        # Validate fields against schema manually for robustness
        validated_data = {}
        for k, v in metadata_dict.items():
            if k in DocumentMetadata.model_fields:
                validated_data[k] = validate_metadata_field(k, v)
        
        return validated_data

    except Exception as e:
        print(f"DEBUG: Critical Error in extract_metadata_llm: {e}", flush=True)
        return {}

def _extract_tags_from_content(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    match = TAGS_LINE_PATTERN.search(content)
    if match:
        json_str = match.group(1)
        try:
            raw_data = json.loads(json_str)
            validated = DocumentMetadata(**raw_data)
            start, end = match.span()
            new_content = content[:start] + content[end:].lstrip()
            return validated.model_dump(exclude_none=True), new_content
        except Exception:
            pass
    return None, content

def add_metadata_to_document(doc: Document, add_tags_llm: bool, max_chars: int = 5000) -> Document:
    # 1. Manual Extraction
    extracted_meta, new_content = _extract_tags_from_content(doc.page_content)
    if new_content != doc.page_content:
        doc.page_content = new_content
    if extracted_meta:
        doc.metadata.update(extracted_meta)
        return doc

    # 2. LLM Extraction
    if add_tags_llm:
        print(f"DEBUG: Triggering LLM extraction for {doc.metadata.get('source')}...", flush=True)
        
        preview_text = doc.page_content[:max_chars]
        llm_meta = extract_metadata_llm(preview_text)
        
        if llm_meta:
            clean_meta = {k: v for k, v in llm_meta.items() if v not in [None, ""]}
            doc.metadata.update(clean_meta)
            print(f"DEBUG: Successfully tagged {doc.metadata.get('source')}", flush=True)
        else:
            print(f"DEBUG: Failed to tag {doc.metadata.get('source')}", flush=True)

    return doc

def format_source_filename(source: str) -> str:
    """Format filename for logging display."""
    path = Path(source)
    name = path.name
    if len(name) > 30:
        return name[:27] + "..."
    return name