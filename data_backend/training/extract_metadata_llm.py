import json
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Literal, Optional, Tuple
import requests

# --- Modern LangChain Core Imports ---
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

# --- Pydantic V2 Imports ---
from pydantic import BaseModel, Field


try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except AttributeError:
    # Fallback for older Python versions if needed
    pass

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
    main_topic: Optional[str] = Field(None, description="Precise technical subject (e.g., 'DOM XSS', 'OAuth Flow')")
    key_concepts: List[str] = Field(default_factory=list, description="Specific technical keywords (e.g., ['windows forensics', 'docker hardening', 'iframe'])")
    code_language: Optional[str] = Field(None, description="Programming language if code is present (e.g., 'python', 'bash')")
    section_type: List[Literal[
        'reconnaissance',
        'resource_development',
        'initial_access',
        'execution',
        'persistence',
        'privilege_escalation',
        'defense_evasion',
        'credential_access',
        'discovery',
        'lateral_movement',
        'collection',
        'command_and_control',
        'exfiltration',
        'impact'
    ]] = Field(None, description="Strict mapping to MITRE ATT&CK Enterprise Tactics.")
    has_instructions: bool = Field(False, description="True if text contains reproduction steps or commands")
    is_technical_content: bool = Field(True, description="TRUE if content is actionable/technical. FALSE if content is marketing, table of contents, legal disclaimers, or generic fluff.")
def get_metadata_extraction_prompt() -> ChatPromptTemplate:
    template_str = """You are a Senior IT Engineer and document tagging specialist. Analyze the text and return a JSON object with determining if the content is worth keeping.

Text:
{text}

### CRITERIA TO 'DISCARD' (is_technical_content = false)
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays, or Copyright/Legal pages.
- **Junk**: Unreadable OCR, Table of Contents, or Dedication pages.

### CRITERIA TO 'KEEP' (is_technical_content = true)
- Contains **actionable** content: code snippets, exploit payloads, command-line usage.
- Explains specific vulnerabilities (CVEs), architectural internals, or bypass techniques.
- Technical manuals, whitepapers, or detailed tutorials.

Instructions:
1. Return ONLY valid JSON.
2. 'key_concepts' must be atomic technical terms suitable for database filtering and mixed with major themes.
3. If the text matches 'DISCARD' criteria, set "is_technical_content": false. 
4. Use the following schema:
{format_instructions}

### Example Output
Input: 
"To exploit the IDOR, change the user_id parameter in the POST /api/v1/user request.
```bash
curl -X POST https://example.com/api/v1/user -d 'user_id=1337'
```"

Output:
{{
    "main_topic": "IDOR",
    "key_concepts": ["broken access control", "parameter tampering", "api"],
    "code_language": "bash",
    "section_type": ["initial_access","persistence"]
    "has_instructions": true,
    "is_technical_content": true
}}"""
    return ChatPromptTemplate.from_template(template_str)

# prevent accumulation of open sockets
session = requests.Session()

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
        
    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.1
    }
    
    try:
        response = session.post(url, json=payload, headers=headers, timeout=240)
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
    
def extract_text_parts(text: str, part_size: int = 1800, part_count: int = 17) -> str:
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

def extract_metadata_llm(text: str) -> Dict[str, Any]:
    parser = PydanticOutputParser(pydantic_object=DocumentMetadata)
    try:
        prompt_template = get_metadata_extraction_prompt()
        format_instructions = parser.get_format_instructions()
        
        safe_text = extract_text_parts(text).replace("{", "{{").replace("}", "}}")
        prompt_value = prompt_template.invoke({
            "text": safe_text, 
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

def add_metadata_to_document(doc: Document, add_tags_llm: bool) -> Optional[Document]:
    # 1. Manual Extraction (unchanged)
    extracted_meta, new_content = _extract_tags_from_content(doc.page_content)
    if new_content != doc.page_content:
        doc.page_content = new_content
    if extracted_meta:
        doc.metadata.update(extracted_meta)
        return doc

    # 2. LLM Extraction
    if add_tags_llm:
        print(f"DEBUG: Triggering LLM extraction for {doc.metadata.get('source')}...", flush=True)
        
        llm_meta = extract_metadata_llm(doc.page_content)
        
        if llm_meta:
            is_valid = llm_meta.pop('is_technical_content', True)
            # 2. Decide to Drop
            if is_valid is False:
                print(f"🚫 Skipping Chunk: {doc.metadata.get('source')} (Flagged by LLM){f' in Chapter: {doc.metadata.get("chapter_title")}' if doc.metadata.get('chapter_title') else ''}", flush=True)
                return None  # This signals split_document to drop it
            
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