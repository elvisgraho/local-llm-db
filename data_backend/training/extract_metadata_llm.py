import json
import re
import sys
import logging
from typing import Dict, Any, List, Literal, Optional, Tuple
import unicodedata
import requests

# --- Modern LangChain & Pydantic Imports ---
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, model_validator

# --- Local Imports ---
# Ensure query.global_vars exists in your project structure
from query.global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL

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

'''
# CURL FULL MITRE
curl -s https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json \
     https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json \
     https://raw.githubusercontent.com/mitre/cti/master/ics-attack/ics-attack.json \
| jq -r '.objects[] | select(.type=="x-mitre-tactic") | .name' | sort | uniq | sed 's/.*/        "&",/'
'''

class DocumentMetadata(BaseModel):
    """
    Schema for Knowledge Graph extraction.
    """
    # 1. Classification
    is_technical_content: bool = Field(..., description="True if text contains actionable exploits, code analysis, or threat intel. False for news/marketing.")

    # 2. Core Metadata
    main_topic: Optional[str] = Field(None, description="Subject (e.g., 'Active Directory Security')")
    summary_dense: Optional[str] = Field(None, description="One dense sentence with keywords.")
    code_languages: List[str] = Field(None, description="Programming languages used in the script.")

    # 3. MITRE
    mitre_tactics: List[Literal[
        "Collection",
        "Command and Control",
        "Credential Access",
        "Defense Evasion",
        "Discovery",
        "Evasion",
        "Execution",
        "Exfiltration",
        "Impact",
        "Impair Process Control",
        "Inhibit Response Function",
        "Initial Access",
        "Lateral Movement",
        "Network Effects",
        "Persistence",
        "Privilege Escalation",
        "Reconnaissance",
        "Remote Service Effects",
        "Resource Development"
    ]] = Field(default_factory=list, description="List of tactics found (e.g., 'initial_access', 'execution').")
    mitre_technique_primary_ids: List[str] = Field(None, description="Primary T MITRE IDs (e.g., 'T1059' without a dot) NEVER Sub-technique")

    @model_validator(mode='after')
    def validate_content(self):
        # If not technical, wipe the graph data to save space/tokens
        if not self.is_technical_content:
            self.mitre_tactics = []
            self.mitre_technique_primary_ids = []
        return self

def get_metadata_extraction_prompt() -> ChatPromptTemplate:
    template_str = """You are a Principal Security Research Assistant building a cybersecurity knowledge database.
Extract structured metadata from the provided text.

### GLOBAL EXCLUSION RULES (STRICT)
1. **NO GENERIC COMMANDS:** strictly IGNORE standard shell operations: `cd`, `ls`, `mv`, `cp`, `mkdir`, `cat`, `echo`, `chmod`, `chown`.
2. **NO GENERIC PATHS:** IGNORE `/tmp`, `/home`, `C:\\Users`, `Program Files` unless part of a specific exploit chain.
3. **NO LOCAL INFRA:** IGNORE `localhost`, `127.0.0.1`, `0.0.0.0`, `192.168.x.x`.

### CRITERIA TO 'DISCARD' (is_technical_content = false)
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays, or Copyright/Legal pages.
- **Junk**: Unreadable OCR, Table of Contents, or Dedication pages.

### CRITERIA TO 'KEEP' (is_technical_content = true)
- Contains **actionable** content: code snippets, exploit payloads, command-line usage.
- Explains specific vulnerabilities (CVEs), architectural internals, or bypass techniques.
- Technical manuals, whitepapers, or detailed tutorials.

### ACTUAL INPUT TEXT
{text}

### 4. OUTPUT INSTRUCTIONS
- Return valid JSON matching the schema below.
- {format_instructions}
"""
    return ChatPromptTemplate.from_template(template_str)

# --- 4. Helper Functions ---

# Use session to prevent open socket accumulation
session = requests.Session()

def get_llm_response(prompt_text: str, system_content: Optional[str] = None, temperature: int = 0.3) -> str:
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
        "model": LOCAL_MAIN_MODEL,
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

def _extract_tags_from_content(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
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
            print(f"DEBUG: Triggering LLM extraction for {doc.metadata.get('source')}...", flush=True)
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
                    print(f"🚫 Skipping Chunk: {src} {f'[{chapter}]' if chapter else ''} (Flagged as non-technical)", flush=True)
                    return None

                
        except Exception as e:
            logger.error(f"LLM Extraction failed for {doc.metadata.get('source')}: {e}")

    # 3. Merge Strategy (Manual > LLM)
    final_meta = llm_meta.copy()
    
    if manual_meta:
        # If manual tags exist, we force technical=True
        
        # Override specific fields
        if manual_meta.get("mitre_technique_id"):
            final_meta["mitre_technique_id"] = manual_meta["mitre_technique_id"]
        # Merge Code Languages
        if manual_meta.get("code_languages"):
            final_meta.setdefault("code_languages", []).extend(manual_meta["code_languages"])

    # 4. Update Document
    if final_meta:
        doc.metadata.update(final_meta)
        print(f"DEBUG: Successfully tagged {doc.metadata.get('source')}", flush=True)
        
    return doc
