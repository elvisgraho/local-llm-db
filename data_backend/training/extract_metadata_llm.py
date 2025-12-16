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

class Entity(BaseModel):
    """
    Atomic entity for graph node population.
    Only extracts UNIQUE, SEARCHABLE identifiers.
    """
    name: str = Field(..., description="Canonical name. normalize casing (e.g., 'powershell' -> 'PowerShell').")
    
    category: Literal[
        'TOOL',             # Burp Suite, Cobalt Strike, Metasploit, nmap
        'VULNERABILITY',    # CVE-2023-1234, Log4Shell, EternalBlue
        'THREAT_ACTOR',     # Lazarus, APT29, Fancy Bear
        'FILE_PATH',        # /etc/shadow, C:\Windows\System32\drivers
        'NET_IOC',          # IP addresses (public only), Domains, unique API endpoints.
        'OFFENSIVE_CMD',    # SPECIFIC malicious arguments or complex one-liners.
        'TECHNIQUE',        # SQL Injection, DLL Sideloading, Kerberoasting
        'CONFIG_KEY',       # Registry keys, unique env vars
        'API_HEADER'        # X-Forwarded-For, Authorization
    ] = Field(..., description="The precise category of the entity.")

class DocumentMetadata(BaseModel):
    """
    Schema for Knowledge Graph extraction.
    """
    # 1. Chain of Thought (Helps JSON stability and logic)
    reasoning: str = Field(..., description="Analyze the text. Why is this technical? Identify specific entities vs noise.")

    # 2. Classification
    is_technical_content: bool = Field(..., description="True if text contains actionable exploits, code analysis, or threat intel. False for news/marketing.")

    # 3. Core Metadata
    main_topic: Optional[str] = Field(None, description="Subject (e.g., 'Active Directory Security')")
    summary_dense: Optional[str] = Field(None, description="One dense sentence with keywords.")
    
    # 4. Graph Data
    entities: List[Entity] = Field(default_factory=list, description="List of unique entities.")

    # 5. MITRE
    mitre_tactics: List[str] = Field(default_factory=list, description="List of tactics found (e.g., 'initial_access', 'execution').")
    mitre_technique_id: Optional[str] = Field(None, description="Primary MITRE ID (e.g., 'T1059').")

    @model_validator(mode='after')
    def validate_content(self):
        # If not technical, wipe the graph data to save space/tokens
        if not self.is_technical_content:
            self.entities = []
            self.mitre_tactics = []
            self.mitre_technique_id = None
        return self

def get_metadata_extraction_prompt() -> ChatPromptTemplate:
    template_str = """You are a Principal Security Research Assistant building a cybersecurity knowledge graph.
Extract structured metadata from the provided text.

### GLOBAL EXCLUSION RULES (STRICT)
1. **NO GENERIC COMMANDS:** strictly IGNORE standard shell operations: `cd`, `ls`, `mv`, `cp`, `mkdir`, `cat`, `echo`, `chmod`, `chown`.
2. **NO GENERIC PATHS:** IGNORE `/tmp`, `/home`, `C:\\Users`, `Program Files` unless part of a specific exploit chain.
3. **NO LOCAL INFRA:** IGNORE `localhost`, `127.0.0.1`, `0.0.0.0`, `192.168.x.x`.
4. **NO COMMON TERMS:** IGNORE `Internet`, `Computer`, `Malware` (too generic), `Hacker`, `Server`.

### CATEGORY DEFINITIONS
- **TOOL**: Specific software (e.g., `Mimikatz`, `Burp Suite`).
- **VULNERABILITY**: CVE IDs or named bugs (e.g., `CVE-2021-44228`, `PrintNightmare`).
- **THREAT_ACTOR**: Hacking groups (e.g., `APT28`, `Fin7`).
- **FILE_PATH**: Critical system paths (e.g., `/etc/passwd`, `ntds.dit`).
- **NET_IOC**: Public IPs, Malicious Domains, or unique API URIs (e.g., `/api/admin/upload`).
- **OFFENSIVE_CMD**: Unique attack strings or payload signatures (e.g., `Invoke-WebRequest -Uri...`, `ReflectivePEInjection`). **NEVER** simple file moves.
- **TECHNIQUE**: Attack concepts (e.g., `Pass-the-Hash`, `SQL Injection`).
- **CONFIG_KEY**: Registry keys or configuration flags.

#### FEW-SHOT EXAMPLE (Follow this logic):
**Input:** "The attacker used Lazarus Group tactics. They executed 'mv payload.exe /tmp' and then ran 'rundll32.exe user32.dll,LockWorkStation'. The exploit targets CVE-2023-9999."

**Output:**
{{
  "reasoning": "Text describes an actor (Lazarus), a specific CVE, and a suspicious rundll32 command. 'mv' command is ignored as noise.",
  "is_technical_content": true,
  "main_topic": "System Execution",
  "summary_dense": "Lazarus Group exploited CVE-2023-9999 using rundll32 for execution.",
  "entities": [
    {{"name": "Lazarus Group", "category": "THREAT_ACTOR"}},
    {{"name": "rundll32.exe user32.dll,LockWorkStation", "category": "OFFENSIVE_CMD"}},
    {{"name": "CVE-2023-9999", "category": "VULNERABILITY"}}
  ],
  "mitre_tactics": ["execution"],
  "mitre_technique_id": "T1218"
}}

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
    if not text: return {}

    # 1. Remove "Thinking" blocks
    text = THINKING_PATTERN.sub('', text).strip()
    
    # 2. Extract JSON from Markdown
    match = JSON_BLOCK_PATTERN.search(text)
    if match:
        text = match.group(1)
    else:
        start, end = text.find('{'), text.rfind('}')
        if start != -1 and end != -1:
            text = text[start : end + 1]

    # 3. Clean trailing commas
    text = re.sub(r',(\s*[\]}])', r'\1', text)
    
    # 4. [NEW] Normalize Unicode & Fix weird formatting
    # NFKC normalizes:
    #   \u2011 (Non-breaking hyphen) -> -
    #   \u2013 (En dash)             -> -
    #   \u201c (Left Double Quote)   -> "
    #   ½                            -> 1/2
    text = unicodedata.normalize('NFKC', text)

    # 5. [NEW] Explicit safety replace for JSON-breaking escapes
    # Sometimes LLMs escape the unicode like "\\u2011" which normalization misses
    text = text.replace("\\u2011", "-") \
               .replace("\\u2013", "-") \
               .replace("\\u2014", "-") \
               .replace("\\u2010", "-") \
               .replace("\\u00a0", " ") # Non-breaking space

    try:
        return json.loads(text)
    except Exception:
        return {}
    
def extract_text_parts(text: str, part_size: int = 2400, part_count: int = 20) -> str:
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

# --- 5. Logic: Manual Tag Parsing ---
def _parse_manual_tags(tag_string: str) -> Dict[str, Any]:
    """Converts 'Tags: T1059, Python' into schema dict."""
    raw_tags = [t.strip() for t in tag_string.split(',') if t.strip()]
    
    meta = {
        "reasoning": "",
        "is_technical_content": True,
        "entities": [],
        "mitre_technique_id": None
    }

    for tag in raw_tags:
        # MITRE ID
        if MITRE_ID_PATTERN.match(tag):
            meta["mitre_technique_id"] = tag
            meta["entities"].append({"name": tag, "category": "CONCEPT"})
        # Code Languages (Basic Heuristic)
        elif tag.lower() in ['python', 'bash', 'powershell', 'c++', 'go', 'javascript']:
            meta.setdefault("code_languages", []).append(tag.lower())
            meta["entities"].append({"name": tag, "category": "TOOL"})
        # Default Entity
        else:
            meta["entities"].append({"name": tag, "category": "CONCEPT"})
            
    return meta

def _extract_tags_from_content(content: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Finds 'Tags: ...', parses, and STRIPS it from content."""
    match = MANUAL_TAGS_PATTERN.search(content)
    if match:
        tag_str = match.group(1)
        try:
            partial_meta = _parse_manual_tags(tag_str)
            # Remove the line from content
            start, end = match.span()
            new_content = content[:start] + content[end:].lstrip()
            return partial_meta, new_content
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
                
                llm_meta.pop('reasoning', "")
                is_valid = llm_meta.pop('is_technical_content', True)
                if is_valid is False:
                    chapter = doc.metadata.get('chapter_title', '')
                    src = doc.metadata.get('source')
                    print(f"🚫 Skipping Chunk: {src} {f'[{chapter}]' if chapter else ''} (Flagged as non-technical)", flush=True)
                    return None
                
                # Cleanup internal 'reasoning' field so it doesn't pollute DB
                llm_meta.pop('reasoning', None)
                
        except Exception as e:
            logger.error(f"LLM Extraction failed for {doc.metadata.get('source')}: {e}")

    # 3. Merge Strategy (Manual > LLM)
    final_meta = llm_meta.copy()
    
    if manual_meta:
        # If manual tags exist, we force technical=True
        
        # Override specific fields
        if manual_meta.get("mitre_technique_id"):
            final_meta["mitre_technique_id"] = manual_meta["mitre_technique_id"]
            
        # Merge Entities (Deduplicate by name)
        existing_names = {e['name'] for e in final_meta.get('entities', [])}
        for entity in manual_meta.get('entities', []):
            if entity['name'] not in existing_names:
                final_meta.setdefault('entities', []).append(entity)
                
        # Merge Code Languages
        if manual_meta.get("code_languages"):
            final_meta.setdefault("code_languages", []).extend(manual_meta["code_languages"])

    # 4. Update Document
    if final_meta:
        doc.metadata.update(final_meta)
        print(f"DEBUG: Successfully tagged {doc.metadata.get('source')}", flush=True)
        
    return doc
