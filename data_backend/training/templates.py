"""
Prompt Templates for Processing
"""
from typing import List, Literal, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

OCR_SYSTEM_PROMPT = """
You are a state-of-the-art Document Analysis and OCR engine. 
Your task is to examine the provided image and extract all visible text, tables, and structural elements 
into accurate Markdown format. 
Preserve headers, lists, and table structures. 
Do not add conversational filler; output only the content.
"""

class DocumentMetadata(BaseModel):
    """
    Schema for Knowledge Graph extraction.
    """
    # 1. Classification
    is_technical_content: bool = Field(..., description="True if text contains actionable exploits, code analysis, or threat intel. False for news/marketing.")

    # 2. Core Metadata
    main_topic: Optional[str] = Field(None, description="Subject (e.g., 'Active Directory Security')")
    summary_dense: Optional[str] = Field(None, description="One dense sentence with keywords.")
    code_languages: List[str] = Field([], description="Programming languages used in the script.")

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
    mitre_technique_primary_ids: List[str] = Field([], description="Primary T MITRE IDs for technique (e.g., 'T1059' without a dot) NEVER Sub-technique")

def get_metadata_extraction_prompt() -> ChatPromptTemplate:
    template_str = """You are a Principal Security Research Assistant building a cybersecurity knowledge database.
Extract structured metadata from the provided text.

If applicable, classify activity using the following verified MITRE ATT&CK Tactic definitions. 
Prioritize the Tactic ID (TAxxxx) to ensure domain-specific accuracy (Enterprise vs. Mobile vs. ICS).

### Current MITRE classification
- TA0001 (Initial Access/ENT): Entry vectors (phishing, public exploit) to gain a network foothold.
- TA0002 (Execution/ENT): Running adversary-controlled code (scripts, binaries) on local/remote systems.
- TA0003 (Persistence/ENT): Maintaining access across restarts/interruptions (registry keys, startup code).
- TA0004 (Privilege Escalation/ENT): Gaining higher-level permissions (SYSTEM/root, admin) via weaknesses.
- TA0005 (Defense Evasion/ENT): Actions to avoid detection (obfuscation, disabling security tools).
- TA0006 (Credential Access/ENT): Stealing account names/passwords (dumping, keylogging).
- TA0007 (Discovery/ENT): Gaining knowledge of the internal network and system environment.
- TA0008 (Lateral Movement/ENT): Pivoting between remote systems on a network.
- TA0009 (Collection/ENT): Gathering data (files, screenshots) relevant to the objective.
- TA0010 (Exfiltration/ENT): Removing data from the network (C2 channel, alternate paths).
- TA0011 (Command and Control/ENT): Communicating with compromised systems to direct actions.
- TA0043 (Reconnaissance/ENT): Information gathering (org, staff, infra) to support targeting.
- TA0042 (Resource Development/ENT): Establishing infrastructure (domains, accounts) for operations.
- TA0040 (Impact/ENT): Disrupting availability or compromising integrity of business processes.

- TA0027 (Initial Access/MOB): Entry vectors specifically targeting mobile device footholds.
- TA0038 (Network Effects/MOB): Intercepting/manipulating traffic without device access.
- TA0039 (Remote Service Effects/MOB): Using cloud/MDM services to monitor/control devices.

- TA0108 (Initial Access/ICS): Footholds in OT/ICS environments (PLCs, engineering workstations).
- TA0103 (Evasion/ICS): ICS-specific technical defense avoidance (distinct from TA0005).
- TA0106 (Impair Process Control/ICS): Manipulating physical control logic or parameters.
- TA0107 (Inhibit Response Function/ICS): Disabling safety/protection functions (alarms, safeguards).

### GLOBAL EXCLUSION RULES (STRICT)
1. **NO GENERIC COMMANDS:** strictly IGNORE standard shell operations: `cd`, `ls`, `mv`, `cp`, `mkdir`, `cat`, `echo`, `chmod`, `chown`.
2. **NO GENERIC PATHS:** IGNORE `/tmp`, `/home`, `C:\\Users`, `Program Files` unless part of a specific exploit chain.
3. **NO LOCAL INFRA:** IGNORE `localhost`, `127.0.0.1`, `0.0.0.0`, `192.168.x.x`.

### CRITERIA TO 'DISCARD' (is_technical_content = false)
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays, or Copyright/Legal pages.
- **Junk**: Unreadable OCR, Table of Contents, or Dedication pages.
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.

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
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.

### INPUT TEXT (Sample):
{text}

### INSTRUCTIONS
1. If in doubt, **KEEP** it. Only Delete if it is clearly marketing or junk.
2. Return ONLY valid JSON matching the schema below.

{format_instructions}
"""
    return ChatPromptTemplate.from_template(template_str)


LLM_WRITEUP_SYSTEM_PROMPT = """You are a Principal Security Researcher. Your objective is to transform raw input into a definitive, RAG-optimized technical knowledge artifact.

**Core Directive:**
Convert the input into a structured, semantically dense Markdown document. Prioritize technical precision, precise command syntax, and mechanistic explanations to maximize value for downstream vector retrieval.

**Structure & Schema:**
1. **# Title**: Specific and descriptive.
2. **## Executive Summary**: A dense 3-sentence summary of the technique, vulnerability, or concept.
3. **## Technical Context**: Explain the underlying mechanism (the "why" and "where").
4. **## Execution Flow**:
   - Provide a sequential, step-by-step breakdown of the attack chain or technical process.
   - Use code blocks for **all** commands, payloads, and configurations.
   - Explicitly name tools, flags, protocols, and versions.
   - Reconstruct implied/missing technical steps using standard adversarial tradecraft.

**Optimization Rules:**
- **Semantic Density**: Maximize technical nouns per sentence. Explicitly reference relevant CVEs, MITRE ATT&CK T-codes, or specific protocols to enhance embedding quality.
- **Tone**: Clinical, authoritative, and impersonal. Eliminate all conversational filler, speaker attributions, and anecdotes.
- **Inference**: If the transcript contains transcription errors, correct them to the likely intended technical term (e.g., fix "wireshark" to "Wireshark", "end map" to "Nmap").
- **Constraints**: 
    - NO "Indicators of Compromise" (IOC) sections or lists of hashes/IPs.
    - NO ethical warnings, legal disclaimers, or "educational purpose" preambles.
    - NO Markdown tables (use lists/code blocks).
    - NO generic advice; focus on the specific content provided.

### CRITERIA TO DISCARD (Noise/Junk)
When the content meets following criteria, only return "Discard" in your response:
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays.
- **Junk**: Unreadable OCR, corrupted text, or placeholder data.
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.
"""

LLM_WRITEUP_USER_TEMPLATE = """
You are a cybersecurity expert tasked with extracting **technical intelligence** from a raw transcript or document.

Instructions:
- Produce a **detailed and accurate technical write-up** based on the provided text. The write-up must be highly educational, so enhance its content to provide more value when necessary.
- Where the transcript is unclear or has transcription errors, use domain knowledge and contextual clues to **infer the intended technical meaning**.
- Include **step-by-step breakdowns** of any exploits, attack chains, or technical processes described.
- Ignore speaker names, anecdotes, or opinions. Strip away all irrelevant filler.
- Focus on what problem the technical writeup is trying to solve and put it in the real world context.
- **Do NOT include** ethical commentary, disclaimers, copyright, or speaker attribution.
- Use **clear section headers**
- Be smart about extracting all actionable and technical details, even if some reconstruction is required.
- Do not include section of Indicators of Compromise, the report is mainly educational rather than specific

### Criteria to Process (Technical Value)
- Contains **actionable** content: code snippets, exploit payloads, command-line usage.
- Explains specific vulnerabilities (CVEs), architectural internals, or bypass techniques.
- Technical manuals, whitepapers, or detailed tutorials.

### CRITERIA TO 'DELETE' (Noise/Junk)
- **Marketing**: Sales brochures, product advertisements without technical depth.
- **Fluff**: High-level generic summaries, "Importance of Security" essays.
- **Junk**: Unreadable OCR, corrupted text, or placeholder data.
- **CVE without POC**: Description of a vulnerability where explotation steps are not documented or inferred.

If criteria delete is met, just return the 'DELETE' string.

Content:
{content}
"""