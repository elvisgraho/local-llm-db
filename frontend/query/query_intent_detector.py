"""
Query Intent Detection for transparent MITRE-aware RAG.

Analyzes user queries in natural language and maps them to MITRE ATT&CK
tactics, techniques, and tools WITHOUT requiring users to use MITRE terminology.

Research basis:
- Intent recognition via pattern matching (Broder, 2002)
- Verb-object extraction for attack action detection
- Multi-signal intent scoring
"""

import logging
import re
from typing import Dict, List, Set, Optional
from dataclasses import dataclass

from query.mitre_knowledge_graph import (
    get_tactics_for_query,
    get_tools_for_query,
    expand_with_attack_chain,
    get_techniques_for_tools,
    MITRE_TACTICS
)

logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """
    Detected intent from user query.
    """
    # Original query
    query: str

    # Detected MITRE context
    primary_tactics: Set[str]  # Directly detected
    expanded_tactics: Dict[str, float]  # With attack chain (tactic -> boost weight)
    techniques: Set[str]  # T-codes
    tools: Set[str]  # Detected tools

    # Boost keywords for enhanced matching
    boost_keywords: Set[str]

    # Confidence score (0.0 to 1.0)
    confidence: float


# Action verbs that indicate offensive security operations
OFFENSIVE_ACTION_VERBS = {
    "bypass", "evade", "exploit", "dump", "extract", "steal",
    "crack", "break", "disable", "inject", "execute", "run",
    "escalate", "elevate", "pivot", "move", "spread", "hide",
    "obfuscate", "encode", "decrypt", "exfiltrate", "harvest",
    "enumerate", "scan", "discover", "find", "list", "map"
}


def detect_query_intent(query_text: str) -> Optional[QueryIntent]:
    """
    Analyze query and detect MITRE-related intent.

    Args:
        query_text: User's natural language query

    Returns:
        QueryIntent object if intent detected, None otherwise
    """
    query_lower = query_text.lower()

    # Stage 1: Detect tactics from keywords
    primary_tactics = get_tactics_for_query(query_lower)

    # Stage 2: Detect tools
    tools = get_tools_for_query(query_lower)

    # Stage 3: Expand using attack chain relationships
    expanded_tactics = expand_with_attack_chain(primary_tactics)

    # Stage 4: Map tools to techniques
    techniques = get_techniques_for_tools(tools)

    # Stage 5: Extract boost keywords
    boost_keywords = extract_boost_keywords(query_lower, primary_tactics, tools)

    # Calculate confidence
    confidence = calculate_intent_confidence(
        primary_tactics=primary_tactics,
        tools=tools,
        query_text=query_lower
    )

    # Only return intent if we detected something meaningful
    if not primary_tactics and not tools:
        return None

    intent = QueryIntent(
        query=query_text,
        primary_tactics=primary_tactics,
        expanded_tactics=expanded_tactics,
        techniques=techniques,
        tools=tools,
        boost_keywords=boost_keywords,
        confidence=confidence
    )

    # Log detected intent
    if primary_tactics or tools:
        tactics_str = ", ".join(sorted(primary_tactics)[:3])
        tools_str = ", ".join(sorted(tools)[:3])
        logger.info(f"Intent detected (conf={confidence:.2f}): tactics=[{tactics_str}], tools=[{tools_str}]")

    return intent


def extract_boost_keywords(
    query_lower: str,
    tactics: Set[str],
    tools: Set[str]
) -> Set[str]:
    """
    Extract additional keywords that should boost document relevance.

    These are domain-specific terms related to the detected tactics and tools.

    Args:
        query_lower: Lowercased query
        tactics: Detected tactics
        tools: Detected tools

    Returns:
        Set of boost keywords
    """
    boost_keywords = set()

    # Add tool names
    boost_keywords.update(tools)

    # Add tactic-specific boost terms
    for tactic_name in tactics:
        if tactic_name in MITRE_TACTICS:
            tactic_obj = MITRE_TACTICS[tactic_name]
            # Add representative tools for this tactic
            boost_keywords.update(tactic_obj.tools[:5])  # Top 5 tools

    # Extract technical terms (CamelCase, kebab-case, snake_case)
    technical_terms = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query_lower)  # CamelCase
    boost_keywords.update([t.lower() for t in technical_terms])

    # Extract acronyms (2-5 uppercase letters)
    acronyms = re.findall(r'\b[A-Z]{2,5}\b', query_lower.upper())
    boost_keywords.update([a.lower() for a in acronyms])

    return boost_keywords


def calculate_intent_confidence(
    primary_tactics: Set[str],
    tools: Set[str],
    query_text: str
) -> float:
    """
    Calculate confidence score for detected intent.

    Confidence factors:
    - Number of tactics detected
    - Number of tools detected
    - Presence of offensive action verbs
    - Query length and specificity

    Args:
        primary_tactics: Detected tactics
        tools: Detected tools
        query_text: Lowercased query

    Returns:
        Confidence score (0.0 to 1.0)
    """
    confidence = 0.0

    # Factor 1: Tactic detection (up to 0.4)
    if primary_tactics:
        # More tactics = higher confidence (capped at 0.4)
        confidence += min(0.4, len(primary_tactics) * 0.15)

    # Factor 2: Tool detection (up to 0.3)
    if tools:
        confidence += min(0.3, len(tools) * 0.15)

    # Factor 3: Offensive action verbs (up to 0.2)
    words = query_text.split()
    offensive_verbs_found = sum(1 for word in words if word in OFFENSIVE_ACTION_VERBS)
    if offensive_verbs_found:
        confidence += min(0.2, offensive_verbs_found * 0.1)

    # Factor 4: Query specificity (up to 0.1)
    # Longer, more specific queries are more confident
    word_count = len(words)
    if word_count >= 5:
        confidence += 0.1
    elif word_count >= 3:
        confidence += 0.05

    # Cap at 1.0
    return min(confidence, 1.0)


def get_intent_boost_for_document(
    intent: Optional[QueryIntent],
    doc_tactics: List[str],
    doc_techniques: List[str],
    doc_tools: Optional[str] = None
) -> float:
    """
    Calculate intent-based boost score for a document.

    This extends the existing metadata boost with attack chain awareness.

    Args:
        intent: Detected query intent (None if no intent detected)
        doc_tactics: Document's MITRE tactics (from metadata)
        doc_techniques: Document's technique IDs (from metadata)
        doc_tools: Tools mentioned in document (optional)

    Returns:
        Boost score (0.0 to 1.0)
    """
    if not intent:
        return 0.0

    boost = 0.0

    # 1. Primary tactic matching (0.25 boost)
    for doc_tactic in doc_tactics:
        if doc_tactic in intent.primary_tactics:
            boost += 0.25
            break  # One match is enough

    # 2. Expanded tactic matching (weighted boost)
    if boost == 0.0:  # Only check expanded if no primary match
        for doc_tactic in doc_tactics:
            if doc_tactic in intent.expanded_tactics:
                weight = intent.expanded_tactics[doc_tactic]
                boost += 0.25 * weight
                break

    # 3. Technique matching (0.15 boost)
    if intent.techniques and doc_techniques:
        doc_techniques_set = set(doc_techniques)
        if intent.techniques & doc_techniques_set:
            boost += 0.15

    # 4. Tool matching (0.10 boost)
    if doc_tools and intent.tools:
        doc_tools_lower = doc_tools.lower()
        for tool in intent.tools:
            if tool.lower() in doc_tools_lower:
                boost += 0.10
                break

    # Apply confidence scaling
    boost *= intent.confidence

    return min(boost, 1.0)


def format_intent_for_logging(intent: Optional[QueryIntent]) -> str:
    """
    Format intent for debug logging.

    Args:
        intent: Detected intent

    Returns:
        Formatted string for logging
    """
    if not intent:
        return "No intent detected"

    parts = []

    if intent.primary_tactics:
        tactics = ", ".join(sorted(intent.primary_tactics)[:3])
        parts.append(f"tactics=[{tactics}]")

    if intent.tools:
        tools = ", ".join(sorted(intent.tools)[:3])
        parts.append(f"tools=[{tools}]")

    if intent.techniques:
        techniques = ", ".join(sorted(intent.techniques)[:3])
        parts.append(f"techniques=[{techniques}]")

    parts.append(f"conf={intent.confidence:.2f}")

    return " | ".join(parts)
