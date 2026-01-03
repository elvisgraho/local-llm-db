"""
Metadata-Enhanced RAG using latest research techniques.

This module automatically leverages extracted metadata (main_topic, summary_dense,
code_languages, mitre_tactics) to improve retrieval quality without requiring
user configuration.

Research-backed techniques:
1. Query-Metadata Semantic Matching (ColBERT-style late interaction)
2. Intent-Aware MITRE Boosting (transparent attack chain reasoning)
3. Metadata-Aware Re-ranking (LlamaIndex approach)
4. Diversity-Based Selection (MMR with metadata awareness)
5. Context Augmentation (prepending metadata to chunks)
"""

import logging
import re
from typing import List, Tuple, Dict, Any, Optional, Set
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Import intent detection (lazy import to avoid circular dependencies)
_intent_detector = None

def get_intent_detector():
    """Lazy import of intent detector to avoid circular dependencies."""
    global _intent_detector
    if _intent_detector is None:
        from query.query_intent_detector import detect_query_intent, get_intent_boost_for_document
        _intent_detector = {
            'detect': detect_query_intent,
            'boost': get_intent_boost_for_document
        }
    return _intent_detector

# Import MITRE knowledge graph
from query.mitre_knowledge_graph import MITRE_TACTICS

### `CODE_LANGUAGE_KEYWORDS`
CODE_LANGUAGE_KEYWORDS = {
    "PowerShell": [
        "invoke-", "new-object", "$env:", "write-host", "start-process", 
        "-windowstyle hidden", "-enc ", ".downloadstring", "invoke-webrequest", 
        "iwr ", "iex ", "get-content", "powershell.exe"
    ],
    "Python": [
        "import os", "import socket", "import subprocess", "requests.get", 
        "if __name__ ==", "def main():", "os.system", "ctypes.windll", 
        "socket.socket", "sys.argv"
    ],
    "Bash": [
        "#!/bin/bash", "curl ", "wget ", "chmod +x", "grep ", 
        "awk '", "sed '", "echo $", "sudo ", "nc -e", "/dev/tcp", 
        "cat /etc/", "rm -rf"
    ],
    "C": [
        "#include <stdio.h>", "#include <windows.h>", "int main(", 
        "virtualalloc", "memcpy", "malloc", "createprocess", 
        "entrypoint", "socket(", "struct sockaddr"
    ],
    "C++": [
        "#include <iostream>", "std::", "cout <<", "namespace std", 
        "createremotethread", "virtualprotect", "loadlibrary", 
        "getprocaddress", "reinterpret_cast"
    ],
    "C#": [
        "using system;", "namespace ", "class program", "static void main", 
        "[dllimport", "pinvoke", "console.writeline", "system.diagnostics", 
        "var client =", "new tcpclient"
    ],
    "JavaScript": [
        "console.log", "document.location", "window.location", 
        "fetch(", "xmlhttprequest", "require('child_process')", 
        "eval(", "process.env", "fs.readfile"
    ],
    "Go": [
        "package main", "func main()", "fmt.println", "import (", 
        "go func", "syscall.", "os/exec", "net.dial"
    ],
    "Rust": [
        "fn main()", "let mut ", "println!", "cargo ", "impl ", 
        "unsafe {", "std::process", "match "
    ],
    "Ruby": [
        "def ", "require '", "puts ", "gem install", "attr_accessor", 
        "socket.new", "class ", "module "
    ],
    "Java": [
        "public class", "public static void main", "system.out.println", 
        "import java.", "processbuilder", "runtime.getruntime", 
        "new arraylist"
    ],
    "PHP": [
        "<?php", "$_GET", "$_POST", "print_r", "shell_exec", "system(", 
        "passthru", "preg_match", "file_get_contents"
    ],
    "SQL": [
        "select * from", "insert into", "update ", "delete from", "drop table", 
        "union select", "xp_cmdshell", "information_schema", "-- -"
    ],
    "VBA": [
        "sub autoopen", "sub document_open", "dim ", "set ", "createobject", 
        "wscript.shell", "msgbox", "application.run"
    ],
    "Assembly": [
        "mov eax", "xor eax", "push ", "pop ", "ret", "int 0x80", 
        "syscall", "lea ", "jmp short"
    ]
}


def calculate_metadata_relevance_score(
    query_text: str,
    doc: Document,
    query_lower: Optional[str] = None,
    query_intent: Optional[Any] = None
) -> float:
    """
    Calculate semantic relevance between query and document metadata.

    Returns a boost score (0.0 to 1.0) based on:
    - Intent-aware MITRE matching (with attack chain reasoning)
    - Query-topic alignment (main_topic matching)
    - Query-tactic matching (MITRE tactics)
    - Query-language matching (code_languages)
    - Summary keyword overlap

    Research basis:
    - Late interaction scoring (ColBERT) adapted for metadata
    - Multi-hop reasoning for attack chain awareness
    """
    if query_lower is None:
        query_lower = query_text.lower()

    metadata = doc.metadata
    boost_score = 0.0
    matches = []

    # NEW: Intent-aware MITRE boosting (priority boost)
    if query_intent is not None:
        detector = get_intent_detector()

        # Extract document tactics and techniques
        doc_tactics_str = metadata.get('mitre_tactics', '')
        doc_techniques_str = metadata.get('mitre_technique_primary_ids', '')

        # Parse comma-separated strings to lists
        doc_tactics = [t.strip().split('(')[0].strip() for t in doc_tactics_str.split(',') if t.strip()]
        doc_techniques = [t.strip() for t in doc_techniques_str.split(',') if t.strip()]

        # Calculate intent boost
        intent_boost = detector['boost'](query_intent, doc_tactics, doc_techniques)

        if intent_boost > 0:
            boost_score += intent_boost
            matches.append(f"intent:{intent_boost:.2f}")
            # If we have strong intent match, we can return early
            if intent_boost >= 0.4:
                logger.debug(f"Strong intent match: {intent_boost:.2f} - {', '.join(matches)}")
                return min(boost_score, 1.0)

    # 1. Main Topic Matching (0.3 weight - highest)
    main_topic = metadata.get('main_topic', '')
    if main_topic and isinstance(main_topic, str):
        topic_lower = main_topic.lower()
        # Check for topic keywords in query
        topic_words = set(topic_lower.split())
        query_words = set(query_lower.split())

        overlap = topic_words & query_words
        if overlap:
            boost_score += 0.3
            matches.append(f"topic:{main_topic}")

    # 2. MITRE Tactic Matching (0.25 weight)
    mitre_tactics = metadata.get('mitre_tactics', '')
    if mitre_tactics:
        # Handle both list and flattened string formats
        if isinstance(mitre_tactics, str):
            tactics_list = [t.strip() for t in mitre_tactics.split(',')]
        else:
            tactics_list = mitre_tactics if isinstance(mitre_tactics, list) else []

        for tactic in tactics_list:
            tactic_clean = tactic.split('(')[0].strip()  # Remove (TA0001) suffix

            # Get keywords from knowledge graph
            if tactic_clean in MITRE_TACTICS:
                keywords = MITRE_TACTICS[tactic_clean].keywords
            else:
                keywords = []

            for keyword in keywords:
                if keyword in query_lower:
                    boost_score += 0.25
                    matches.append(f"tactic:{tactic_clean}")
                    break  # One match per tactic

            # Also check if tactic name itself is in query
            if tactic_clean.lower() in query_lower:
                boost_score += 0.25
                matches.append(f"tactic:{tactic_clean}")

    # 3. Code Language Matching (0.2 weight)
    code_languages = metadata.get('code_languages', '')
    if code_languages:
        if isinstance(code_languages, str):
            langs_list = [l.strip() for l in code_languages.split(',')]
        else:
            langs_list = code_languages if isinstance(code_languages, list) else []

        for lang in langs_list:
            keywords = CODE_LANGUAGE_KEYWORDS.get(lang, [])

            for keyword in keywords:
                if keyword in query_lower:
                    boost_score += 0.2
                    matches.append(f"lang:{lang}")
                    break

            # Direct language mention
            if lang.lower() in query_lower:
                boost_score += 0.2
                matches.append(f"lang:{lang}")

    # 4. Summary Dense Keyword Overlap (0.15 weight)
    summary_dense = metadata.get('summary_dense', '')
    if summary_dense and isinstance(summary_dense, str):
        # Extract important keywords (skip common words)
        summary_words = set(re.findall(r'\b\w{4,}\b', summary_dense.lower()))
        query_words = set(re.findall(r'\b\w{4,}\b', query_lower))

        overlap = summary_words & query_words
        if overlap:
            # Score based on overlap ratio
            overlap_ratio = len(overlap) / max(len(query_words), 1)
            boost_score += min(0.15, overlap_ratio * 0.5)
            matches.append(f"summary:{len(overlap)} keywords")

    # 5. MITRE Technique ID Matching (0.1 weight)
    technique_ids = metadata.get('mitre_technique_primary_ids', '')
    if technique_ids:
        if isinstance(technique_ids, str):
            ids_list = [i.strip() for i in technique_ids.split(',')]
        else:
            ids_list = technique_ids if isinstance(technique_ids, list) else []

        for tid in ids_list:
            if tid.lower() in query_lower:
                boost_score += 0.1
                matches.append(f"technique:{tid}")

    # Cap at 1.0
    boost_score = min(boost_score, 1.0)

    if boost_score > 0:
        logger.debug(f"Metadata boost: {boost_score:.2f} - {', '.join(matches[:3])}")

    return boost_score


def rerank_with_metadata_awareness(
    query_text: str,
    results: List[Tuple[Document, float]],
    k: int,
    alpha: float = 0.7,
    query_intent: Optional[Any] = None
) -> List[Tuple[Document, float]]:
    """
    Hybrid re-ranking combining original scores with metadata + intent relevance.

    Final score = (alpha * original_score) + ((1-alpha) * metadata_boost)

    Args:
        query_text: User query
        results: List of (Document, score) tuples from retrieval
        k: Number of results to return
        alpha: Weight for original score (0.7 = 70% original, 30% metadata)
        query_intent: Detected MITRE intent (optional, for attack chain awareness)

    Research basis:
    - LlamaIndex hybrid ranking + metadata signals
    - Multi-hop reasoning for attack chains
    """
    if not results:
        return []

    query_lower = query_text.lower()
    reranked = []

    # Normalize original scores to 0-1 range
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score if max_score > min_score else 1.0

    for doc, original_score in results:
        # Normalize original score
        norm_score = (original_score - min_score) / score_range if score_range > 0 else 0.5

        # Calculate metadata boost (now intent-aware)
        metadata_boost = calculate_metadata_relevance_score(
            query_text, doc, query_lower, query_intent
        )

        # Hybrid score
        final_score = (alpha * norm_score) + ((1 - alpha) * metadata_boost)

        reranked.append((doc, final_score))

    # Sort by final score
    reranked.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Metadata-aware reranking completed. Top score: {reranked[0][1]:.3f}")

    return reranked[:k]


def augment_context_with_metadata(doc: Document) -> str:
    """
    Prepend metadata summary to document content for richer context.

    Research basis: Contextual augmentation improves LLM understanding.
    """
    metadata = doc.metadata
    metadata_lines = []

    # Add structured metadata header
    main_topic = metadata.get('main_topic')
    if main_topic and isinstance(main_topic, str):
        metadata_lines.append(f"[Topic: {main_topic}]")

    mitre_tactics = metadata.get('mitre_tactics')
    if mitre_tactics and isinstance(mitre_tactics, str) and mitre_tactics.strip():
        metadata_lines.append(f"[MITRE Tactics: {mitre_tactics}]")

    code_languages = metadata.get('code_languages')
    if code_languages and isinstance(code_languages, str) and code_languages.strip():
        metadata_lines.append(f"[Languages: {code_languages}]")

    summary_dense = metadata.get('summary_dense')
    if summary_dense and isinstance(summary_dense, str):
        # Truncate if too long
        summary_short = summary_dense[:200] + "..." if len(summary_dense) > 200 else summary_dense
        metadata_lines.append(f"[Summary: {summary_short}]")

    if metadata_lines:
        metadata_header = "\n".join(metadata_lines)
        return f"{metadata_header}\n\n{doc.page_content}"

    return doc.page_content


def select_diverse_results(
    results: List[Tuple[Document, float]],
    k: int,
    diversity_weight: float = 0.3
) -> List[Tuple[Document, float]]:
    """
    Maximal Marginal Relevance (MMR) selection with metadata diversity.

    Balances relevance with diversity across:
    - Different MITRE tactics
    - Different code languages
    - Different main topics

    Research basis: MMR (Carbonell & Goldstein) + metadata-aware diversity.
    """
    if len(results) <= k:
        return results

    selected = []
    selected_tactics: Set[str] = set()
    selected_languages: Set[str] = set()
    selected_topics: Set[str] = set()

    # Always pick the top result
    selected.append(results[0])

    # Extract metadata from top result
    top_meta = results[0][0].metadata
    if top_meta.get('mitre_tactics'):
        selected_tactics.update(str(top_meta['mitre_tactics']).split(','))
    if top_meta.get('code_languages'):
        selected_languages.update(str(top_meta['code_languages']).split(','))
    if top_meta.get('main_topic'):
        selected_topics.add(str(top_meta['main_topic']))

    # Iteratively select remaining k-1 documents
    candidates = results[1:]

    for _ in range(k - 1):
        if not candidates:
            break

        best_score = -1
        best_idx = 0

        for idx, (doc, relevance_score) in enumerate(candidates):
            # Calculate diversity bonus
            diversity_bonus = 0.0
            meta = doc.metadata

            # Check if document introduces new tactics
            doc_tactics = str(meta.get('mitre_tactics', '')).split(',')
            new_tactics = set(t.strip() for t in doc_tactics if t.strip()) - selected_tactics
            if new_tactics:
                diversity_bonus += 0.3

            # Check if document introduces new languages
            doc_langs = str(meta.get('code_languages', '')).split(',')
            new_langs = set(l.strip() for l in doc_langs if l.strip()) - selected_languages
            if new_langs:
                diversity_bonus += 0.2

            # Check if document introduces new topic
            doc_topic = meta.get('main_topic')
            if doc_topic and doc_topic not in selected_topics:
                diversity_bonus += 0.2

            # Combined score
            combined_score = (1 - diversity_weight) * relevance_score + diversity_weight * diversity_bonus

            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx

        # Add best candidate
        selected_doc, selected_score = candidates[best_idx]
        selected.append((selected_doc, selected_score))

        # Update diversity tracking
        meta = selected_doc.metadata
        if meta.get('mitre_tactics'):
            selected_tactics.update(str(meta['mitre_tactics']).split(','))
        if meta.get('code_languages'):
            selected_languages.update(str(meta['code_languages']).split(','))
        if meta.get('main_topic'):
            selected_topics.add(str(meta['main_topic']))

        # Remove selected candidate
        candidates.pop(best_idx)

    logger.info(f"Diversity selection: {len(selected_tactics)} tactics, {len(selected_languages)} languages, {len(selected_topics)} topics")

    return selected
