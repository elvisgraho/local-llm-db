"""
Centralized LLM client for all training operations.
Handles API communication, response parsing, and error handling.
"""
import json
import re
import sys
import logging
import unicodedata
import requests
from typing import Dict, Any, Optional

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

THINKING_PATTERN = re.compile(
    r'<(?P<tag>thinking|thought|reasoning|think)\b[^>]*>.*?</(?P=tag)>|'
    r'\s*\[/?(?:thinking|thought|reasoning|think)\b[^\]]*\]\s*|'
    r'\s*\((?:thinking|thought|reasoning|think)\b[^)]*\)\s*',
    flags=re.DOTALL | re.IGNORECASE
)
JSON_BLOCK_PATTERN = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)

session = requests.Session()


class LLMClient:
    """Centralized client for LLM API interactions."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        ocr_model_name: Optional[str] = None
    ):
        self.api_url = api_url or config.llm.api_url
        self.model_name = model_name or config.llm.model_name
        self.ocr_model_name = ocr_model_name or config.llm.ocr_model_name

    def get_response(
        self,
        prompt_text: str,
        system_content: Optional[str] = None,
        temperature: float = 0.3,
        model_name: Optional[str] = None
    ) -> str:
        """Get response from LLM API."""
        headers = {"Content-Type": "application/json"}

        base = self.api_url.rstrip('/')
        if not base.endswith("/v1/chat/completions"):
            url = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"
        else:
            url = base

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        messages.append({"role": "user", "content": prompt_text})

        payload = {
            "model": model_name or self.model_name,
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

    @staticmethod
    def clean_and_parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        if not text:
            return {}

        text = unicodedata.normalize('NFKC', text)
        text = text.replace("\\u2010", "-").replace("\\u2011", "-") \
                   .replace("\\u2013", "-").replace("\\u2014", "-") \
                   .replace("\\u00a0", " ")

        text = THINKING_PATTERN.sub('', text).strip()

        json_str = ""
        match = JSON_BLOCK_PATTERN.search(text)
        if match:
            json_str = match.group(1)

        if not json_str:
            start_match = re.search(r'({|\[)', text)
            end_match = re.search(r'(}|\])', text[::-1])

            if start_match and end_match:
                start_pos = start_match.start(1)
                end_pos = len(text) - end_match.end(1)

                if end_pos > start_pos:
                    json_str = text[start_pos : end_pos + 1]
                else:
                    return {}
            else:
                return {}

        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

        try:
            return json.loads(json_str)
        except Exception:
            return {}

    @staticmethod
    def extract_text_parts(text: str, part_size: int = 2000, part_count: int = 20) -> str:
        """Extract uniformly spaced parts from text for sampling."""
        L = len(text)

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


_default_client = None


def get_llm_client() -> LLMClient:
    """Get singleton LLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def get_llm_response(
    prompt_text: str,
    system_content: Optional[str] = None,
    temperature: float = 0.3,
    model_name: Optional[str] = None
) -> str:
    """Convenience function for backward compatibility."""
    return get_llm_client().get_response(prompt_text, system_content, temperature, model_name)


def clean_and_parse_json(text: str) -> Dict[str, Any]:
    """Convenience function for backward compatibility."""
    return LLMClient.clean_and_parse_json(text)


def extract_text_parts(text: str, part_size: int = 2000, part_count: int = 20) -> str:
    """Convenience function for backward compatibility."""
    return LLMClient.extract_text_parts(text, part_size, part_count)
