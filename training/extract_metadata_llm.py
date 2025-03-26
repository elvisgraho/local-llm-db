"""
LLM-based metadata extraction for document chunks.

This module provides functionality to extract meaningful metadata from document chunks
using LLM analysis. It focuses on identifying key aspects of the content while
ignoring unnecessary details.
"""

from typing import Dict, Any
import json
import requests
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.schema import Document
import re
import sys
from pathlib import Path
import logging

# Add parent directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir.parent))

from backend.global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL

logger = logging.getLogger(__name__)

class DocumentMetadata(BaseModel):
    """Schema for document metadata extracted by LLM."""
    # Basic content metadata (Chroma-compatible)
    content_type: str = Field(description="Type of content (e.g., 'technical_doc', 'code_example', 'explanation')")
    main_topic: str = Field(description="Main topic or subject of the content")
    key_concepts: str = Field(description="Comma-separated list of key concepts or terms")
    has_code: bool = Field(description="Whether the content contains code examples")
    has_instructions: bool = Field(description="Whether the content contains instructions, commands, or step-by-step guidance")
    is_tutorial: bool = Field(description="Whether the content is tutorial-like")
    is_reference: bool = Field(description="Whether the content is reference material")
    
    # Graph-specific metadata
    section_type: str = Field(description="Type of section (e.g., 'scenario', 'mitigation', 'impact', 'explanation')")

def get_metadata_extraction_prompt() -> str:
    """Create a prompt template for metadata extraction."""
    return """You are a metadata extraction specialist. Your task is to analyze the given text and extract concise, keyword-based metadata that would be useful for both document retrieval and graph-based analysis.

Text to analyze:
{text}

{format_instructions}

IMPORTANT: Provide ONLY short, keyword-based responses. Do not write full sentences or explanations.

For each field:
- content_type: Single word or hyphenated term (e.g., "technical-doc", "code-example")
- main_topic: 1-3 words maximum
- key_concepts: Comma-separated list of single words or hyphenated terms
- has_code: true/false
- has_instructions: true/false (contains commands, steps, or guidance)
- is_tutorial: true/false
- is_reference: true/false
- section_type: One word (scenario/mitigation/impact/explanation)

Example format:
{
    "content_type": "technical-doc",
    "main_topic": "python-async",
    "key_concepts": "async-await, coroutines, event-loop",
    "has_code": true,
    "has_instructions": true,
    "is_tutorial": true,
    "is_reference": false,
    "section_type": "explanation"
}

Provide only the metadata in the specified format."""

def _get_llm_response(prompt: str) -> str:
    """Helper function to get response from local LLM."""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": LOCAL_MAIN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3  # Lower temperature for more focused metadata extraction
        }
        
        response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        response_data = response.json()
        content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        if not content:
            return ""
            
        # Remove <think> tags and their content
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Clean up the response text
        content = content.strip()
        
        # Try to extract JSON content if it exists
        json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        else:
            # If no JSON block found, try to find JSON-like content
            json_match = re.search(r'({.*?})', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
        
        # Remove any markdown code block markers
        content = re.sub(r'^```json\s*|\s*```$', '', content, flags=re.MULTILINE)
        # Remove any leading/trailing whitespace
        content = content.strip()
        
        return content
        
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"LLM error: {str(e)}")
        return ""

def validate_metadata_field(field_name: str, value: Any) -> Any:
    """
    Validate a single metadata field and return a safe default if validation fails.
    
    Args:
        field_name: Name of the field to validate
        value: Value to validate
        
    Returns:
        Validated value or safe default
    """
    try:
        # Get the field type from the DocumentMetadata model
        field_type = DocumentMetadata.model_fields[field_name].annotation
        
        # Convert value to the expected type
        if field_type == str:
            # Handle lists by joining with commas
            if isinstance(value, list):
                return ", ".join(str(item) for item in value)
            return str(value)
        elif field_type == bool:
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes')
            return bool(value)
        elif field_type == int:
            if isinstance(value, str):
                try:
                    return int(value)
                except ValueError:
                    return 1
            return int(value)
        else:
            return value
    except Exception as e:
        logger.error(f"Error validating field {field_name}: {str(e)}")
        # Return safe defaults based on field type
        if field_name == "content_type":
            return "unknown"
        elif field_name == "main_topic":
            return "unknown"
        elif field_name == "key_concepts":
            return ""
        elif field_name in ["has_code", "has_instructions", "is_tutorial", "is_reference"]:
            return False
        elif field_name == "complexity_level":
            return "unknown"
        elif field_name == "section_type":
            return "unknown"
        elif field_name == "parent_section":
            return "none"
        elif field_name == "section_level":
            return 1
        else:
            return None

def extract_metadata_llm(text: str) -> Dict[str, Any]:
    """
    Extract metadata from text using local LLM.
    
    Args:
        text: The text to analyze
    
    Returns:
        Dictionary containing extracted metadata
    """
    parser = PydanticOutputParser(pydantic_object=DocumentMetadata)
    prompt = get_metadata_extraction_prompt()
    
    try:
        # Get format instructions
        format_instructions = parser.get_format_instructions()
        
        # Create the prompt using string concatenation instead of format()
        formatted_prompt = prompt.replace("{text}", text).replace("{format_instructions}", format_instructions)
        
        # Get response from local LLM
        response_text = _get_llm_response(formatted_prompt)
        
        if not response_text:
            return {
                "content_type": "unknown",
                "main_topic": "unknown",
                "key_concepts": "",
                "has_code": False,
                "has_instructions": False,
                "is_tutorial": False,
                "is_reference": False,
                "section_type": "unknown"
            }
        
        # Try to parse as JSON first
        try:
            # Clean up the response text before parsing
            response_text = response_text.strip()
            
            # If the response is incomplete JSON, try to complete it
            if response_text.startswith('{') and not response_text.endswith('}'):
                # Add missing fields with default values
                response_text = response_text.rstrip(',') + '}'
            
            if response_text.startswith('{') and response_text.endswith('}'):
                metadata_dict = json.loads(response_text)
                # Validate each field individually
                validated_metadata = {}
                for field_name, value in metadata_dict.items():
                    if field_name in DocumentMetadata.model_fields:
                        validated_metadata[field_name] = validate_metadata_field(field_name, value)
                return validated_metadata
            else:
                # If not valid JSON, try parsing with Pydantic
                metadata = parser.parse(response_text)
                return metadata.model_dump()
        except json.JSONDecodeError:
            # If JSON parsing fails, try parsing with Pydantic
            try:
                metadata = parser.parse(response_text)
                return metadata.model_dump()
            except Exception:
                # Return default metadata if parsing fails
                return {
                    "content_type": "unknown",
                    "main_topic": "unknown",
                    "key_concepts": "",
                    "has_code": False,
                    "has_instructions": False,
                    "is_tutorial": False,
                    "is_reference": False,
                    "section_type": "unknown"
                }
            
    except Exception as e:
        logger.error(f"Error extracting metadata: {str(e)}")
        # Return default metadata if extraction fails
        return {
            "content_type": "unknown",
            "main_topic": "unknown",
            "key_concepts": "",
            "has_code": False,
            "has_instructions": False,
            "is_tutorial": False,
            "is_reference": False,
            "section_type": "unknown"
        }

def add_metadata_to_document(doc: Document, max_chars: int = 5000) -> Document:
    """
    Add LLM-extracted metadata to a document.
    
    Args:
        doc: The document to process
        max_chars: Maximum number of characters to analyze for metadata
    
    Returns:
        Document with added metadata
    """
    # Take first max_chars of content for analysis
    preview_text = doc.page_content[:max_chars]
    
    # Extract metadata
    metadata = extract_metadata_llm(preview_text)
    
    # Add metadata to document
    if "llm_metadata" not in doc.metadata:
        doc.metadata["llm_metadata"] = {}
    doc.metadata["llm_metadata"].update(metadata)
    
    return doc 