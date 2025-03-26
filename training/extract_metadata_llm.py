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
from backend.global_vars import LOCAL_MAIN_MODEL, LOCAL_LLM_API_URL

class DocumentMetadata(BaseModel):
    """Schema for document metadata extracted by LLM."""
    content_type: str = Field(description="Type of content (e.g., 'technical_doc', 'code_example', 'explanation')")
    main_topic: str = Field(description="Main topic or subject of the content")
    key_concepts: list[str] = Field(description="Key concepts or terms mentioned")
    has_code: bool = Field(description="Whether the content contains code examples")
    has_diagrams: bool = Field(description="Whether the content contains diagrams or visual elements")
    complexity_level: str = Field(description="Complexity level (e.g., 'beginner', 'intermediate', 'advanced')")
    is_tutorial: bool = Field(description="Whether the content is tutorial-like")
    is_reference: bool = Field(description="Whether the content is reference material")

def get_metadata_extraction_prompt() -> str:
    """Create a prompt template for metadata extraction."""
    return """You are a metadata extraction specialist. Your task is to analyze the given text and extract concise, keyword-based metadata that would be useful for other LLMs to understand the content.

Text to analyze:
{text}

{format_instructions}

IMPORTANT: Provide ONLY short, keyword-based responses. Do not write full sentences or explanations.

For each field:
- content_type: Single word or hyphenated term (e.g., "technical-doc", "code-example")
- main_topic: 1-3 words maximum
- key_concepts: List of single words or hyphenated terms
- has_code: true/false
- has_diagrams: true/false
- complexity_level: One word (beginner/intermediate/advanced)
- is_tutorial: true/false
- is_reference: true/false

Example format:
{
    "content_type": "technical-doc",
    "main_topic": "python-async",
    "key_concepts": ["async-await", "coroutines", "event-loop"],
    "has_code": true,
    "has_diagrams": false,
    "complexity_level": "intermediate",
    "is_tutorial": true,
    "is_reference": false
}

Provide only the metadata in the specified format."""

def _get_llm_response(prompt: str) -> str:
    """Helper function to get response from local LLM."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LOCAL_MAIN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3  # Lower temperature for more focused metadata extraction
    }
    response = requests.post(LOCAL_LLM_API_URL, json=payload, headers=headers)
    response_data = response.json()
    content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    # Remove <think> tags and their content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    return content.strip()

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
        # Format the prompt with the text and format instructions
        formatted_prompt = prompt.format(
            text=text,
            format_instructions=parser.get_format_instructions()
        )
        
        # Get response from local LLM
        response_text = _get_llm_response(formatted_prompt)
        
        # Try to parse as JSON first
        try:
            metadata_dict = json.loads(response_text)
            # Validate the JSON structure against our model
            metadata = DocumentMetadata(**metadata_dict)
            return metadata.model_dump()
        except json.JSONDecodeError:
            # If not JSON, try parsing with Pydantic directly
            metadata = parser.parse(response_text)
            return metadata.model_dump()
            
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return {}

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