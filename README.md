# Local LLM Data Bank

A comprehensive implementation of different RAG (Retrieval Augmented Generation) approaches for document processing and querying.

## Features

- Multiple RAG implementations:
  - Standard RAG with Chroma vectorstore
  - GraphRAG for hierarchical document relationships
  - Knowledge Augmented Graph (KAG) for semantic relationships
  - LightRAG for efficient document retrieval
- Support for multiple document types (PDF, TXT, MD)
- Local LLM integration via LM Studio
- Web interface for document querying
- Document metadata extraction and processing

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your documents in the `training/data` directory

3. Run the desired RAG implementation:
```bash
# Standard RAG
python training/populate_rag.py

# GraphRAG
python training/populate_graphrag.py

# KAG
python training/populate_kag.py

# LightRAG
python training/populate_lightrag.py
```

4. Start the web interface:
```bash
uvicorn query.main:app --reload
```

## Project Structure

- `training/`: Document processing and RAG implementations
- `query/`: Query interface and API endpoints
- `frontend/`: Web interface files

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies
