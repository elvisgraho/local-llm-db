"""
RAG (Retrieval Augmented Generation) Implementation

This module implements a standard RAG approach using Chroma vectorstore for document
retrieval. Key features:
1. Document chunking and embedding
2. Vector similarity search
3. Integration with LM Studio for local LLM inference
4. Basic question-answering capabilities

The standard implementation provides:
- Efficient document retrieval
- Semantic search capabilities
- Simple query interface
- Integration with local LLM
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function
from extract_metadata_llm import extract_metadata_llm
from query.database_paths import CHROMA_PATH, RAG_DB_DIR
from load_documents import load_documents, extract_metadata, process_single_file
import re
import argparse
from dataclasses import dataclass
from typing import List, Optional
import requests
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGResponse:
    """Container for RAG query responses."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class RAGSystem:
    """Main RAG system class for document processing and querying."""
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200, batch_size: int = 32):
        """Initialize the RAG system with configurable parameters."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.embeddings = get_embedding_function()
        self.vectorstore = None
        self._load_vectorstore()
        self._setup_backup()
    
    def _setup_backup(self) -> None:
        """Setup backup directory for vectorstore."""
        self.backup_dir = Path(CHROMA_PATH).parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _backup_vectorstore(self) -> None:
        """Create a backup of the vectorstore."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.backup_dir / f"chroma_backup_{timestamp}"
            shutil.copytree(CHROMA_PATH, backup_path)
            logger.info(f"Created vectorstore backup at {backup_path}")
        except Exception as e:
            logger.error(f"Failed to create vectorstore backup: {str(e)}")
    
    def _load_vectorstore(self) -> None:
        """Load or create the vectorstore with error handling."""
        vectorstore_path = str(CHROMA_PATH)
        try:
            if os.path.exists(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "chroma.sqlite3")):
                logger.info("Loading existing vectorstore")
                self.vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=self.embeddings)
            else:
                logger.info("Creating new vectorstore")
                self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=vectorstore_path)
        except Exception as e:
            logger.error(f"Error loading vectorstore: {str(e)}")
            self._backup_vectorstore()  # Backup before creating new
            self.vectorstore = Chroma(embedding_function=self.embeddings, persist_directory=vectorstore_path)
    
    def process_documents(self, documents: List[Document]) -> None:
        """Process documents in batches for better performance."""
        try:
            # Split documents into chunks
            all_chunks = []
            for doc in documents:
                chunks = split_document(doc)
                all_chunks.extend(chunks)
            
            # Process chunks in batches
            for i in range(0, len(all_chunks), self.batch_size):
                batch = all_chunks[i:i + self.batch_size]
                self._process_chunk_batch(batch)
                
            # Save after processing all documents
            self.vectorstore.persist()
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            self._backup_vectorstore()
            raise
    
    def _process_chunk_batch(self, chunks: List[Document]) -> None:
        """Process a batch of chunks efficiently."""
        try:
            # Get embeddings for the batch
            texts = [chunk.page_content for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
        except Exception as e:
            logger.error(f"Error processing chunk batch: {str(e)}")
            raise
    
    def query(self, query: str, k: int = 4, temperature: float = 0.7) -> RAGResponse:
        """
        Process a query and return relevant information with improved error handling.
        
        Args:
            query: The query string
            k: Number of relevant documents to retrieve
            temperature: Temperature for LLM response generation
            
        Returns:
            RAGResponse containing answer and sources
        """
        try:
            if not self.vectorstore:
                self._load_vectorstore()
            
            # Retrieve relevant documents
            docs = self.vectorstore.similarity_search(query, k=k)
            
            # Extract sources and content
            sources = []
            context = []
            for doc in docs:
                source = {
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": doc.metadata.get("relevance_score", 0.0)
                }
                sources.append(source)
                context.append(doc.page_content)
            
            # Generate answer using LLM
            prompt = self._create_prompt(query, context)
            answer = self._get_llm_response(prompt, temperature)
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                metadata={
                    "query": query,
                    "k": k,
                    "temperature": temperature,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your query.",
                sources=[],
                metadata={"error": str(e)}
            )
    
    def _get_llm_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Get response from local LLM with improved error handling."""
        try:
            headers = {"Content-Type": "application/json"}
            payload = {
                "model": "local-model",  # Update with your model name
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 1000  # Limit response length
            }
            
            response = requests.post(
                "http://localhost:1234/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30  # Add timeout
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
            
        except requests.Timeout:
            logger.error("LLM request timed out")
            return "I apologize, but the request timed out. Please try again."
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {str(e)}")
            return "I apologize, but I encountered an error while processing your query."
        except Exception as e:
            logger.error(f"Unexpected error in LLM response: {str(e)}")
            return "I apologize, but I encountered an unexpected error."

def split_document(doc: Document) -> List[Document]:
    """Split a single document into chunks with improved parameters for security documentation."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
        separators=[
            "\n\n## ",  # Main headers
            "\n\n### ",  # Subheaders
            "\n\n#### ",  # Sub-subheaders
            "\n```",  # Code blocks
            "\n\n",     # Double newlines
            "\n**",
            "\n",       # Single newlines
            " ",        # Spaces
            ""         # No separator
        ],
        keep_separator=True
    )
    
    try:
        # Pre-process security-specific content
        content = doc.page_content
        
        # Normalize bullet points and numbered lists
        content = re.sub(r'^\s*[-•]\s*', '• ', content, flags=re.MULTILINE)
        content = re.sub(r'^\s*(\d+\.)\s*', r'\1 ', content, flags=re.MULTILINE)
        
        # Ensure consistent formatting for code blocks
        content = re.sub(r'```(\w+)?\n', r'```\n', content)
        
        # Create a copy of the document with processed content
        processed_doc = Document(
            page_content=content,
            metadata=doc.metadata.copy()  # Make a copy of metadata
        )
        
        # Split the document
        doc_chunks = text_splitter.split_documents([processed_doc])
        
        # Process each chunk with LLM metadata extraction
        processed_chunks = []
        for chunk in doc_chunks:
            # Ensure source metadata is preserved
            if not chunk.metadata.get("source") and doc.metadata.get("source"):
                chunk.metadata["source"] = doc.metadata["source"]
            
            # Extract LLM-based metadata
            llm_metadata = extract_metadata_llm(chunk.page_content)
            chunk.metadata.update(llm_metadata)
            
            # Add file-based metadata
            chunk.metadata.update(extract_metadata(chunk.metadata.get("source", "")))
            
            processed_chunks.append(chunk)
            
        return processed_chunks
            
    except Exception as e:
        logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
        return []

def process_document(doc: Document) -> None:
    """Process a single document and update the vectorstore."""
    embeddings = get_embedding_function()
    
    # Ensure source metadata exists
    if not doc.metadata.get("source"):
        logger.warning(f"Document missing source metadata: {doc.metadata}")
        return
    
    # Split document into chunks
    chunks = split_document(doc)
    if not chunks:
        logger.warning(f"No valid chunks created for document: {doc.metadata.get('source', 'unknown')}")
        return
    
    # Initialize or update vectorstore
    try:
        vectorstore_path = str(CHROMA_PATH)
        if os.path.exists(vectorstore_path) and os.path.exists(os.path.join(vectorstore_path, "chroma.sqlite3")):
            logger.info("Loading existing vectorstore")
            vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embeddings)
            
            # Check for duplicates before adding
            existing_docs = vectorstore.get()
            existing_sources = set(doc.metadata.get("source", "") for doc in existing_docs["documents"])
            new_chunks = []
            
            for chunk in chunks:
                # Ensure chunk has source metadata
                if not chunk.metadata.get("source"):
                    chunk.metadata["source"] = doc.metadata["source"]
                
                # Only add if source is not in existing sources
                if chunk.metadata["source"] not in existing_sources:
                    new_chunks.append(chunk)
            
            if new_chunks:
                vectorstore.add_documents(new_chunks)
                logger.info(f"Added {len(new_chunks)} new chunks to vectorstore")
            else:
                logger.info("No new chunks to add - all chunks already exist in vectorstore")
        else:
            logger.info("Creating new vectorstore")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=vectorstore_path)
    except Exception as e:
        logger.warning(f"Failed to load existing vectorstore: {str(e)}")
        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=vectorstore_path)
    
    # Save after each document processing
    os.makedirs(RAG_DB_DIR, exist_ok=True)
    vectorstore.persist_directory = vectorstore_path  # Set persist directory
    logger.debug(f"Updated vectorstore saved to {vectorstore_path}")

def clear_vectorstore():
    """Clear the RAG database."""
    if os.path.exists(RAG_DB_DIR):
        for root, dirs, files in os.walk(RAG_DB_DIR, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        logger.info("Cleared RAG database")

def process_file_to_vectorstore(file_path: Path) -> None:
    """Process a single file and update the vectorstore."""
    try:
        # Load documents from the file
        documents = process_single_file(file_path)
        if not documents:
            logger.warning(f"No valid documents found in file: {file_path.name}")
            return
            
        # Process each document
        for doc in documents:
            try:
                process_document(doc)
            except Exception as e:
                logger.error(f"Error processing document in {file_path.name}: {str(e)}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {str(e)}")

def main():
    """Main function to populate the RAG database file by file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database before populating")
    args = parser.parse_args()
    
    if args.reset:
        clear_vectorstore()
        logger.info("Cleared existing RAG database")

    try:
        # Get all documents using load_documents functionality
        all_documents = load_documents()
        
        if not all_documents:
            logger.error("No valid documents found to process")
            return
            
        total_docs = len(all_documents)
        processed_docs = 0
        failed_docs = 0
            
        # Process files one by one
        for doc in tqdm(all_documents, desc="Processing documents", total=total_docs):
            try:
                logger.info(f"Processing document {processed_docs + 1}/{total_docs}")
                # Use process_file_to_vectorstore for RAG implementation
                process_file_to_vectorstore(Path(doc.metadata.get("source", "")))
                processed_docs += 1
                    
            except Exception as e:
                logger.error(f"Error processing document from {doc.metadata.get('source', 'unknown')}: {str(e)}")
                failed_docs += 1
                continue
        
        # Log final statistics
        logger.info(f"RAG database population completed:")
        logger.info(f"- Total documents: {total_docs}")
        logger.info(f"- Successfully processed documents: {processed_docs}")
        logger.info(f"- Failed documents: {failed_docs}")
        
        if processed_docs == 0:
            logger.error("No documents were successfully processed")
        else:
            logger.info("Successfully populated RAG database")
        
    except Exception as e:
        logger.error(f"Error in database population: {str(e)}")
        raise

if __name__ == "__main__":
    main()
