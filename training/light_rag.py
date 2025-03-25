import os
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    DirectoryLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from get_embedding_function import get_embedding_function
import requests
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_PATH = "data"
VECTORSTORE_PATH = "vectorstore"
LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"

class LMStudioLLM(LLM):
    """Custom LLM class to use LM Studio's local API."""
    
    def _clean_response(self, response: str) -> str:
        """Clean the response by removing reasoning artifacts and extra whitespace."""
        # Remove <think> tags and their content
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        # Remove any other potential reasoning artifacts
        response = re.sub(r'<reason>.*?</reason>', '', response, flags=re.DOTALL)
        response = re.sub(r'<thought>.*?</thought>', '', response, flags=re.DOTALL)
        # Clean up extra whitespace
        response = re.sub(r'\s+', ' ', response)
        return response.strip()
    
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "deepseek-r1-distill-qwen-14b",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        try:
            response = requests.post(LM_STUDIO_API_URL, json=payload, headers=headers)
            response_data = response.json()
            raw_response = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return self._clean_response(raw_response)
        except Exception as e:
            logger.error(f"Error calling LM Studio API: {str(e)}")
            return "Sorry, I encountered an error while processing your query."
    
    @property
    def _llm_type(self) -> str:
        return "lm_studio"

def preprocess_text(text: str) -> str:
    """Clean and normalize text before chunking."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\r\n', '\n')
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def load_documents() -> List[Document]:
    """Load documents from various file types."""
    loaders = {
        "**/*.pdf": PyPDFDirectoryLoader,
        "**/*.txt": TextLoader,
        "**/*.md": UnstructuredMarkdownLoader
    }
    
    all_documents = []
    for glob_pattern, loader_class in loaders.items():
        try:
            loader = DirectoryLoader(
                DATA_PATH,
                glob=glob_pattern,
                loader_class=loader_class,
                show_progress=True
            )
            documents = loader.load()
            
            for doc in documents:
                doc.page_content = preprocess_text(doc.page_content)
                if len(doc.page_content.strip()) >= 10:  # Basic validation
                    all_documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} documents from {glob_pattern}")
        except Exception as e:
            logger.error(f"Error loading {glob_pattern}: {str(e)}")
    
    return all_documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=[
            "\n\n## ",
            "\n\n### ",
            "\n\n#### ",
            "\n```",
            "\n\n",
            "\n",
            "\n**",
            " ",
            ""
        ],
    )
    
    chunks = []
    for doc in tqdm(documents, desc="Splitting documents"):
        try:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"Error splitting document {doc.metadata.get('source', 'unknown')}: {str(e)}")
    
    return chunks

def create_vectorstore(chunks: List[Document]):
    """Create and save a FAISS vectorstore from document chunks."""
    embeddings = get_embedding_function()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the vectorstore
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info(f"Vectorstore saved to {VECTORSTORE_PATH}")

def load_vectorstore():
    """Load the existing vectorstore."""
    embeddings = get_embedding_function()
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings)
    return vectorstore

def create_qa_chain(vectorstore):
    """Create a question-answering chain using the vectorstore."""
    llm = LMStudioLLM()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
    )
    
    return qa_chain

def query_documents(query: str, qa_chain) -> str:
    """Query the documents using the QA chain."""
    try:
        response = qa_chain.invoke({"query": query})
        return response["result"]
    except Exception as e:
        logger.error(f"Error querying documents: {str(e)}")
        return "Sorry, I encountered an error while processing your query."

def main():
    """Main function to demonstrate usage."""
    try:
        # Load and process documents
        documents = load_documents()
        if not documents:
            logger.error("No valid documents found to process")
            return
            
        chunks = split_documents(documents)
        if not chunks:
            logger.error("No valid chunks created")
            return
            
        # Create and save vectorstore
        create_vectorstore(chunks)
        
        # Load vectorstore and create QA chain
        vectorstore = load_vectorstore()
        qa_chain = create_qa_chain(vectorstore)
        
        # Example query
        query = "What are the main security considerations?"
        answer = query_documents(query, qa_chain)
        print(f"\nQuery: {query}")
        print(f"Answer: {answer}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 