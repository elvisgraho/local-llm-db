import os
import requests
import logging
from langchain_core.embeddings import Embeddings
from query.global_vars import LOCAL_LLM_API_URL

logger = logging.getLogger(__name__)

# prevent accumulation of open sockets
session = requests.Session()

class GenericOpenAIEmbeddings(Embeddings):
    """Generic embedding function for any OpenAI-compatible API (LM Studio, Ollama, vLLM)."""

    def __init__(self, base_url: str = None, model_name: str = None):
        # Default to global vars if not provided
        raw_url = base_url or os.getenv("EMBEDDING_API_URL", LOCAL_LLM_API_URL)
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-nomic-embed-text-v1.5")
        
        # Clean URL construction
        clean_url = raw_url.rstrip('/')
        if clean_url.endswith("/v1"):
            self.api_url = f"{clean_url}/embeddings"
        elif clean_url.endswith("/embeddings"):
             self.api_url = clean_url
        else:
            self.api_url = f"{clean_url}/v1/embeddings"
            
    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        headers = {"Content-Type": "application/json"}
        # 'input' is standard OpenAI format
        payload = {"model": self.model_name, "input": texts}

        try:
            response = session.post(self.api_url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            response_data = response.json()
            
            # Handle different API response shapes (Ollama vs LM Studio)
            if "data" in response_data:
                data = response_data["data"]
                # Ensure sorted by index
                data.sort(key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in data]
            else:
                raise ValueError(f"Unexpected response format: {response_data}")
            
        except Exception as e:
            logger.error(f"Embedding Error ({self.api_url}): {str(e)}")
            raise

    def embed_query(self, text):
        return self.embed_documents([text])[0]

def get_embedding_function(base_url=None, model_name=None):
    """Return an instance of the embedding class with optional overrides."""
    return GenericOpenAIEmbeddings(base_url, model_name)