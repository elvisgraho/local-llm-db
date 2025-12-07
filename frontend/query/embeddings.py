# --- START OF FILE query/embeddings.py ---
import os
import requests
import logging
from langchain_core.embeddings import Embeddings

# Import global config to keep settings centralized
from query.global_vars import LOCAL_LLM_API_URL

logger = logging.getLogger(__name__)

# Default to the global API URL, but target the embeddings endpoint
raw_url = os.getenv("EMBEDDING_API_URL", LOCAL_LLM_API_URL)
# Strip specific endpoints to get the base
if "/chat/completions" in raw_url:
    base = raw_url.split("/chat/completions")[0]
elif "/embeddings" in raw_url:
    base = raw_url.split("/embeddings")[0]
else:
    base = raw_url.rstrip('/')
# Reconstruct correctly
if base.endswith('/v1'):
    BASE_URL = f"{base}/embeddings"
else:
    BASE_URL = f"{base}/v1/embeddings"
    
# Get model name from env or default
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-embedder_collection")

class LMStudioEmbeddings(Embeddings):
    """Custom embedding function to use LM Studio's local API."""

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        headers = {"Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL_NAME, "input": texts}

        try:
            response = requests.post(BASE_URL, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            response_data = response.json()
            
            if "data" not in response_data or not isinstance(response_data["data"], list):
                raise ValueError(f"Invalid response format from API: {response_data}")

            # Sort by index to ensure order is preserved (common issue with async APIs)
            data = response_data["data"]
            data.sort(key=lambda x: x.get("index", 0))
            
            embeddings = [item["embedding"] for item in data if "embedding" in item]
            return embeddings
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Embedding API at {BASE_URL}. Is the server running?")
            raise
        except Exception as e:
            logger.error(f"Error getting embeddings: {str(e)}")
            raise

    def embed_query(self, text):
        """Generate an embedding for a single query string."""
        return self.embed_documents([text])[0]

def get_embedding_function():
    """Return an instance of the embedding class."""
    return LMStudioEmbeddings()