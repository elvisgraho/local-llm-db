from langchain.embeddings.base import Embeddings
import os
import requests
import logging

logger = logging.getLogger(__name__)

# Fetch base URL from environment (set by UI), default to localhost
_base_url = os.getenv("LOCAL_LLM_API_URL", "http://localhost:1234").rstrip("/")
LM_STUDIO_API_URL = f"{_base_url}/v1/embeddings"

# Fetch model name from environment
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-embedder_collection")

# prevent accumulation of open sockets
session = requests.Session()

class LMStudioEmbeddings(Embeddings):
    """Custom embedding function to use LM Studio's local API."""

    def embed_documents(self, texts):
        """Generate embeddings for a list of documents."""
        headers = {"Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL_NAME, "input": texts}

        try:
            response = session.post(LM_STUDIO_API_URL, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            
            response_data = response.json()
            logger.debug(f"Received response from LM Studio API: {response_data}")

            if "data" not in response_data or not isinstance(response_data["data"], list):
                raise ValueError("Invalid response format from LM Studio API: " + str(response_data))

            embeddings = [item["embedding"] for item in response_data["data"] if "embedding" in item]
            logger.debug(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to LM Studio API at {LM_STUDIO_API_URL}. Is LM Studio running?")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to LM Studio API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in embed_documents: {str(e)}")
            raise

    def embed_query(self, text):
        """Generate an embedding for a single query string."""
        return self.embed_documents([text])[0]  # Return the first embedding


def get_embedding_function():
    """Return an instance of the embedding class."""
    return LMStudioEmbeddings()