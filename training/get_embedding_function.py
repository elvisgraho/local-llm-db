from chromadb import Embeddings
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.embeddings import OllamaEmbeddings 
import requests


LM_STUDIO_API_URL = "http://localhost:1234/v1/embeddings"  # Change if needed
EMBEDDING_MODEL_NAME = "text-embedding-mxbai-embed-large-v1"  # Update this!


class LMStudioEmbeddings(Embeddings):
    """Custom embedding function to use LM Studio's local API."""

    def embed_documents(self, texts):
        headers = {"Content-Type": "application/json"}
        payload = {"model": EMBEDDING_MODEL_NAME, "input": texts}

        response = requests.post(LM_STUDIO_API_URL, json=payload, headers=headers)
        response_data = response.json()

        if "data" not in response_data or not isinstance(response_data["data"], list):
            raise ValueError("Invalid response format from LM Studio API: " + str(response_data))

        return [item["embedding"] for item in response_data["data"] if "embedding" in item]

    def embed_query(self, text):
        """Generate an embedding for a single query string."""
        return self.embed_documents([text])[0]  # Return the first embedding


def get_embedding_function():
    """Return an instance of the embedding class."""
    return LMStudioEmbeddings()

# def get_embedding_function():
#     # embeddings = BedrockEmbeddings(
#     #     credentials_profile_name="default", region_name="us-east-1"
#     # )
#     embeddings = OllamaEmbeddings(model="mxbai-embed-large") 
#     return embeddings
