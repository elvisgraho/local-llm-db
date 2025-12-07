import logging
from typing import List, Any, Optional, Dict
from keybert import KeyBERT

# Local Imports
from query.data_service import data_service

logger = logging.getLogger(__name__)

# Constants based on typical Embedding Model limits (e.g. BERT/Nomic)
# We do not guess; we respect the engineering limit of the vector space.
VECTOR_CAPACITY_CHARS = 1000  # Approx 250-300 tokens. Beyond this, dense vectors lose precision.

class KeyBERTAdapter:
    """Adapts LangChain Embeddings to be compatible with KeyBERT's .encode() expectation."""
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def encode(self, docs: List[str]) -> Any:
        if isinstance(docs, str):
            return [self.embedding_model.embed_query(docs)]
        return self.embedding_model.embed_documents(docs)

class QueryProcessor:
    def __init__(self):
        self._kw_model = None

    @property
    def kw_model(self):
        """Lazy load KeyBERT using the existing embedding function."""
        if self._kw_model is None:
            logger.info("Initializing KeyBERT for Semantic Distillation...")
            adapter = KeyBERTAdapter(data_service.embedding_function)
            self._kw_model = KeyBERT(model=adapter)
        return self._kw_model

    def _distill_text(self, text: str, top_n: int = 10) -> str:
        """
        Mathematically reduces text to its most semantically significant components.
        Used to compress history or massive logs into a search-optimized string.
        """
        try:
            keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english', 
                top_n=top_n
            )
            # Join top keywords to form a dense search query
            return " ".join([kw[0] for kw in keywords])
        except Exception as e:
            logger.error(f"Distillation failed: {e}")
            return text[:VECTOR_CAPACITY_CHARS] # Fallback: Truncate

    def process_query(self, query_text: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Constructs the optimal retrieval query by fusing History Anchors with User Intent.
        
        Strategy:
        1. History Anchors: Mathematically extract the 'Subject' of the conversation from history.
        2. Intent Optimization: If current query is massive (noise), distill it. If concise, keep raw.
        3. Fusion: Subject + Intent = Perfect RAG Context.
        """
        # --- Step 1: Extract Semantic Anchors from History ---
        # We always do this. It solves "Why?" and "Tell me more" deterministically.
        history_anchors = ""
        if conversation_history:
            # We look at the last 2 interactions to find the "Active Topic"
            recent_history = conversation_history[-2:]
            history_text = " ".join([m.get("content", "") for m in recent_history])
            
            if history_text.strip():
                # Extract the top 5 entities/concepts from history to anchor the search
                history_anchors = self._distill_text(history_text, top_n=5)
                logger.info(f"History Anchors: [{history_anchors}]")

        # --- Step 2: Optimize Current Query Density ---
        # If the query fits in the vector space, we keep it raw (preserves "Why", "How", precise syntax).
        # If it exceeds capacity (logs/code), we distill it to keywords to prevent vector dilution.
        if len(query_text) > VECTOR_CAPACITY_CHARS:
            logger.info("Input exceeds vector capacity. Distilling to core semantic information.")
            # Extract more keywords for the main query (top 20) to capture details
            refined_query = self._distill_text(query_text, top_n=20)
        else:
            refined_query = query_text

        # --- Step 3: Semantic Fusion ---
        # Combine the Context (History) with the Intent (Query)
        final_search_text = f"{history_anchors} {refined_query}".strip()
        
        logger.info(f"Optimal Search Vector Text: {final_search_text}")
        return final_search_text

# Singleton instance
query_processor = QueryProcessor()