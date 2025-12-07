# --- START OF FILE query/query_refinement.py ---
import logging
from typing import List, Dict, Any
from keybert import KeyBERT

# Local Imports
from query.llm_service import get_llm_response
from query.data_service import data_service

logger = logging.getLogger(__name__)

# --- TEMPLATES ---

# This prompt is strictly for resolving pronouns and context
CONTEXTUALIZE_TEMPLATE = """
Given a chat history and the latest user question which might reference context in the history, 
formulate a standalone question which can be understood without the chat history.
DO NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{history}

Latest Question: 
{question}

Standalone Question:
"""

class KeyBERTAdapter:
    """Adapts LangChain Embeddings to be compatible with KeyBERT's .encode() expectation."""
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def encode(self, docs: List[str]) -> Any:
        # KeyBERT might pass a single string or a list
        if isinstance(docs, str):
            # embed_query returns a list of floats, KeyBERT expects list of list of floats or numpy array
            return [self.embedding_model.embed_query(docs)]
        return self.embedding_model.embed_documents(docs)

class QueryProcessor:
    def __init__(self):
        self._kw_model = None

    @property
    def kw_model(self):
        """Lazy load KeyBERT using the existing embedding function to save RAM."""
        if self._kw_model is None:
            logger.info("Initializing KeyBERT with existing embedding model...")
            adapter = KeyBERTAdapter(data_service.embedding_function)
            self._kw_model = KeyBERT(model=adapter)
        return self._kw_model

    def extract_keywords(self, text: str, top_n: int = 5) -> str:
        """
        Extracts key phrases from a large text block locally.
        Great for when users paste code or logs.
        """
        try:
            # If text is short, just return it
            if len(text.split()) < 20:
                return text

            # Extract keywords (returns list of tuples [(word, score)])
            keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english', 
                top_n=top_n
            )
            
            # Join them back into a search string
            refined = " ".join([kw[0] for kw in keywords])
            logger.info(f"KeyBERT extracted: {refined}")
            return refined
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return text

    def contextualize_query(self, query: str, history: List[Dict], llm_config: Dict) -> str:
        """
        Uses the Main LLM to resolve pronouns (it, that, the code).
        """
        if not history:
            return query

        # Limit history to last 3 turns to keep prompt small/fast
        recent_history = history[-3:]
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])

        prompt = CONTEXTUALIZE_TEMPLATE.format(history=history_str, question=query)
        
        try:
            # Use a low temperature for deterministic rephrasing
            refined = get_llm_response(prompt, llm_config=llm_config, temperature=0.1)
            # Cleanup output
            cleaned = refined.replace("Standalone Question:", "").strip()
            logger.info(f"Contextualized: '{query}' -> '{cleaned}'")
            return cleaned
        except Exception as e:
            logger.error(f"Contextualization failed: {e}")
            return query

    def process_query(self, query_text: str, history: List[Dict], llm_config: Dict) -> str:
        """
        The Master Strategy:
        1. If history exists -> Contextualize (turn "fix it" into "fix the auth error").
        2. If result is huge (e.g., user pasted a log) -> Extract Keywords.
        3. Return the optimized search string.
        """
        # Step 1: Contextualize (Fix pronouns)
        search_query = self.contextualize_query(query_text, history, llm_config)

        # Step 2: If the contextualized query is massive (tokens), reduce it to keywords
        # 100 words is roughly where vector search starts getting noisy
        if len(search_query.split()) > 100:
            logger.info("Query is too long, applying KeyBERT extraction...")
            search_query = self.extract_keywords(search_query)

        return search_query

# Singleton instance
query_processor = QueryProcessor()