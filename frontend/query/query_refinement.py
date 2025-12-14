import logging
from typing import List, Any, Optional, Dict, Tuple
from keybert import KeyBERT
from query.data_service import data_service

logger = logging.getLogger(__name__)

# Constants
VECTOR_CAPACITY_CHARS = 1000 
DEFAULT_TOP_N_QUERY = 15

class KeyBERTAdapter:
    """
    Adapts LangChain Embeddings to be compatible with KeyBERT.
    """
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model

    def encode(self, docs: List[str]) -> Any:
        if isinstance(docs, str):
            docs = [docs]
        try:
            return self.embedding_model.embed_documents(docs)
        except Exception as e:
            logger.error(f"Embedding encoding failed: {e}")
            return []

class QueryProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QueryProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._kw_model: Optional[KeyBERT] = None
        self._initialized = True

    def _get_model(self) -> Optional[KeyBERT]:
        if self._kw_model:
            return self._kw_model
        
        if not data_service or not data_service.embedding_function:
             return None
             
        try:
            adapter = KeyBERTAdapter(data_service.embedding_function)
            self._kw_model = KeyBERT(model=adapter)
            return self._kw_model
        except Exception:
            return None

    def _extract_keywords_raw(self, text: str, top_n: int = 10, use_mmr: bool = True) -> List[Tuple[str, float]]:
        """
        Returns the raw list of (keyword, score) tuples. 
        Used for internal logic to determine query density.
        """
        model = self._get_model()
        if not model or not text.strip():
            return []

        try:
            # We use a lower diversity here to ensure we catch all relevant topics
            # to judge density, before filtering.
            keywords = model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english', 
                use_mmr=use_mmr,
                diversity=0.5, 
                top_n=top_n
            )
            return keywords
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
            return []

    def process_query(self, query_text: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Constructs the optimal retrieval query by fusing History Anchors with User Intent.
        
        Strategy:
        1. Extract Keywords from Current Query.
        2. Check Semantic Density: 
           - If Query has < 2 strong keywords, it is "Context Dependent" (e.g. "Why?", "It failed").
           - If Query has many keywords, it is "Self Contained".
        3. Dynamically inject history based on Density.
        """
        # --- Step 1: Analyze Current Query Density ---
        # We try to extract up to 10 keywords.
        query_kw_tuples = self._extract_keywords_raw(query_text, top_n=10, use_mmr=True)
        
        # A "Strong" keyword is one that isn't just a stopword (KeyBERT handles stopwords, 
        # but we also check if the list is empty).
        query_keywords = [kw[0] for kw in query_kw_tuples]
        
        # Determine Dependency
        # If we found 0 or 1 keyword, the user likely typed "Why?" or "Tell me more".
        # Even "Java" (1 keyword) benefits from history context to know *what* about Java.
        # "Python Loop Error" (3 keywords) is specific.
        is_context_dependent = len(query_keywords) < 2
        
        # --- Step 2: Handle Input Capacity (Noise Reduction) ---
        if len(query_text) > VECTOR_CAPACITY_CHARS:
            # If text is huge (logs), we MUST use the extracted keywords
            refined_query = " ".join(query_keywords)
        else:
            # If text fits, keep raw text to preserve syntax/grammar
            refined_query = query_text

        # --- Step 3: Dynamic History Injection ---
        history_anchors = ""
        
        if conversation_history:
            # logic: If the query is context dependent, we need DEEP history.
            # If the query is specific, we need LIGHT history (or none) to avoid vector pollution.
            
            target_history_count = 5 if is_context_dependent else 1
            
            recent_history = conversation_history[-2:]
            history_text = " ".join([m.get("content", "") for m in recent_history])
            
            if history_text.strip():
                # Extract history keywords
                hist_tuples = self._extract_keywords_raw(
                    history_text, 
                    top_n=target_history_count, 
                    use_mmr=True
                )
                history_anchors = " ".join([kw[0] for kw in hist_tuples])

        # --- Step 4: Semantic Fusion ---
        # Construct the final vector search string
        final_parts = []
        
        # If we have history anchors, add them as context
        if history_anchors:
            final_parts.append(f"Context: {history_anchors}")
            
        # Add the actual question/query
        final_parts.append(f"Query: {refined_query}")
        
        final_search_text = " ".join(final_parts)
        
        logger.info(f"Query Processing: Dependent={is_context_dependent} | Keywords Found={len(query_keywords)} | Final Vector: {final_search_text}")
        
        return final_search_text

# Singleton instance
query_processor = QueryProcessor()