import logging
from typing import List, Any, Optional, Dict, Tuple
from keybert import KeyBERT
from query.data_service import data_service

logger = logging.getLogger(__name__)

# Constants
# If a query is longer than this, it's likely a log paste or code dump.
# We summarize it to prevent "context stuffing" the retrieval.
LOG_DUMP_THRESHOLD_CHARS = 1000 

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

    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extracts keywords using general semantic density.
        
        CRITICAL CHANGE: stop_words=None.
        In cybersecurity and coding, 'import', 'from', 'who', 'am', 'i' are 
        critical tokens. We must not strip them.
        """
        model = self._get_model()
        if not model or not text.strip():
            return []

        try:
            keywords = model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 2), # Captures "buffer overflow", "syntax error"
                stop_words=None,              # GENERALIZATION: Preserves code/shell syntax
                use_mmr=True,                 # Maximize diversity of keywords
                diversity=0.3,                # Low diversity to stay focused on topic
                top_n=top_n
            )
            return [kw[0] for kw in keywords]
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
            # Fallback: if extraction fails, return empty so we default to raw text
            return []

    def process_query(self, query_text: str, conversation_history: Optional[List[Dict]] = None) -> str:
        """
        Prepares the optimal query for Hybrid Search (Vector + Keyword).
        
        Strategy:
        1. Identify if input is a "Log Dump" or "Natural Question".
        2. If Log Dump -> Extract semantic keywords (reduce noise).
        3. If Natural Question -> Keep raw text (preserve specificity).
        4. Inject 'Anchors' from conversation history to resolve ambiguity (e.g., "it", "that function").
        """
        
        # --- Step 1: Input Analysis ---
        is_log_dump = len(query_text) > LOG_DUMP_THRESHOLD_CHARS
        
        cleaned_query = query_text
        
        # --- Step 2: Query Refinement ---
        if is_log_dump:
            # If user pastes a massive error log, the vector search will dilute.
            # We extract the top 10 most semantic terms from the log.
            # This turns a 2000-char stack trace into "NullPointerException authentication failure timeout"
            extracted = self._extract_keywords(query_text, top_n=15)
            if extracted:
                cleaned_query = " ".join(extracted)
        
        # --- Step 3: History Context Injection (Augmentation) ---
        history_augmentation = ""
        
        if conversation_history:
            # logic: We want to grab the "Topic" of the last interaction, not the whole text.
            # This helps disambiguate queries like "how do I fix it?"
            
            # Get last AI response (usually contains the most relevant technical terms)
            last_message = conversation_history[-1].get("content", "")
            
            if last_message:
                # We extract only 2-3 keywords. 
                # Why? We want to nudge the vector, not drag it completely to the past.
                hist_keywords = self._extract_keywords(last_message, top_n=3)
                history_augmentation = " ".join(hist_keywords)

        # --- Step 4: Final Construction ---
        # We combine the user's immediate intent with the historical context.
        # We do NOT use labels like "Context:" because they add noise to BM25/Vector scores.
        
        if history_augmentation:
            # "python loop error" + "pandas dataframe"
            final_search_text = f"{cleaned_query} {history_augmentation}"
        else:
            final_search_text = cleaned_query
            
        # Logging for observability
        logger.info(f"Query Processed: LogDump={is_log_dump} | Context Added='{history_augmentation}'")
        
        return final_search_text

# Singleton instance
query_processor = QueryProcessor()