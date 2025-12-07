"""
Data Service for managing RAG system resources.

This service handles loading and managing all data resources used by the RAG system,
including vector stores, graphs, and embedding functions.

Updates:
- Pure LCEL implementation (No langchain.chains imports).
- Removed LegacyChainWrapper (Standard .invoke() API).
- Strict typing and path management.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pickle
import networkx as nx
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# --- LangChain Core Imports (Pure LCEL) ---
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- Integration Imports ---
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# --- Local Imports ---
from query.database_paths import get_db_paths, DEFAULT_DB_NAME
from query.llm_service import get_llm_response
from query.embeddings import get_embedding_function, LMStudioEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomLLM(LLM):
    """
    Custom LLM wrapper for the local LLM service.
    Inherits from langchain_core LLM.
    """
    llm_config: Optional[Dict[str, Any]] = None
    
    @property
    def _llm_type(self) -> str:
        return "custom_local_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return get_llm_response(prompt, llm_config=self.llm_config)

class DataService:
    """
    Singleton service for RAG resources (Embeddings, Vector Stores, Graphs).
    """
    
    _instance: Optional['DataService'] = None
    _initialized: bool = False

    def __new__(cls) -> 'DataService':
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, '_initialized', False):
            return
            
        logger.info("Initializing DataService singleton...")
        
        # Core Resources
        self._embedding_function: Optional[LMStudioEmbeddings] = None
        self._reranker: Optional[CrossEncoder] = None
        
        # Caches
        self._chroma_cache: Dict[Tuple[str, str], Chroma] = {}
        self._kag_graph_cache: Dict[Tuple[str, str], nx.DiGraph] = {}
        # BM25 Cache: (Index, Corpus, DocIDs)
        self._bm25_cache: Dict[Tuple[str, str], Tuple[Optional[BM25Okapi], Optional[List[str]], Optional[List[str]]]] = {}
        
        # Vector Store & Chain
        self._vectorstore: Optional[FAISS] = None
        self._qa_chain: Optional[Runnable] = None
        
        self._initialized = True
        logger.info("DataService initialized.")

    @property
    def embedding_function(self) -> LMStudioEmbeddings:
        """Get or initialize the embedding function."""
        if self._embedding_function is None:
            logger.info("Loading embedding function...")
            try:
                self._embedding_function = get_embedding_function()
            except Exception as e:
                logger.error(f"Failed to load embedding function: {e}")
                raise
        return self._embedding_function

    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """Get or initialize the reranking model (lazy load)."""
        if self._reranker is None:
            logger.info("Loading reranker model (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
            try:
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            except Exception as e:
                logger.error(f"Failed to load reranker: {e}")
                self._reranker = None
        return self._reranker

    def get_chroma_db(self, rag_type: str, db_name: str) -> Optional[Chroma]:
        """Get or initialize a Chroma database."""
        cache_key = (rag_type, db_name)
        if cache_key in self._chroma_cache:
            return self._chroma_cache[cache_key]

        logger.info(f"Loading ChromaDB: type='{rag_type}', name='{db_name}'")
        try:
            db_paths = get_db_paths(rag_type, db_name)
            
            # Prioritize 'chroma_path', fallback to 'vectorstore_path' for older lightrag structs
            chroma_path = db_paths.get("chroma_path")
            if not chroma_path and rag_type == 'lightrag':
                 chroma_path = db_paths.get("vectorstore_path")

            if not chroma_path:
                raise ValueError(f"No valid path determined for {rag_type}/{db_name}")

            chroma_path_obj = Path(chroma_path)
            chroma_path_obj.mkdir(parents=True, exist_ok=True)
            
            db = Chroma(
                persist_directory=str(chroma_path_obj),
                embedding_function=self.embedding_function
            )
            
            self._chroma_cache[cache_key] = db
            return db

        except Exception as e:
            logger.error(f"Error loading ChromaDB {cache_key}: {e}", exc_info=True)
            return None

    def get_bm25_index(self, rag_type: str, db_name: str) -> Optional[BM25Okapi]:
        """Get or build BM25 index for hybrid search."""
        cache_key = (rag_type, db_name)
        if cache_key in self._bm25_cache:
            return self._bm25_cache[cache_key][0]

        logger.info(f"Building BM25 index for {cache_key}...")
        self._build_bm25_index(rag_type, db_name)
        return self._bm25_cache.get(cache_key, (None, None, None))[0]

    def _build_bm25_index(self, rag_type: str, db_name: str) -> None:
        """
        Builds or loads cached BM25 index.
        Optimized for large KBs by using pickle cache.
        """
        cache_key = (rag_type, db_name)
        db_paths = get_db_paths(rag_type, db_name)
        bm25_cache_path = db_paths["db_dir"] / "bm25_index.pkl"

        try:
            # 1. Try Load from Disk
            if bm25_cache_path.exists():
                logger.info(f"Loading cached BM25 index from {bm25_cache_path}")
                with open(bm25_cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Simple validation could be added here (check file modification times)
                    self._bm25_cache[cache_key] = cached_data
                    return

            # 2. Build from Scratch (Only if no cache)
            logger.info(f"Building BM25 index from scratch for {cache_key}...")
            db = self.get_chroma_db(rag_type, db_name)
            if not db:
                raise ValueError("DB not available")

            # Warning: For massive DBs, this fetch might still be slow. 
            # Ideally, limit this or use a generator if Chroma supports it.
            data = db.get(include=["documents", "ids"])
            corpus = data.get("documents", [])
            doc_ids = data.get("ids", [])

            if not corpus:
                logger.warning(f"Empty corpus in {cache_key} - cannot build BM25.")
                self._bm25_cache[cache_key] = (None, None, None)
                return

            tokenized_corpus = [doc.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            
            cache_tuple = (bm25, corpus, doc_ids)
            self._bm25_cache[cache_key] = cache_tuple
            
            # 3. Save to Disk
            logger.info(f"Saving BM25 index to {bm25_cache_path}")
            with open(bm25_cache_path, 'wb') as f:
                pickle.dump(cache_tuple, f)

        except Exception as e:
            logger.error(f"BM25 build failed for {cache_key}: {e}")
            self._bm25_cache[cache_key] = (None, None, None)

    def get_kag_graph(self, rag_type: str, db_name: str) -> Optional[nx.DiGraph]:
        """Load NetworkX graph for KAG."""
        if rag_type != 'kag':
            logger.warning("Graph requested for non-KAG type.")
            return None

        cache_key = (rag_type, db_name)
        if cache_key in self._kag_graph_cache:
            return self._kag_graph_cache[cache_key]

        try:
            db_paths = get_db_paths(rag_type, db_name)
            graph_path = db_paths.get("graph_path")

            if not graph_path or not graph_path.exists():
                logger.warning(f"Graph file missing: {graph_path}")
                return None

            logger.info(f"Loading Graph from {graph_path}")
            with open(graph_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            graph = nx.DiGraph()
            for n in data.get('nodes', []):
                if 'id' in n:
                    graph.add_node(n['id'], **n.get('data', {}))
            for e in data.get('edges', []):
                if 'source' in e and 'target' in e:
                    graph.add_edge(e['source'], e['target'], **e.get('data', {}))

            self._kag_graph_cache[cache_key] = graph
            return graph
        except Exception as e:
            logger.error(f"Graph load failed {cache_key}: {e}")
            return None

    # --- Legacy Data Loading (Wrapped in Modern Interface) ---

    @property
    def vectorstore(self) -> Optional[FAISS]:
        """
        Attempts to load the legacy FAISS store if it exists on disk.
        """
        if self._vectorstore:
            return self._vectorstore
        
        try:
            paths = get_db_paths('lightrag', DEFAULT_DB_NAME)
            path = paths.get("vectorstore_path")
            
            if path and path.exists():
                logger.info(f"Loading legacy FAISS from {path}")
                self._vectorstore = FAISS.load_local(
                    str(path),
                    self.embedding_function,
                    allow_dangerous_deserialization=True
                )
                return self._vectorstore
        except Exception as e:
            logger.warning(f"Legacy FAISS load failed: {e}")
        
        return None

    @property
    def qa_chain(self) -> Optional[Runnable]:
        """
        Constructs a pure LCEL RAG chain using the loaded vectorstore.
        
        Usage:
            response = data_service.qa_chain.invoke({"input": "Your question here"})
            print(response) # String output
        """
        if self._qa_chain:
            return self._qa_chain

        vs = self.vectorstore
        if vs:
            logger.info("Constructing Pure LCEL RAG Chain")
            
            llm = CustomLLM(llm_config={})
            retriever = vs.as_retriever(search_kwargs={"k": 3})
            
            prompt = ChatPromptTemplate.from_template(
                """Answer the question based only on the following context:

<context>
{context}
</context>

Question: {input}"""
            )

            def format_docs(docs: List[Document]) -> str:
                return "\n\n".join(doc.page_content for doc in docs)

            # Pure LCEL Construction
            # 1. RunnableParallel: Retrieves docs and passes the input question through
            # 2. Prompt: Formats the context and input
            # 3. LLM: Generates response
            # 4. StrOutputParser: Extracts string from LLM result
            
            self._qa_chain = (
                RunnableParallel({
                    "context": (lambda x: x["input"]) | retriever | format_docs,
                    "input": lambda x: x["input"]
                })
                | prompt
                | llm
                | StrOutputParser()
            )

        return self._qa_chain

    def clear_cache(self) -> None:
        """Clear all cached data."""
        logger.warning("Clearing DataService cache...")
        self._chroma_cache.clear()
        self._kag_graph_cache.clear()
        self._bm25_cache.clear()
        self._vectorstore = None
        self._qa_chain = None
        logger.info("Cache cleared.")

def initialize_data_service() -> None:
    """Pre-load shared models."""
    try:
        logger.info("Pre-loading common resources...")
        _ = data_service.embedding_function
        _ = data_service.reranker
    except Exception as e:
        logger.error(f"Init warning: {e}")

data_service = DataService()