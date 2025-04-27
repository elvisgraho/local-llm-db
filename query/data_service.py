"""
Data Service for managing RAG system resources.

This service handles loading and managing all data resources used by the RAG system,
including vector stores, graphs, and embedding functions. It provides a singleton
interface to access these resources efficiently.

Classes:
    CustomLLM: A custom LLM implementation that uses the local LLM service.
    DataService: A singleton service class that manages all RAG system resources.

Example:
    >>> from query.data_service import data_service
    >>> # Access resources through the singleton instance
    >>> embedding_function = data_service.embedding_function
    >>> chroma_db = data_service.chroma_db
"""

import json
import networkx as nx
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from query.database_paths import get_db_paths, DEFAULT_DB_NAME, DATABASE_DIR # Import necessary items
from query.llm_service import get_llm_response
from training.get_embedding_function import get_embedding_function, LMStudioEmbeddings
import logging # Add logging

logger = logging.getLogger(__name__) # Setup logger

class CustomLLM(LLM):
    """Custom LLM class that uses our LLM service.
    
    This class implements the LangChain LLM interface to integrate with our local LLM service.
    It provides a simple wrapper around the get_llm_response function.
    
    Attributes:
        _llm_type (str): The type identifier for this LLM implementation.
    """
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the local LLM service with the given prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM.
            stop (Optional[List[str]]): Optional list of stop sequences.
            
        Returns:
            str: The response from the LLM.
        """
        return get_llm_response(prompt)

    @property
    def _llm_type(self) -> str:
        """Get the type identifier for this LLM implementation.
        
        Returns:
            str: The type identifier.
        """
        return "custom"

class DataService:
    """A singleton service class that manages all RAG system resources.
    
    This class provides lazy loading and caching of various resources used by the RAG system,
    including embedding functions, vector stores, and graph structures.
    
    Attributes:
        _instance (Optional[DataService]): The singleton instance.
        _initialized (bool): Flag indicating if initialization is complete.
        _embedding_function (Optional[LMStudioEmbeddings]): Cached embedding function.
        _chroma_cache (Dict[Tuple[str, str], Chroma]): Cache for loaded Chroma databases.
        _vectorstore (Optional[FAISS]): Legacy FAISS vector store.
        _kag_graph_cache (Dict[Tuple[str, str], nx.DiGraph]): Cache for loaded KAG graphs.
        _qa_chain (Optional[RetrievalQA]): Legacy QA chain using FAISS.
        _bm25_cache (Dict[Tuple[str, str], Tuple[Optional[BM25Okapi], Optional[List[str]], Optional[List[str]]]]): Cache for BM25 index, corpus, and doc IDs per DB.
        _reranker (Optional[CrossEncoder]): Cached reranking model.
    """
    
    _instance: Optional['DataService'] = None
    _initialized: bool = False

    def __new__(cls) -> 'DataService':
        """Create or get the singleton instance.
        
        Returns:
            DataService: The singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize the data service if not already initialized."""
        # Ensure initialization runs only once for the singleton
        if getattr(self, '_initialized', False):
            return
        logger.info("Initializing DataService singleton...")
        self._embedding_function: Optional[LMStudioEmbeddings] = None
        self._chroma_cache: Dict[tuple[str, str], Chroma] = {}
        self._vectorstore: Optional[FAISS] = None # Legacy FAISS path
        self._kag_graph_cache: Dict[tuple[str, str], nx.DiGraph] = {}
        self._qa_chain: Optional[RetrievalQA] = None
        self._bm25_cache: Dict[tuple[str, str], tuple[Optional[BM25Okapi], Optional[List[str]], Optional[List[str]]]] = {}
        self._reranker: Optional[CrossEncoder] = None
        self._initialized = True
        logger.info("DataService initialized.")

    @property
    def embedding_function(self) -> LMStudioEmbeddings:
        """Get or initialize the embedding function.
        
        Returns:
            LMStudioEmbeddings: The embedding function.
        """
        if self._embedding_function is None:
            logger.info("Loading embedding function...")
            self._embedding_function = get_embedding_function()
            logger.info("Embedding function loaded.")
        return self._embedding_function

    def get_chroma_db(self, rag_type: str, db_name: str) -> Chroma:
        """Get or initialize a Chroma database for a specific RAG type and name.

        Args:
            rag_type (str): The type of RAG ('rag', 'lightrag').
            db_name (str): The specific name of the database instance.

        Returns:
            Chroma: The Chroma database instance.

        Raises:
            ValueError: If the rag_type is invalid or path cannot be determined.
            Exception: If there's an error initializing the database.
        """
        cache_key = (rag_type, db_name)
        if cache_key in self._chroma_cache:
            logger.debug(f"Returning cached ChromaDB for {cache_key}")
            return self._chroma_cache[cache_key]

        logger.info(f"Loading ChromaDB for rag_type='{rag_type}', db_name='{db_name}'")
        try:
            # Determine the correct path using get_db_paths
            db_paths = get_db_paths(rag_type, db_name)
            # Prioritize 'chroma_path', fallback to 'vectorstore_path' for lightrag compatibility
            chroma_path = db_paths.get("chroma_path")
            if rag_type == 'lightrag' and not chroma_path:
                 chroma_path = db_paths.get("vectorstore_path") # Fallback for lightrag

            if not chroma_path:
                raise ValueError(f"Could not determine Chroma path for rag_type='{rag_type}', db_name='{db_name}'")

            chroma_path_str = str(chroma_path)
            logger.info(f"ChromaDB path determined: {chroma_path_str}")

            # Ensure the database directory exists
            chroma_path.mkdir(parents=True, exist_ok=True)

            # Check if the database likely exists (basic check for the sqlite file)
            db_exists = (chroma_path / "chroma.sqlite3").exists()
            logger.info(f"ChromaDB at {chroma_path_str} {'exists' if db_exists else 'does not exist (will be created)'}.")

            # Load or create the database
            db = Chroma(
                persist_directory=chroma_path_str,
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB for {cache_key} loaded successfully.")
            self._chroma_cache[cache_key] = db
            return db

        except ValueError as ve: # Catch specific path errors
             logger.error(f"Configuration error for ChromaDB {cache_key}: {ve}")
             raise
        except Exception as e:
            logger.error(f"Error initializing Chroma database for {cache_key} at {chroma_path_str}: {e}", exc_info=True)
            raise # Re-raise other exceptions

    def get_bm25_index(self, rag_type: str, db_name: str) -> Optional[BM25Okapi]:
        """Get or initialize the BM25 index for a specific ChromaDB instance.

        Args:
            rag_type (str): The RAG type of the ChromaDB.
            db_name (str): The name of the ChromaDB instance.

        Returns:
            Optional[BM25Okapi]: The BM25 index, or None if loading fails.
        """
        cache_key = (rag_type, db_name)
        if cache_key in self._bm25_cache:
            logger.debug(f"Returning cached BM25 index for {cache_key}")
            return self._bm25_cache[cache_key][0] # Return only the index

        logger.info(f"Loading BM25 index for {cache_key}...")
        self._load_bm25_data(rag_type, db_name)
        return self._bm25_cache.get(cache_key, (None, None, None))[0]

    def _load_bm25_data(self, rag_type: str, db_name: str) -> None:
        """Load data from a specific ChromaDB instance and build its BM25 index."""
        cache_key = (rag_type, db_name)
        bm25_index = None
        corpus = None
        doc_ids = None
        try:
            # Ensure the corresponding ChromaDB is loaded
            db = self.get_chroma_db(rag_type, db_name)
            if db is None:
                logger.warning(f"ChromaDB {cache_key} not available for BM25 indexing.")
                return

            logger.info(f"Retrieving documents from ChromaDB {cache_key} for BM25...")
            all_docs = db.get(include=["documents", "metadatas"]) # IDs are included by default

            if not all_docs or not all_docs.get("ids"):
                logger.warning(f"No documents found in ChromaDB {cache_key} for BM25 indexing.")
                return

            corpus = all_docs["documents"]
            doc_ids = all_docs["ids"] # Use ChromaDB internal IDs

            if not corpus:
                logger.warning(f"Corpus is empty for {cache_key}, cannot build BM25 index.")
                return

            logger.info(f"Tokenizing {len(corpus)} documents for BM25 index {cache_key}...")
            # Simple tokenization (split by space) - consider a more robust tokenizer if needed
            tokenized_corpus = [doc.split(" ") for doc in corpus]

            logger.info(f"Building BM25 index for {cache_key}...")
            bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 index for {cache_key} built successfully.")

        except Exception as e:
            logger.error(f"Error loading data or building BM25 index for {cache_key}: {e}", exc_info=True)
            # Ensure attributes are reset if loading fails
            bm25_index = None
            corpus = None
            doc_ids = None
        finally:
             # Store results (even if None) in cache to avoid re-attempting on failure
             self._bm25_cache[cache_key] = (bm25_index, corpus, doc_ids)


    # --- Legacy FAISS properties - Keep for potential backward compatibility ---
    # --- but clearly mark as using default paths and not the new structure ---
    @property
    def vectorstore(self) -> Optional[FAISS]:
        """Get or initialize the LEGACY FAISS vectorstore (uses default path)."""
        # Determine the default path (assuming it was under 'lightrag/default')
        # This path might need adjustment based on the old structure.
        # For now, let's assume a plausible default path.
        try:
            legacy_paths = get_db_paths('lightrag', DEFAULT_DB_NAME) # Get paths for default lightrag
            legacy_faiss_path = legacy_paths.get("vectorstore_path")
            if legacy_faiss_path and legacy_faiss_path.exists():
                 if self._vectorstore is None:
                     logger.warning(f"Loading LEGACY FAISS vectorstore from default path: {legacy_faiss_path}")
                     self._vectorstore = FAISS.load_local(
                         str(legacy_faiss_path),
                         self.embedding_function,
                         allow_dangerous_deserialization=True
                     )
                 return self._vectorstore
            else:
                 logger.warning(f"Legacy FAISS path not found or does not exist: {legacy_faiss_path}")
                 return None
        except Exception as e:
            logger.error(f"Error loading legacy FAISS vectorstore: {e}", exc_info=True)
            return None


    def get_kag_graph(self, rag_type: str, db_name: str) -> nx.DiGraph:
        """Get or initialize the KAG graph for a specific RAG type and name.

        Args:
            rag_type (str): The type of RAG ('kag').
            db_name (str): The specific name of the database instance.

        Returns:
            nx.DiGraph: The KAG graph.

        Raises:
            ValueError: If rag_type is not 'kag' or path cannot be determined.
            FileNotFoundError: If the graph file does not exist.
            Exception: If there's an error loading the graph.
        """
        if rag_type != 'kag':
            raise ValueError(f"KAG graph is only applicable for rag_type='kag', not '{rag_type}'")

        cache_key = (rag_type, db_name)
        if cache_key in self._kag_graph_cache:
            logger.debug(f"Returning cached KAG graph for {cache_key}")
            return self._kag_graph_cache[cache_key]

        logger.info(f"Loading KAG graph for {cache_key}...")
        try:
            db_paths = get_db_paths(rag_type, db_name)
            graph_path = db_paths.get("graph_path")

            if not graph_path:
                raise ValueError(f"Could not determine graph path for {cache_key}")

            graph_path_str = str(graph_path)
            logger.info(f"KAG graph path determined: {graph_path_str}")

            if not graph_path.exists():
                 raise FileNotFoundError(f"KAG graph file not found at: {graph_path_str}")

            with open(graph_path_str, 'r') as f:
                graph_data = json.load(f)

            graph = nx.DiGraph()
            # Safely access nodes and edges, providing defaults if keys are missing
            nodes_data = graph_data.get('nodes', [])
            edges_data = graph_data.get('edges', [])

            for node in nodes_data:
                 node_id = node.get('id')
                 node_attrs = node.get('data', {})
                 if node_id is not None:
                      graph.add_node(node_id, **node_attrs)
                 else:
                      logger.warning(f"Skipping node with missing 'id' in graph {cache_key}")

            for edge in edges_data:
                 source = edge.get('source')
                 target = edge.get('target')
                 edge_attrs = edge.get('data', {})
                 if source is not None and target is not None:
                      graph.add_edge(source, target, **edge_attrs)
                 else:
                      logger.warning(f"Skipping edge with missing 'source' or 'target' in graph {cache_key}: {edge}")


            logger.info(f"KAG graph for {cache_key} loaded successfully with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
            self._kag_graph_cache[cache_key] = graph
            return graph

        except FileNotFoundError as fnf:
             logger.error(f"KAG graph file not found for {cache_key}: {fnf}")
             raise # Re-raise specific error
        except ValueError as ve:
             logger.error(f"Configuration error for KAG graph {cache_key}: {ve}")
             raise
        except Exception as e:
            logger.error(f"Error loading KAG graph for {cache_key} from {graph_path_str}: {e}", exc_info=True)
            raise # Re-raise other exceptions


    @property
    def qa_chain(self) -> Optional[RetrievalQA]:
        """Get or initialize the LEGACY QA chain (uses legacy FAISS vectorstore)."""
        legacy_vs = self.vectorstore # Try to get the legacy FAISS store
        if legacy_vs is None:
             logger.warning("Cannot initialize legacy QA chain because legacy FAISS vectorstore failed to load.")
             return None

        if self._qa_chain is None:
            logger.warning("Initializing LEGACY QA chain using default FAISS vectorstore.")
            llm = CustomLLM()
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=legacy_vs.as_retriever(
                    search_kwargs={"k": 3} # Default k=3
                )
            )
        return self._qa_chain

    @property
    def reranker(self) -> Optional[CrossEncoder]:
        """Get or initialize the reranking model.
        
        Returns:
            Optional[CrossEncoder]: The CrossEncoder model, or None if loading fails.
        """
        if self._reranker is None:
            logger.info("Loading reranker model...")
            try:
                # Load a pre-trained CrossEncoder model
                # Consider making the model name configurable if needed
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                logger.info("Reranker model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading reranker model: {e}", exc_info=True)
                self._reranker = None # Ensure it's None on failure
        return self._reranker

    def clear_cache(self) -> None:
        """Clear all cached resources managed by the DataService."""
        logger.warning("Clearing DataService cache...")
        self._embedding_function = None
        self._chroma_cache.clear()
        self._vectorstore = None # Clear legacy FAISS
        self._kag_graph_cache.clear()
        self._qa_chain = None # Clear legacy QA chain
        self._bm25_cache.clear()
        self._reranker = None
        logger.info("DataService cache cleared.")

# --- Initialization Function (Adjusted) ---
def initialize_data_service() -> None:
    """Initialize the data service by pre-loading resources that don't require specific DB names."""
    logger.info("Initializing data service (pre-loading common resources)...")
    try:
        # Pre-load resources that are independent of specific databases
        logger.info("Initializing embedding function...")
        _ = data_service.embedding_function
        logger.info("Initializing reranker...")
        _ = data_service.reranker

        # Optionally, you could try to load default databases if they exist,
        # but this might hide errors if defaults are missing.
        # Example: Try loading default 'rag' DB
        # try:
        #     logger.info("Attempting to pre-load default 'rag' Chroma DB...")
        #     _ = data_service.get_chroma_db('rag', DEFAULT_DB_NAME)
        # except Exception:
        #     logger.warning("Could not pre-load default 'rag' Chroma DB (might not exist yet).")

        logger.info("Common data service resources initialized!")
    except Exception as e:
        logger.error(f"Error during data service initialization: {e}", exc_info=True)
        # Depending on the severity, you might want to exit or just log the error

# Create the singleton instance upon import
data_service = DataService()