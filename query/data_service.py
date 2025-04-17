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
from query.database_paths import CHROMA_PATH, GRAPHRAG_GRAPH_PATH, KAG_GRAPH_PATH, VECTORSTORE_PATH
from query.llm_service import get_llm_response
from training.get_embedding_function import get_embedding_function, LMStudioEmbeddings

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
        _initialized (bool): Whether the instance has been initialized.
        _embedding_function (Optional[LMStudioEmbeddings]): The embedding function.
        _chroma_db (Optional[Chroma]): The Chroma vector store.
        _vectorstore (Optional[FAISS]): The FAISS vector store.
        _graphrag_graph (Optional[nx.DiGraph]): The GraphRAG graph.
        _kag_graph (Optional[nx.DiGraph]): The KAG graph.
        _qa_chain (Optional[RetrievalQA]): The QA chain.
        _bm25_index (Optional[BM25Okapi]): The BM25 index.
        _bm25_corpus (Optional[List[str]]): The corpus for BM25.
        _bm25_doc_ids (Optional[List[str]]): Document IDs corresponding to the BM25 corpus.
        _reranker (Optional[CrossEncoder]): The reranking model.
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
        if not self._initialized:
            self._embedding_function: Optional[LMStudioEmbeddings] = None
            self._chroma_db: Optional[Chroma] = None
            self._vectorstore: Optional[FAISS] = None
            self._graphrag_graph: Optional[nx.DiGraph] = None
            self._kag_graph: Optional[nx.DiGraph] = None
            self._qa_chain: Optional[RetrievalQA] = None
            self._initialized = True
            self._bm25_index: Optional[BM25Okapi] = None
            self._bm25_corpus: Optional[List[str]] = None
            self._bm25_doc_ids: Optional[List[str]] = None
            self._reranker: Optional[CrossEncoder] = None

    @property
    def embedding_function(self) -> LMStudioEmbeddings:
        """Get or initialize the embedding function.
        
        Returns:
            LMStudioEmbeddings: The embedding function.
        """
        if self._embedding_function is None:
            self._embedding_function = get_embedding_function()
        return self._embedding_function

    @property
    def chroma_db(self) -> Chroma:
        """Get or initialize the Chroma database.
        
        Returns:
            Chroma: The Chroma database.
            
        Raises:
            Exception: If there's an error initializing the database.
        """
        if self._chroma_db is None:
            # Ensure the database directory exists
            os.makedirs(str(CHROMA_PATH), exist_ok=True)
            
            # Check if database exists
            db_exists = os.path.exists(str(CHROMA_PATH)) and os.path.exists(os.path.join(str(CHROMA_PATH), "chroma.sqlite3"))
            
            try:
                if db_exists:
                    # Load existing database
                    self._chroma_db = Chroma(
                        persist_directory=str(CHROMA_PATH),
                        embedding_function=self.embedding_function
                    )
                else:
                    # Create new database
                    self._chroma_db = Chroma(
                        persist_directory=str(CHROMA_PATH),
                        embedding_function=self.embedding_function
                    )
                    # Ensure the database is created
                    self._chroma_db.persist()
            except Exception as e:
                print(f"Error initializing Chroma database: {e}")
                raise
                
        return self._chroma_db

    def persist_chroma_db(self) -> None:
        """Explicitly persist the Chroma database.
        
        Raises:
            Exception: If there's an error persisting the database.
        """
        if self._chroma_db is not None:
            try:
                self._chroma_db.persist()
            except Exception as e:
                print(f"Error persisting Chroma database: {e}")
                raise

    @property
    def bm25_index(self) -> Optional[BM25Okapi]:
        """Get or initialize the BM25 index. Requires ChromaDB to be loaded.
        
        Returns:
            Optional[BM25Okapi]: The BM25 index, or None if ChromaDB is empty or fails.
        """
        if self._bm25_index is None:
            self._load_bm25_data()
        return self._bm25_index

    def _load_bm25_data(self) -> None:
        """Load data from ChromaDB and build the BM25 index."""
        try:
            # Ensure ChromaDB is loaded
            db = self.chroma_db
            if db is None:
                print("ChromaDB not available for BM25 indexing.")
                return

            # Retrieve all documents from ChromaDB
            # Note: This might be inefficient for very large databases.
            # Consider alternative strategies if performance is an issue.
            all_docs = db.get(include=["documents", "metadatas"])
            
            if not all_docs or not all_docs.get("ids"):
                print("No documents found in ChromaDB for BM25 indexing.")
                return

            self._bm25_corpus = all_docs["documents"]
            self._bm25_doc_ids = all_docs["ids"] # Use ChromaDB internal IDs
            
            if not self._bm25_corpus:
                print("Corpus is empty, cannot build BM25 index.")
                return
                
            # Simple tokenization (split by space)
            tokenized_corpus = [doc.split(" ") for doc in self._bm25_corpus]
            
            self._bm25_index = BM25Okapi(tokenized_corpus)
            print(f"BM25 index built successfully with {len(self._bm25_corpus)} documents.")

        except Exception as e:
            print(f"Error loading data for BM25 index: {e}")
            # Ensure attributes are reset if loading fails
            self._bm25_index = None
            self._bm25_corpus = None
            self._bm25_doc_ids = None

    @property
    def vectorstore(self) -> FAISS:
        """Get or initialize the FAISS vectorstore.
        
        Returns:
            FAISS: The FAISS vectorstore.
        """
        if self._vectorstore is None:
            self._vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                self.embedding_function,
                allow_dangerous_deserialization=True  # Safe since we created the file ourselves
            )
        return self._vectorstore

    @property
    def graphrag_graph(self) -> nx.DiGraph:
        """Get or initialize the GraphRAG graph.
        
        Returns:
            nx.DiGraph: The GraphRAG graph.
        """
        if self._graphrag_graph is None:
            with open(GRAPHRAG_GRAPH_PATH, 'r') as f:
                graph_data = json.load(f)
            
            self._graphrag_graph = nx.DiGraph()
            for node in graph_data['nodes']:
                self._graphrag_graph.add_node(node['id'], **node['data'])
            for edge in graph_data['edges']:
                self._graphrag_graph.add_edge(edge['source'], edge['target'], **edge['data'])
        return self._graphrag_graph

    @property
    def kag_graph(self) -> nx.DiGraph:
        """Get or initialize the KAG graph.
        
        Returns:
            nx.DiGraph: The KAG graph.
        """
        if self._kag_graph is None:
            with open(KAG_GRAPH_PATH, 'r') as f:
                graph_data = json.load(f)
            
            self._kag_graph = nx.DiGraph()
            for node in graph_data['nodes']:
                self._kag_graph.add_node(node['id'], **node['data'])
            for edge in graph_data['edges']:
                self._kag_graph.add_edge(edge['source'], edge['target'], **edge['data'])
        return self._kag_graph

    @property
    def qa_chain(self) -> RetrievalQA:
        """Get or initialize the QA chain.
        
        Returns:
            RetrievalQA: The QA chain.
        """
        if self._qa_chain is None:
            llm = CustomLLM()
            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
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
            try:
                # Load a pre-trained CrossEncoder model
                self._reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("Reranker model loaded successfully.")
            except Exception as e:
                print(f"Error loading reranker model: {e}")
                self._reranker = None
        return self._reranker

    def clear_cache(self) -> None:
        """Clear all cached resources."""
        self._embedding_function = None
        self._chroma_db = None
        self._vectorstore = None
        self._graphrag_graph = None
        self._kag_graph = None
        self._qa_chain = None
        self._bm25_index = None
        self._bm25_corpus = None
        self._bm25_doc_ids = None
        self._reranker = None

def initialize_data_service() -> None:
    """Initialize the data service by pre-loading resources."""
    print("Initializing data service...")
    # Access properties to trigger lazy loading
    _ = data_service.embedding_function
    _ = data_service.chroma_db
    _ = data_service.vectorstore
    _ = data_service.graphrag_graph
    _ = data_service.kag_graph
    _ = data_service.qa_chain
    _ = data_service.bm25_index # Trigger BM25 loading
    _ = data_service.reranker # Trigger reranker loading
    print("Data service initialized successfully!")

# Create a singleton instance
data_service = DataService()