"""
Data Service for managing RAG system resources.

This service handles loading and managing all data resources used by the RAG system,
including vector stores, graphs, and embedding functions. It provides a singleton
interface to access these resources efficiently.
"""
import json
import networkx as nx
import os
from typing import Optional
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from query.database_paths import CHROMA_PATH, GRAPHRAG_GRAPH_PATH, KAG_GRAPH_PATH, VECTORSTORE_PATH
from query.llm_service import get_llm_response
from training.get_embedding_function import get_embedding_function, LMStudioEmbeddings

class CustomLLM(LLM):
    """Custom LLM class that uses our LLM service."""
    
    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        return get_llm_response(prompt)

    @property
    def _llm_type(self) -> str:
        return "custom"

class DataService:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._embedding_function = None
            self._chroma_db = None
            self._vectorstore = None
            self._graphrag_graph = None
            self._kag_graph = None
            self._qa_chain = None
            self._initialized = True

    @property
    def embedding_function(self) -> LMStudioEmbeddings:
        """Get or initialize the embedding function."""
        if self._embedding_function is None:
            self._embedding_function = get_embedding_function()
        return self._embedding_function

    @property
    def chroma_db(self) -> Chroma:
        """Get or initialize the Chroma database."""
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

    def persist_chroma_db(self):
        """Explicitly persist the Chroma database."""
        if self._chroma_db is not None:
            try:
                self._chroma_db.persist()
            except Exception as e:
                print(f"Error persisting Chroma database: {e}")
                raise

    @property
    def vectorstore(self) -> FAISS:
        """Get or initialize the FAISS vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                self.embedding_function,
                allow_dangerous_deserialization=True  # Safe since we created the file ourselves
            )
        return self._vectorstore

    @property
    def graphrag_graph(self) -> nx.DiGraph:
        """Get or initialize the GraphRAG graph."""
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
        """Get or initialize the KAG graph."""
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
        """Get or initialize the QA chain."""
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

    def clear_cache(self):
        """Clear all cached resources."""
        self._embedding_function = None
        self._chroma_db = None
        self._vectorstore = None
        self._graphrag_graph = None
        self._kag_graph = None
        self._qa_chain = None

def initialize_data_service():
    """Initialize the data service by pre-loading resources."""
    print("Initializing data service...")
    # Access properties to trigger lazy loading
    _ = data_service.embedding_function
    _ = data_service.chroma_db
    _ = data_service.vectorstore
    _ = data_service.graphrag_graph
    _ = data_service.kag_graph
    _ = data_service.qa_chain
    print("Data service initialized successfully!")

# Create a singleton instance
data_service = DataService() 