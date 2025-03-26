"""
Data Service for managing RAG system resources.

This service handles loading and managing all data resources used by the RAG system,
including vector stores, graphs, and embedding functions. It provides a singleton
interface to access these resources efficiently.
"""

import os
import json
import networkx as nx
from typing import Optional, Dict, Any
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from get_embedding_function import get_embedding_function, LMStudioEmbeddings
from backend.database_paths import (
    CHROMA_PATH,
    KAG_GRAPH_PATH,
    GRAPHRAG_GRAPH_PATH,
    VECTORSTORE_PATH
)
from training.populate_lightrag import LMStudioLLM

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
            self._chroma_db = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embedding_function
            )
        return self._chroma_db

    @property
    def vectorstore(self) -> FAISS:
        """Get or initialize the FAISS vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = FAISS.load_local(
                VECTORSTORE_PATH,
                self.embedding_function
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
            llm = LMStudioLLM()
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

# Create a singleton instance
data_service = DataService() 