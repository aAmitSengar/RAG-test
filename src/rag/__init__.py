"""RAG (Retrieval-Augmented Generation) package.

This package provides utilities for building a simple RAG system using
FAISS for retrieval and transformers for generation.
"""

__version__ = "0.1.0"

from .config import Config
from .retriever import Retriever
from .generator import Generator

__all__ = ["Config", "Retriever", "Generator"]
