"""Utility functions for the RAG system."""

import logging
from pathlib import Path
from typing import List


def setup_logging(level=logging.INFO):
    """
    Configure logging for the RAG system.
    
    Args:
        level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(Path(__file__).parent.parent.parent / "rag.log")
        ]
    )


def load_documents(docs_file: Path) -> List[str]:
    """
    Load documents from a file.
    
    Args:
        docs_file: Path to the documents file.
        
    Returns:
        List of document strings.
        
    Raises:
        FileNotFoundError: If the documents file doesn't exist.
    """
    if not docs_file.exists():
        raise FileNotFoundError(f"Documents file not found: {docs_file}")
    
    with open(docs_file, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    
    return docs


def save_documents(docs_file: Path, documents: List[str]):
    """
    Save documents to a file.
    
    Args:
        docs_file: Path where to save documents.
        documents: List of document strings.
    """
    docs_file.parent.mkdir(parents=True, exist_ok=True)
    with open(docs_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc.strip() + "\n")
