"""Retriever module for FAISS-based document retrieval."""

import logging
from typing import List
from pathlib import Path

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-based document retriever."""

    def __init__(self, config: Config):
        """
        Initialize the retriever.
        
        Args:
            config: Configuration object containing model and file paths.
        """
        self.config = config
        self.encoder = None
        self._load_encoder()

    def _load_encoder(self):
        """Load the sentence transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.config.emb_model}")
            self.encoder = SentenceTransformer(
                self.config.emb_model,
                local_files_only=self.config.emb_model_is_local
            )
            logger.info("✓ Encoder loaded successfully")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def build_index(self) -> int:
        """
        Build FAISS index from documents.
        
        Returns:
            Number of indexed documents.
            
        Raises:
            FileNotFoundError: If docs.txt does not exist.
        """
        if not self.config.docs_file.exists():
            raise FileNotFoundError(
                f"Documents file not found: {self.config.docs_file}\n"
                "Please create a docs.txt file with one document per line."
            )

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: "
                "pip install faiss-cpu"
            )

        # Load documents
        with open(self.config.docs_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            raise ValueError(f"No documents found in {self.config.docs_file}")

        logger.info(f"Building FAISS index from {len(texts)} documents...")
        
        # Encode all documents
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Create and populate index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # Save index
        faiss.write_index(index, str(self.config.index_file))
        logger.info(f"✓ Index saved to {self.config.index_file}")
        
        return len(texts)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve top-k documents relevant to the query.
        
        Args:
            query: Search query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of relevant document snippets.
            
        Raises:
            FileNotFoundError: If index file does not exist.
        """
        if not self.config.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.config.index_file}\n"
                "Please run build_index() first."
            )

        try:
            import faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required. Install with: "
                "pip install faiss-cpu"
            )

        try:
            # Load index
            index = faiss.read_index(str(self.config.index_file))
            
            # Encode query
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True
            )
            
            # Search
            k = min(k, index.ntotal)
            distances, ids = index.search(query_embedding, k)
            
            # Load documents
            with open(self.config.docs_file, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
            
            # Retrieve results
            results = [docs[i] for i in ids[0] if i < len(docs)]
            logger.info(f"Retrieved {len(results)} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
