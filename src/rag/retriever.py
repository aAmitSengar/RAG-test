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
        logger.info("[Retriever] Initializing retriever component")
        self._load_encoder()

    def _load_encoder(self):
        """Load the sentence transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info("[Retriever][Step 1/4] Loading embedding model")
            logger.info(f"[Retriever] Model source: {self.config.emb_model}")
            self.encoder = SentenceTransformer(
                self.config.emb_model,
                local_files_only=self.config.emb_model_is_local
            )
            logger.info("[Retriever] ✓ Encoder loaded successfully")
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

        # STEP A: Load raw documents from docs.txt.
        # Each non-empty line is treated as one retrievable unit.
        logger.info("[Retriever][Index Build][Step A/4] Loading documents")
        with open(self.config.docs_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        if not texts:
            raise ValueError(f"No documents found in {self.config.docs_file}")

        logger.info(
            f"[Retriever][Index Build][Step B/4] Encoding {len(texts)} document(s) into vectors"
        )
        
        # STEP B: Convert text to dense vectors via sentence-transformer encoder.
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # STEP C: Create FAISS index and add all embeddings.
        dim = embeddings.shape[1]
        logger.info(f"[Retriever][Index Build][Step C/4] Creating FAISS IndexFlatL2 (dim={dim})")
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        
        # STEP D: Persist index to disk for later retrieval calls.
        logger.info("[Retriever][Index Build][Step D/4] Saving index to disk")
        faiss.write_index(index, str(self.config.index_file))
        logger.info(f"[Retriever] ✓ Index saved to {self.config.index_file}")
        
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
            # STEP 1: Load precomputed FAISS index.
            logger.info("[Retriever][Query][Step 1/4] Loading FAISS index")
            index = faiss.read_index(str(self.config.index_file))
            
            # STEP 2: Convert user query into an embedding vector.
            logger.info("[Retriever][Query][Step 2/4] Encoding query")
            query_embedding = self.encoder.encode(
                [query],
                convert_to_numpy=True
            )
            
            # STEP 3: Search nearest vectors in FAISS.
            k = min(k, index.ntotal)
            logger.info(
                f"[Retriever][Query][Step 3/4] Searching top-{k} result(s) in index with {index.ntotal} item(s)"
            )
            distances, ids = index.search(query_embedding, k)
            
            # STEP 4: Map FAISS ids back to the original document strings.
            logger.info("[Retriever][Query][Step 4/4] Resolving ids to source documents")
            with open(self.config.docs_file, "r", encoding="utf-8") as f:
                docs = [line.strip() for line in f if line.strip()]
            
            # Keep only valid indices (guard against edge-case id mismatches).
            results = [docs[i] for i in ids[0] if i < len(docs)]
            logger.info(f"[Retriever] Retrieved {len(results)} document(s)")
            logger.debug(f"[Retriever] Distances: {np.array2string(distances[0], precision=4)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            raise
