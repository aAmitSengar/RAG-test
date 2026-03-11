"""Configuration module for RAG system."""

import logging
import os
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration management for the RAG system."""

    def __init__(self):
        """Initialize configuration from environment and project structure.

        Teaching points:
        - We allow users to control behavior without touching code (12-factor style).
        - Paths are resolved first so later components can rely on them.
        - Many RAG-quality knobs (chunk sizes, hybrid weights) are env-driven.
        """
        logger.info("[Config] Initializing configuration")

        # --- Project structure ---
        self.project_root = Path(__file__).parent.parent.parent

        # Prefer DATA_DIR from .env; else default to ./data under project root
        env_data_dir = os.getenv("DATA_DIR")
        self.data_dir = Path(env_data_dir).resolve() if env_data_dir else (self.project_root / "data")
        self.models_dir = self.project_root / "models"
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)

        # --- Data artifacts ---
        self.docs_file = self.data_dir / "docs.txt"
        self.index_file = self.data_dir / "faiss.index"
        self.meta_file = self.data_dir / "faiss_meta.json"
        # Saved dense vectors for MMR/diversity selection
        self.vecs_file = self.data_dir / "faiss_vecs.npy"

        # --- Model sources (env > local folder > default HF id) ---
        self.emb_model = self._resolve_model_path(
            os.getenv("EMB_MODEL"),
            self.models_dir / "all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2",
        )
        self.gen_model = self._resolve_model_path(
            os.getenv("GEN_MODEL"),
            self.models_dir / "t5-small",
            "t5-small",
        )

        # --- SSL (for environments that need explicit certs) ---
        self._setup_ssl_certificates()

        # --- Retrieval knobs ---
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "5"))
        self.chunk_size_chars = int(os.getenv("CHUNK_SIZE_CHARS", "700"))
        self.chunk_overlap_chars = int(os.getenv("CHUNK_OVERLAP_CHARS", "120"))
        self.fetch_k = int(os.getenv("FETCH_K", "10"))
        self.min_relevance_score = float(os.getenv("MIN_RELEVANCE_SCORE", "0.35"))
        self.max_context_chars = int(os.getenv("MAX_CONTEXT_CHARS", "2500"))
        self.hybrid_search_enabled = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
        self.hybrid_dense_weight = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.65"))
        self.hybrid_sparse_weight = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.35"))
        self.query_rewrite_enabled = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"
        self.context_compression_enabled = (
            os.getenv("CONTEXT_COMPRESSION_ENABLED", "true").lower() == "true"
        )
        self.compression_max_sentences = int(os.getenv("COMPRESSION_MAX_SENTENCES", "2"))
        self.compression_max_chars = int(os.getenv("COMPRESSION_MAX_CHARS", "420"))

        # --- Generation knobs ---
        self.citations_enabled = os.getenv("CITATIONS_ENABLED", "true").lower() == "true"
        self.do_sample = os.getenv("DO_SAMPLE", "false").lower() == "true"
        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "200"))
        self.answer_language = os.getenv("ANSWER_LANGUAGE", "Hinglish")
        self.persona = os.getenv("PERSONA", "You are a knowledgeable history teacher.")
        self.step_by_step_mode = os.getenv("STEP_BY_STEP_MODE", "false").lower() == "false"

        # --- Local-only & scoring strategies ---
        self.use_local_only = os.getenv("USE_LOCAL_ONLY", "false").lower() == "true"
        self.use_index_ip_cosine = os.getenv("USE_INDEX_IP_COSINE", "true").lower() == "true"
        self.mmr_lambda = float(os.getenv("MMR_LAMBDA", "0.7"))
        self.rerank_enabled = os.getenv("RERANK_ENABLED", "false").lower() == "true"

        # --- Logging level (applies to root logger) ---
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.getLogger().setLevel(level)

        logger.info(
            "[Config] ready | docs=%s | index=%s | meta=%s | emb_model=%s | gen_model=%s",
            self.docs_file, self.index_file, self.meta_file, self.emb_model, self.gen_model
        )

    @staticmethod
    def _resolve_model_path(
        env_model: Optional[str], local_path: Path, default_model: str
    ) -> str:
        """
        Resolve the model path with priority:
        1. Environment variable if set
        2. Local path if it exists
        3. Default HF model name
        """
        if env_model:
            return env_model

        if local_path.exists():
            return str(local_path)

        return default_model

    @staticmethod
    def _setup_ssl_certificates():
        """Set up SSL certificates, especially important for macOS."""
        try:
            import certifi

            cert_path = certifi.where()
            os.environ["SSL_CERT_FILE"] = cert_path
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
            os.environ.setdefault("CURL_CA_BUNDLE", cert_path)
            logger.debug("[Config] SSL certificates configured via certifi")
        except ImportError:
            # certifi not installed, SSL will use system defaults
            logger.debug("[Config] certifi not installed; using system SSL defaults")

    @property
    def emb_model_is_local(self) -> bool:
        """Check if embedding model is a local path."""
        return Path(self.emb_model).exists()

    @property
    def gen_model_is_local(self) -> bool:
        """Check if generation model is a local path."""
        return Path(self.gen_model).exists()

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  project_root={self.project_root},\n"
            f"  docs_file={self.docs_file},\n"
            f"  index_file={self.index_file},\n"
            f"  meta_file={self.meta_file},\n"
            f"  emb_model={self.emb_model},\n"
            f"  gen_model={self.gen_model},\n"
            f"  retrieval_k={self.retrieval_k},\n"
            f"  chunk_size_chars={self.chunk_size_chars},\n"
            f"  chunk_overlap_chars={self.chunk_overlap_chars},\n"
            f"  fetch_k={self.fetch_k},\n"
            f"  min_relevance_score={self.min_relevance_score},\n"
            f"  max_context_chars={self.max_context_chars},\n"
            f"  citations_enabled={self.citations_enabled},\n"
            f"  do_sample={self.do_sample},\n"
            f"  step_by_step_mode={self.step_by_step_mode},\n"
            f"  use_local_only={self.use_local_only},\n"
            f"  hybrid_search_enabled={self.hybrid_search_enabled},\n"
            f"  hybrid_dense_weight={self.hybrid_dense_weight},\n"
            f"  hybrid_sparse_weight={self.hybrid_sparse_weight},\n"
            f"  query_rewrite_enabled={self.query_rewrite_enabled},\n"
            f"  context_compression_enabled={self.context_compression_enabled},\n"
            f"  compression_max_sentences={self.compression_max_sentences},\n"
            f"  compression_max_chars={self.compression_max_chars}\n"
            f")"
        )
