"""Configuration module for RAG system."""

import os
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)


class Config:
    """Centralized configuration management for the RAG system."""

    def __init__(self):
        """Initialize configuration from environment and project structure."""
        logger.info("[Config] Initializing configuration")

        # 1) Resolve important project directories.
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        
        # 2) Ensure required folders exist before any file/model operations.
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # 3) Define data file locations used by retriever and pipeline.
        self.docs_file = self.data_dir / "docs.txt"
        self.index_file = self.data_dir / "faiss.index"
        
        # 4) Resolve embedding and generation model source.
        # Priority: env var > local folder > Hugging Face default id.
        self.emb_model = self._resolve_model_path(
            os.getenv("EMB_MODEL"),
            self.models_dir / "all-MiniLM-L6-v2",
            "all-MiniLM-L6-v2"
        )
        
        self.gen_model = self._resolve_model_path(
            os.getenv("GEN_MODEL"),
            self.models_dir / "t5-small",
            "t5-small"
        )
        
        # 5) Configure SSL cert paths (especially helpful on macOS).
        self._setup_ssl_certificates()
        
        # 6) Runtime knobs for retrieval behavior.
        self.retrieval_k = int(os.getenv("RETRIEVAL_K", "3"))
        self.use_local_only = os.getenv("USE_LOCAL_ONLY", "false").lower() == "true"

        logger.info("[Config] ✓ Configuration ready")
        logger.info(
            "[Config] docs=%s | index=%s | emb_model=%s | gen_model=%s",
            self.docs_file,
            self.index_file,
            self.emb_model,
            self.gen_model,
        )

    @staticmethod
    def _resolve_model_path(
        env_model: Optional[str],
        local_path: Path,
        default_model: str
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
            pass

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
            f"  emb_model={self.emb_model},\n"
            f"  gen_model={self.gen_model},\n"
            f"  use_local_only={self.use_local_only}\n"
            f")"
        )
