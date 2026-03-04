"""Retriever module for FAISS-based document retrieval with chunk metadata."""

import json
import logging
import re
from typing import Dict, List

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-based document retriever."""

    def __init__(self, config: Config):
        """Initialize the retriever."""
        self.config = config
        self.encoder = None
        self._load_encoder()

    def _load_encoder(self):
        """Load the sentence transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("[Retriever] loading embedding model: %s", self.config.emb_model)
            self.encoder = SentenceTransformer(
                self.config.emb_model,
                local_files_only=self.config.emb_model_is_local,
            )
            logger.info("[Retriever] embedding model ready")
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: "
                "pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            logger.error("Error loading embedding model: %s", exc)
            raise

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentence-like segments without external dependencies."""
        candidates = re.split(r"(?<=[.!?])\s+", text.strip())
        return [c.strip() for c in candidates if c.strip()]

    def _chunk_text(self, text: str) -> List[Dict[str, int | str]]:
        """Create overlapping chunks and preserve sentence boundaries when possible."""
        if not text.strip():
            return []

        size = self.config.chunk_size_chars
        overlap = max(0, min(self.config.chunk_overlap_chars, size - 1))

        sentences = self._split_sentences(text)
        if not sentences:
            sentences = [text.strip()]

        chunks: List[Dict[str, int | str]] = []
        current_parts: List[str] = []
        current_len = 0
        start_char = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            add_len = sentence_len + (1 if current_parts else 0)

            if current_parts and current_len + add_len > size:
                chunk_text = " ".join(current_parts).strip()
                end_char = start_char + len(chunk_text)
                chunks.append(
                    {
                        "text": chunk_text,
                        "start_char": start_char,
                        "end_char": end_char,
                    }
                )

                overlap_text = chunk_text[-overlap:] if overlap > 0 else ""
                current_parts = [overlap_text] if overlap_text else []
                current_len = len(overlap_text)
                start_char = max(0, end_char - current_len)

            if sentence_len >= size:
                if current_parts:
                    chunk_text = " ".join(current_parts).strip()
                    end_char = start_char + len(chunk_text)
                    chunks.append(
                        {
                            "text": chunk_text,
                            "start_char": start_char,
                            "end_char": end_char,
                        }
                    )
                    current_parts = []
                    current_len = 0
                    start_char = end_char

                for idx in range(0, sentence_len, size - overlap if size > overlap else size):
                    piece = sentence[idx : idx + size]
                    chunks.append(
                        {
                            "text": piece,
                            "start_char": idx,
                            "end_char": idx + len(piece),
                        }
                    )
                continue

            current_parts.append(sentence)
            current_len += add_len

        if current_parts:
            chunk_text = " ".join(current_parts).strip()
            end_char = start_char + len(chunk_text)
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start_char,
                    "end_char": end_char,
                }
            )

        return [c for c in chunks if c["text"]]

    def _build_chunks_from_docs(self, docs: List[str]) -> List[Dict[str, int | str]]:
        """Create chunk metadata records from input docs."""
        records: List[Dict[str, int | str]] = []
        chunk_id = 0

        for source_doc_id, doc in enumerate(docs):
            for chunk in self._chunk_text(doc):
                records.append(
                    {
                        "chunk_id": chunk_id,
                        "source_doc_id": source_doc_id,
                        "text": str(chunk["text"]),
                        "start_char": int(chunk["start_char"]),
                        "end_char": int(chunk["end_char"]),
                    }
                )
                chunk_id += 1

        return records

    @staticmethod
    def _normalize_scores(distances: np.ndarray) -> np.ndarray:
        """Convert L2 distances into a 0..1 relevance score."""
        if distances.size == 0:
            return distances

        clipped = np.clip(distances, 0.0, None)
        relevance = 1.0 / (1.0 + clipped)
        if relevance.max() == relevance.min():
            return np.ones_like(relevance)
        return (relevance - relevance.min()) / (relevance.max() - relevance.min())

    def _load_metadata(self) -> List[Dict[str, int | str]]:
        """Load chunk metadata from sidecar file."""
        if not self.config.meta_file.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.config.meta_file}. "
                "Please rebuild the index to generate chunk metadata."
            )

        with open(self.config.meta_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if not isinstance(data, list):
            raise ValueError(f"Unexpected metadata format in {self.config.meta_file}")

        return data

    def build_index(self) -> int:
        """Build FAISS index from chunked documents and save metadata."""
        if not self.config.docs_file.exists():
            raise FileNotFoundError(
                f"Documents file not found: {self.config.docs_file}\n"
                "Please create a docs.txt file with one document per line."
            )

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: "
                "pip install faiss-cpu"
            ) from exc

        logger.info("[Retriever] building index from %s", self.config.docs_file)
        with open(self.config.docs_file, "r", encoding="utf-8") as fh:
            docs = [line.strip() for line in fh if line.strip()]

        if not docs:
            raise ValueError(f"No documents found in {self.config.docs_file}")

        chunks = self._build_chunks_from_docs(docs)
        if not chunks:
            raise ValueError("No chunks generated from documents")

        texts = [str(chunk["text"]) for chunk in chunks]
        logger.info(
            "[Retriever] encoding %d chunk(s)",
            len(texts),
        )
        embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        dim = embeddings.shape[1]
        logger.info("[Retriever] creating FAISS IndexFlatL2 (dim=%d)", dim)
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        logger.info("[Retriever] saving index and metadata")
        faiss.write_index(index, str(self.config.index_file))
        with open(self.config.meta_file, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=True, indent=2)

        logger.info(
            "[Retriever] index ready | chunks=%d | index=%s | meta=%s",
            len(chunks),
            self.config.index_file,
            self.config.meta_file,
        )

        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, int | float | str]]:
        """Retrieve top-k chunks with scores and metadata."""
        if not self.config.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.config.index_file}\n"
                "Please run build_index() first."
            )

        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required. Install with: "
                "pip install faiss-cpu"
            ) from exc

        logger.info("[Retriever] retrieving chunks for query")
        index = faiss.read_index(str(self.config.index_file))
        chunks = self._load_metadata()

        query_embedding = self.encoder.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        fetch_k = min(max(self.config.fetch_k, 3 * k), index.ntotal)
        logger.info(
            "[Retriever] searching top-%d candidates (index size=%d)",
            fetch_k,
            index.ntotal,
        )
        distances, ids = index.search(query_embedding, fetch_k)

        candidate_distances = distances[0]
        candidate_ids = ids[0]
        relevance_scores = self._normalize_scores(candidate_distances)

        ranked_results: List[Dict[str, int | float | str]] = []
        seen_texts = set()
        total_chars = 0

        for raw_id, distance, score in zip(candidate_ids, candidate_distances, relevance_scores):
            if raw_id < 0 or raw_id >= len(chunks):
                continue
            if score < self.config.min_relevance_score:
                continue

            chunk = chunks[raw_id]
            text = str(chunk["text"]).strip()
            dedupe_key = text.lower()
            if not text or dedupe_key in seen_texts:
                continue

            if total_chars + len(text) > self.config.max_context_chars:
                continue

            seen_texts.add(dedupe_key)
            total_chars += len(text)
            ranked_results.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source_doc_id": int(chunk["source_doc_id"]),
                    "text": text,
                    "score": float(score),
                    "distance": float(distance),
                }
            )

            if len(ranked_results) >= k:
                break

        ranked_results.sort(key=lambda item: float(item["score"]), reverse=True)
        logger.info(
            "[Retriever] selected %d/%d chunk(s) after filtering (threshold=%.2f)",
            len(ranked_results),
            len(candidate_ids),
            self.config.min_relevance_score,
        )
        if candidate_distances.size > 0:
            logger.debug(
                "[Retriever] Candidate distances: %s",
                np.array2string(candidate_distances, precision=4),
            )

        return ranked_results
