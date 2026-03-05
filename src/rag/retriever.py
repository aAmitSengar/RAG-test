"""Retriever module for FAISS-based document retrieval with chunk metadata."""

import json
import logging
import math
import re
from collections import Counter
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
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for lexical retrieval."""
        return re.findall(r"\b[a-zA-Z0-9]{2,}\b", (text or "").lower())

    @staticmethod
    def _normalize_array(values: np.ndarray) -> np.ndarray:
        """Normalize numeric array to 0..1 range."""
        if values.size == 0:
            return values
        min_v = float(values.min())
        max_v = float(values.max())
        if math.isclose(min_v, max_v):
            return np.zeros_like(values, dtype=float)
        return (values - min_v) / (max_v - min_v)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize user query with light cleanup."""
        text = re.sub(r"\s+", " ", (text or "").strip())
        text = re.sub(r"[?]{2,}", "?", text)
        return text

    def _rewrite_query(self, query: str) -> str:
        """Apply lightweight rule-based query rewriting/expansion."""
        if not getattr(self.config, "query_rewrite_enabled", True):
            return query

        cleaned = self._normalize_text(query)
        tokens = self._tokenize(cleaned)
        if not tokens:
            return cleaned

        expansions = {
            "rich": ["wealthy", "prosperous", "economy", "trade"],
            "poverty": ["poor", "low income", "deprivation"],
            "war": ["conflict", "battle", "military"],
            "independence": ["freedom", "sovereignty"],
            "founded": ["established", "started", "origin"],
            "ai": ["artificial intelligence", "machine learning"],
            "rag": ["retrieval augmented generation", "retriever", "generator"],
        }

        extras: List[str] = []
        for token in tokens:
            extras.extend(expansions.get(token, []))

        if not extras:
            return cleaned

        # Keep expansion bounded to avoid drifting query intent.
        extra_text = " ".join(dict.fromkeys(extras))[:120]
        return f"{cleaned} {extra_text}".strip()

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

    def _bm25_scores(self, query: str, chunks: List[Dict[str, int | str]]) -> np.ndarray:
        """Compute BM25-style sparse lexical scores for each chunk."""
        query_terms = self._tokenize(query)
        if not query_terms or not chunks:
            return np.zeros(len(chunks), dtype=float)

        tokenized_docs = [self._tokenize(str(chunk["text"])) for chunk in chunks]
        doc_lens = np.array([len(tokens) for tokens in tokenized_docs], dtype=float)
        avg_doc_len = float(doc_lens.mean()) if doc_lens.size else 1.0
        avg_doc_len = max(avg_doc_len, 1.0)

        n_docs = len(tokenized_docs)
        doc_freq = Counter()
        for tokens in tokenized_docs:
            doc_freq.update(set(tokens))

        k1 = 1.5
        b = 0.75
        scores = np.zeros(n_docs, dtype=float)

        for i, tokens in enumerate(tokenized_docs):
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = len(tokens)
            for term in query_terms:
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                df = doc_freq.get(term, 0)
                idf = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))
                denom = freq + k1 * (1 - b + b * (dl / avg_doc_len))
                scores[i] += idf * ((freq * (k1 + 1)) / max(denom, 1e-9))

        return scores

    def _compress_text_for_query(self, text: str, query: str) -> str:
        """Query-aware extractive compression for context budgeting."""
        if not text.strip():
            return text
        if not getattr(self.config, "context_compression_enabled", True):
            return text

        max_chars = int(getattr(self.config, "compression_max_chars", 420))
        max_sentences = int(getattr(self.config, "compression_max_sentences", 2))
        if len(text) <= max_chars:
            return text

        query_terms = set(self._tokenize(query))
        sentences = self._split_sentences(text)
        if not sentences:
            return text[:max_chars]

        def score(sentence: str) -> int:
            stokens = set(self._tokenize(sentence))
            term_hits = len(query_terms.intersection(stokens))
            number_bonus = 1 if re.search(r"\b\d{3,4}\b", sentence) else 0
            return term_hits + number_bonus

        ranked = sorted(sentences, key=score, reverse=True)
        selected = ranked[: max(1, max_sentences)]
        compressed = " ".join(selected).strip()
        if len(compressed) > max_chars:
            compressed = compressed[:max_chars].rstrip() + "..."
        return compressed

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
        rewritten_query = self._rewrite_query(query)
        if rewritten_query != query:
            logger.debug("[Retriever] rewritten query: '%s' -> '%s'", query, rewritten_query)

        query_embedding = self.encoder.encode(
            [rewritten_query],
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        fetch_k = min(max(int(getattr(self.config, "fetch_k", 10)), 3 * k), index.ntotal)
        logger.info(
            "[Retriever] searching top-%d candidates (index size=%d)",
            fetch_k,
            index.ntotal,
        )
        distances, ids = index.search(query_embedding, fetch_k)

        candidate_distances = distances[0]
        candidate_ids = ids[0]
        dense_scores = self._normalize_scores(candidate_distances)

        hybrid_enabled = bool(getattr(self.config, "hybrid_search_enabled", True))
        if hybrid_enabled:
            sparse_scores_all = self._bm25_scores(rewritten_query, chunks)
            sparse_scores_norm_all = self._normalize_array(sparse_scores_all)
            sparse_scores = np.array(
                [
                    sparse_scores_norm_all[candidate_id]
                    if 0 <= candidate_id < len(sparse_scores_norm_all)
                    else 0.0
                    for candidate_id in candidate_ids
                ],
                dtype=float,
            )
            dense_weight = float(getattr(self.config, "hybrid_dense_weight", 0.65))
            sparse_weight = float(getattr(self.config, "hybrid_sparse_weight", 0.35))
            weight_sum = max(dense_weight + sparse_weight, 1e-9)
            relevance_scores = (
                dense_weight * dense_scores + sparse_weight * sparse_scores
            ) / weight_sum
        else:
            sparse_scores = np.zeros_like(dense_scores)
            relevance_scores = dense_scores

        min_score = float(getattr(self.config, "min_relevance_score", 0.35))
        max_context_chars = int(getattr(self.config, "max_context_chars", 2500))
        candidate_pool: List[Dict[str, int | float | str]] = []

        for raw_id, distance, score, dense_score, sparse_score in zip(
            candidate_ids, candidate_distances, relevance_scores, dense_scores, sparse_scores
        ):
            if raw_id < 0 or raw_id >= len(chunks):
                continue
            if score < min_score:
                continue

            chunk = chunks[raw_id]
            text = str(chunk["text"]).strip()
            if not text:
                continue
            candidate_pool.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source_doc_id": int(chunk["source_doc_id"]),
                    "text": text,
                    "score": float(score),
                    "dense_score": float(dense_score),
                    "sparse_score": float(sparse_score),
                    "distance": float(distance),
                }
            )

        candidate_pool.sort(key=lambda item: float(item["score"]), reverse=True)

        ranked_results: List[Dict[str, int | float | str]] = []
        seen_texts = set()
        total_chars = 0
        for item in candidate_pool:
            text = self._compress_text_for_query(str(item["text"]), rewritten_query)
            dedupe_key = text.lower()
            if not text or dedupe_key in seen_texts:
                continue
            if total_chars + len(text) > max_context_chars:
                continue

            item["text"] = text
            seen_texts.add(dedupe_key)
            total_chars += len(text)
            ranked_results.append(item)
            if len(ranked_results) >= k:
                break

        logger.info(
            "[Retriever] selected %d/%d chunk(s) after filtering (threshold=%.2f)",
            len(ranked_results),
            len(candidate_ids),
            min_score,
        )
        if candidate_distances.size > 0:
            logger.debug(
                "[Retriever] Candidate distances: %s",
                np.array2string(candidate_distances, precision=4),
            )

        return ranked_results
