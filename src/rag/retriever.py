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

# TODO// sutem prompt // user prompt // persona as a history teacher

class Retriever:
    """FAISS-based document retriever."""

    def __init__(self, config: Config):
        """Initialize the retriever."""
        self.config = config
        self.encoder = None
        self.reranker = None
        self._load_encoder()
        self._load_reranker()

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

    def _load_reranker(self):
        """Load the optional cross-encoder reranker.
        
        Currently not implemented; reranker remains None.
        This allows optional reranking to be skipped without errors.
        """
        self.reranker = None

    
    def _normalize_ip_scores(self, sims: np.ndarray) -> np.ndarray:
        """Min-max normalize inner-product (cosine) similarities to 0..1.

        Teaching points:
        - After switching to IndexFlatIP, FAISS returns *similarities* (higher is better).
        - We normalize to combine with BM25 scores (common 0..1 scale).
        """
        if sims.size == 0:
            return sims
        s_min, s_max = float(sims.min()), float(sims.max())
        if math.isclose(s_min, s_max):
            return np.ones_like(sims)
        return (sims - s_min) / (s_max - s_min)

    def _load_dense_matrix_if_available(self) -> np.ndarray | None:
        """Load the saved dense embedding matrix for MMR (diversity).

        Teaching points:
        - MMR needs the actual vectors to estimate redundancy between candidates.
        - We save these at index-build time for fast selection at query time.
        """
        try:
            return np.load(self.config.vecs_file)
        except Exception:
            return None
        
    def _mmr_order(self, cand_ids: np.ndarray, query_vec: np.ndarray,
                lambda_mult: float, vecs: np.ndarray | None) -> List[int]:
        """Reorder candidate ids via Maximal Marginal Relevance (MMR).

        Args:
            cand_ids: array of candidate row ids returned by FAISS.
            query_vec: the query embedding vector (shape: (D,) or (1, D)).
            lambda_mult: [0..1] balance between relevance (to query) and diversity.
            vecs: cached dense matrix (N x D) for all indexed chunks.

        Returns:
            A list of candidate ids ordered to maximize coverage & reduce redundancy.

        Teaching points:
        - First pick the most relevant; then each next pick maximizes (lambda * relevance - (1-lambda) * redundancy).
        - Works best when vectors are L2-normalized (cosine).
        """
        if vecs is None or cand_ids.size == 0:
            return list(map(int, cand_ids))

        cand_vecs = vecs[cand_ids]
        q = query_vec.reshape(1, -1)  # (1, D)

        # Relevance: cosine similarity to query
        q_sims = (cand_vecs @ q.T).reshape(-1, 1)  # (N, 1)

        selected: List[int] = []
        remaining = list(range(len(cand_ids)))

        while remaining:
            if not selected:
                # First pick = best relevance
                best_idx = int(np.argmax(q_sims[remaining]))
                selected.append(remaining.pop(best_idx))
                continue

            sel_vecs = cand_vecs[[i for i in selected]]  # (S, D)
            # Redundancy: max sim to any already selected item
            sim_to_sel = cand_vecs[remaining] @ sel_vecs.T  # (N_rem, S)
            max_sim_to_sel = sim_to_sel.max(axis=1, keepdims=True)  # (N_rem, 1)

            # MMR objective
            mmr_scores = lambda_mult * q_sims[remaining] - (1.0 - lambda_mult) * max_sim_to_sel
            best_idx = int(np.argmax(mmr_scores))
            selected.append(remaining.pop(best_idx))

        return [int(cand_ids[i]) for i in selected]

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
            """Build FAISS index from docs.txt, with sentence-aware chunking.

            Pipeline:
            docs.txt -> chunks -> embeddings -> (normalize if IP) -> FAISS index
            Also saves: chunk metadata (JSON) + dense matrix (npy) for MMR.

            Teaching points:
            - IndexFlatIP + normalized vectors = cosine similarity.
            - Saving dense matrix once makes MMR selection cheap at query time.
            """
            if not self.config.docs_file.exists():
                raise FileNotFoundError(
                    f"Documents file not found: {self.config.docs_file}\n"
                    "Please create a docs.txt file with one document per line."
                )

            try:
                import faiss
            except ImportError as exc:
                raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

            logger.info("[Retriever] building index from %s", self.config.docs_file)
            with open(self.config.docs_file, "r", encoding="utf-8") as fh:
                docs = [line.strip() for line in fh if line.strip()]
            if not docs:
                raise ValueError(f"No documents found in {self.config.docs_file}")

            # 1) Chunk the documents with overlap
            chunks = self._build_chunks_from_docs(docs)
            if not chunks:
                raise ValueError("No chunks generated from documents")

            # 2) Embed
            texts = [str(chunk["text"]) for chunk in chunks]
            logger.info("[Retriever] encoding %d chunk(s)", len(texts))
            embeddings = self.encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)

            # 3) Build FAISS index (IP for cosine, else L2)
            import faiss  # safe, already checked
            dim = embeddings.shape[1]
            use_ip = bool(getattr(self.config, "use_index_ip_cosine", True))
            if use_ip:
                # Cosine similarity via inner product requires unit vectors
                faiss.normalize_L2(embeddings)
                logger.info("[Retriever] creating FAISS IndexFlatIP (cosine), dim=%d", dim)
                index = faiss.IndexFlatIP(dim)
            else:
                logger.info("[Retriever] creating FAISS IndexFlatL2 (euclidean), dim=%d", dim)
                index = faiss.IndexFlatL2(dim)
            index.add(embeddings)

            # 4) Persist index + metadata + dense matrix (for MMR)
            logger.info("[Retriever] saving index and metadata")
            faiss.write_index(index, str(self.config.index_file))
            with open(self.config.meta_file, "w", encoding="utf-8") as fh:
                json.dump(chunks, fh, ensure_ascii=True, indent=2)
            try:
                np.save(self.config.vecs_file, embeddings)
                logger.info("[Retriever] saved dense matrix for MMR: %s", self.config.vecs_file)
            except Exception as exc:
                logger.warning("[Retriever] failed to save dense matrix for MMR: %s", exc)

            logger.info(
                "[Retriever] index ready | chunks=%d | index=%s | meta=%s",
                len(chunks), self.config.index_file, self.config.meta_file
            )
            return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, int | float | str]]:
        """Retrieve top-k chunks for a query using hybrid scoring + diversity.

        Steps:
        1) Read FAISS index + chunk metadata.
        2) Rewrite query lightly for better recall (optional).
        3) Encode + (if IP) L2-normalize query vector; search FAISS (fetch_k).
        4) Normalize dense scores (cosine path differs from L2 path).
        5) Apply MMR to reduce redundancy (uses saved dense matrix).
        6) If hybrid: combine dense & BM25 sparse scores (weighted).
        7) Threshold, compress by query, dedupe, respect context budget.
        8) (Optional) Cross-encoder rerank for precision.

        Teaching points:
        - Hybrid search = semantic + lexical => better for history/logs.
        - MMR ensures diverse evidence, not 5 near-duplicate slices.
        """
        if not self.config.index_file.exists():
            raise FileNotFoundError(
                f"Index file not found: {self.config.index_file}\n"
                "Please run build_index() first."
            )

        try:
            import faiss
        except ImportError as exc:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu") from exc

        logger.info("[Retriever] retrieving chunks for query")
        index = faiss.read_index(str(self.config.index_file))
        chunks = self._load_metadata()

        # 1) Rewrite query (cheap lexical/semantic expansions)
        rewritten_query = self._rewrite_query(query)
        if rewritten_query != query:
            logger.debug("[Retriever] rewritten query: '%s' -> '%s'", query, rewritten_query)

        # 2) Encode query (normalize if IP)
        query_embedding = self.encoder.encode([rewritten_query], show_progress_bar=False, convert_to_numpy=True)
        use_ip = bool(getattr(self.config, "use_index_ip_cosine", True))
        if use_ip:
            faiss.normalize_L2(query_embedding)

        # 3) FAISS search (get a bigger candidate pool than final K)
        fetch_k = min(max(int(getattr(self.config, "fetch_k", 10)), 3 * k), index.ntotal)
        logger.info("[Retriever] searching top-%d candidates (index size=%d)", fetch_k, index.ntotal)
        distances, ids = index.search(query_embedding, fetch_k)

        # 4) Dense score normalization (IP returns sims; L2 returns distances)
        candidate_ids = ids[0]
        if use_ip:
            sims = distances[0]                 # higher is better
            dense_scores = self._normalize_ip_scores(sims)
        else:
            dists = distances[0]                # lower is better
            dense_scores = self._normalize_scores(dists)

        # 5) MMR diversity ordering
        mmr_lambda = float(getattr(self.config, "mmr_lambda", 0.7))
        vecs = self._load_dense_matrix_if_available()
        candidate_ids = np.array(self._mmr_order(candidate_ids, query_embedding[0], mmr_lambda, vecs))

        # 6) Hybrid fusion (semantic + BM25 lexical)
        hybrid_enabled = bool(getattr(self.config, "hybrid_search_enabled", True))
        if hybrid_enabled:
            sparse_scores_all = self._bm25_scores(rewritten_query, chunks)  # scores for all chunks
            sparse_scores_norm_all = self._normalize_array(sparse_scores_all)
            # Align sparse scores to the FAISS candidate subset
            sparse_scores = np.array(
                [sparse_scores_norm_all[cid] if 0 <= cid < len(sparse_scores_norm_all) else 0.0
                for cid in candidate_ids],
                dtype=float,
            )
            dense_weight = float(getattr(self.config, "hybrid_dense_weight", 0.65))
            sparse_weight = float(getattr(self.config, "hybrid_sparse_weight", 0.35))
            weight_sum = max(dense_weight + sparse_weight, 1e-9)
            relevance_scores = (dense_weight * dense_scores + sparse_weight * sparse_scores) / weight_sum
        else:
            sparse_scores = np.zeros_like(dense_scores)
            relevance_scores = dense_scores

        # 7) Filter by score, compress context, dedupe, and respect char budget
        min_score = float(getattr(self.config, "min_relevance_score", 0.35))
        max_context_chars = int(getattr(self.config, "max_context_chars", 2500))

        candidate_pool: List[Dict[str, int | float | str]] = []
        for idx, raw_id in enumerate(candidate_ids):
            if raw_id < 0 or raw_id >= len(chunks):
                continue
            score = float(relevance_scores[idx])
            if score < min_score:
                continue
            dense_score = float(dense_scores[idx])
            sparse_score = float(sparse_scores[idx])
            chunk = chunks[raw_id]
            text = str(chunk["text"]).strip()
            if not text:
                continue
            candidate_pool.append(
                {
                    "chunk_id": int(chunk["chunk_id"]),
                    "source_doc_id": int(chunk["source_doc_id"]),
                    "text": text,               # will be compressed below
                    "score": score,
                    "dense_score": dense_score,
                    "sparse_score": sparse_score,
                }
            )

        # Sort by combined score
        candidate_pool.sort(key=lambda item: float(item["score"]), reverse=True)

        # Query-aware compression + dedupe + budget control
        ranked_results: List[Dict[str, int | float | str]] = []
        seen_texts = set()
        total_chars = 0
        for item in candidate_pool:
            text = self._compress_text_for_query(str(item["text"]), rewritten_query)
            key = text.lower()
            if not text or key in seen_texts:
                continue
            if total_chars + len(text) > max_context_chars:
                continue

            item["text"] = text
            seen_texts.add(key)
            total_chars += len(text)
            ranked_results.append(item)
            if len(ranked_results) >= k:
                break

        logger.info(
            "[Retriever] selected %d/%d chunk(s) after filtering (threshold=%.2f)",
            len(ranked_results), len(candidate_ids), min_score
        )

        # 8) Optional cross-encoder reranking for extra precision
        if self.reranker and ranked_results:
            try:
                pairs = [(rewritten_query, it["text"]) for it in ranked_results]
                scores = self.reranker.predict(pairs)
                for it, s in zip(ranked_results, scores):
                    it["rerank_score"] = float(s)
                ranked_results = sorted(ranked_results, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                logger.info("[Retriever] reranked %d item(s) via cross-encoder", len(ranked_results))
            except Exception as exc:
                logger.warning("[Retriever] reranker failed: %s (continuing without)", exc)

        return ranked_results