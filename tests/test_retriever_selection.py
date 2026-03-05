"""Tests for retrieval ranking/filtering logic."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from rag.retriever import Retriever


class _FakeIndex:
    def __init__(self):
        self.ntotal = 4

    def search(self, query_embedding, fetch_k):
        assert fetch_k >= 3
        distances = np.array([[0.1, 0.5, 3.0, 5.0]], dtype=float)
        ids = np.array([[0, 1, 2, 3]], dtype=int)
        return distances, ids


class _FakeFaiss:
    @staticmethod
    def read_index(path):
        return _FakeIndex()


def _make_retriever(tmp_path: Path) -> Retriever:
    retriever = object.__new__(Retriever)
    retriever.config = SimpleNamespace(
        index_file=tmp_path / "faiss.index",
        fetch_k=10,
        min_relevance_score=0.35,
        max_context_chars=80,
    )
    retriever.config.index_file.write_text("dummy", encoding="utf-8")
    retriever.encoder = SimpleNamespace(
        encode=lambda *args, **kwargs: np.array([[1.0, 2.0]], dtype=float)
    )
    retriever._load_metadata = lambda: [
        {"chunk_id": 0, "source_doc_id": 0, "text": "alpha context"},
        {"chunk_id": 1, "source_doc_id": 0, "text": "beta context"},
        {"chunk_id": 2, "source_doc_id": 1, "text": "gamma weak"},
        {"chunk_id": 3, "source_doc_id": 1, "text": "delta weak"},
    ]
    return retriever


def test_retrieve_filters_and_sorts(monkeypatch, tmp_path):
    retriever = _make_retriever(tmp_path)
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    results = retriever.retrieve("What is retrieval?", k=3)

    # Last two ids should be filtered by score threshold.
    assert len(results) == 2
    assert [result["chunk_id"] for result in results] == [0, 1]
    assert results[0]["score"] >= results[1]["score"]


def test_retrieve_respects_context_budget(monkeypatch, tmp_path):
    retriever = _make_retriever(tmp_path)
    retriever.config.max_context_chars = 14
    monkeypatch.setitem(sys.modules, "faiss", _FakeFaiss)

    results = retriever.retrieve("budget", k=3)

    # Only one chunk should fit under char budget.
    assert len(results) == 1
    assert results[0]["chunk_id"] == 0


def test_query_rewrite_adds_synonyms_when_enabled(tmp_path):
    retriever = _make_retriever(tmp_path)
    retriever.config.query_rewrite_enabled = True

    rewritten = retriever._rewrite_query("Was India rich?")

    assert "wealthy" in rewritten.lower()
    assert "prosperous" in rewritten.lower()


def test_hybrid_search_can_boost_lexical_match(monkeypatch, tmp_path):
    class _LocalIndex:
        ntotal = 2

        @staticmethod
        def search(query_embedding, fetch_k):
            distances = np.array([[0.01, 0.6]], dtype=float)  # dense prefers chunk 0
            ids = np.array([[0, 1]], dtype=int)
            return distances, ids

    class _LocalFaiss:
        @staticmethod
        def read_index(path):
            return _LocalIndex()

    retriever = object.__new__(Retriever)
    retriever.config = SimpleNamespace(
        index_file=tmp_path / "faiss.index",
        fetch_k=2,
        min_relevance_score=0.0,
        max_context_chars=500,
        hybrid_search_enabled=True,
        hybrid_dense_weight=0.2,
        hybrid_sparse_weight=0.8,
        query_rewrite_enabled=False,
        context_compression_enabled=False,
    )
    retriever.config.index_file.write_text("dummy", encoding="utf-8")
    retriever.encoder = SimpleNamespace(
        encode=lambda *args, **kwargs: np.array([[1.0, 2.0]], dtype=float)
    )
    retriever._load_metadata = lambda: [
        {"chunk_id": 0, "source_doc_id": 0, "text": "generic summary"},
        {"chunk_id": 1, "source_doc_id": 0, "text": "roman trade wealth in india"},
    ]
    monkeypatch.setitem(sys.modules, "faiss", _LocalFaiss)

    results = retriever.retrieve("trade wealth", k=1)

    assert len(results) == 1
    assert results[0]["chunk_id"] == 1
