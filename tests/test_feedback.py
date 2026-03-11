"""Tests for the FeedbackStore correction-learning loop."""

import json
from pathlib import Path

import pytest

from rag.feedback import FeedbackStore


@pytest.fixture()
def store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path)


def test_record_creates_corrections_and_log(store: FeedbackStore, tmp_path: Path):
    store.record(
        question="What is the capital of France?",
        wrong_answer="Lyon",
        correct_answer="The capital of France is Paris.",
    )

    # corrections.txt should have a Q→A line
    lines = store.corrections_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "Q: What is the capital of France?" in lines[0]
    assert "A: The capital of France is Paris." in lines[0]

    # feedback.jsonl should have a JSON entry
    log_lines = store.feedback_log.read_text(encoding="utf-8").splitlines()
    assert len(log_lines) == 1
    entry = json.loads(log_lines[0])
    assert entry["question"] == "What is the capital of France?"
    assert entry["wrong_answer"] == "Lyon"
    assert entry["correct_answer"] == "The capital of France is Paris."
    assert "timestamp" in entry


def test_record_multiple_corrections(store: FeedbackStore):
    store.record("Q1", "bad1", "good1")
    store.record("Q2", "bad2", "good2")

    assert store.correction_count() == 2
    lines = store.corrections_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_list_corrections_returns_all(store: FeedbackStore):
    store.record("Who invented the telephone?", "Edison", "Alexander Graham Bell invented the telephone.")

    entries = store.list_corrections()
    assert len(entries) == 1
    assert entries[0]["question"] == "Who invented the telephone?"


def test_list_corrections_empty_when_no_log(store: FeedbackStore):
    assert store.list_corrections() == []
    assert store.correction_count() == 0


def test_record_raises_on_empty_question(store: FeedbackStore):
    with pytest.raises(ValueError):
        store.record("", "wrong", "correct")


def test_record_raises_on_empty_correct_answer(store: FeedbackStore):
    with pytest.raises(ValueError):
        store.record("Some question?", "wrong answer", "")


def test_corrections_txt_is_used_by_retriever_build_index(tmp_path: Path):
    """Verify that corrections.txt is merged into docs when building the index."""
    from types import SimpleNamespace
    import sys
    import numpy as np
    from rag.retriever import Retriever

    # --- setup fake environment ---
    docs_file = tmp_path / "docs.txt"
    docs_file.write_text("RAG stands for Retrieval Augmented Generation.\n", encoding="utf-8")

    corrections_file = tmp_path / "corrections.txt"
    corrections_file.write_text(
        "Q: What is the capital of France? A: The capital of France is Paris.\n",
        encoding="utf-8",
    )

    index_file = tmp_path / "faiss.index"
    meta_file = tmp_path / "meta.json"

    # Fake faiss so no real GPU/index needed
    captured = {}

    class _FakeIndex:
        def add(self, embeddings):
            captured["n_embeddings"] = len(embeddings)

    class _FakeFaiss:
        @staticmethod
        def IndexFlatL2(dim):
            return _FakeIndex()

        @staticmethod
        def write_index(index, path):
            Path(path).write_text("fake_index", encoding="utf-8")

    retriever = object.__new__(Retriever)
    retriever.config = SimpleNamespace(
        docs_file=docs_file,
        corrections_file=corrections_file,
        index_file=index_file,
        meta_file=meta_file,
        chunk_size_chars=700,
        chunk_overlap_chars=120,
        context_compression_enabled=False,
        compression_max_chars=420,
        compression_max_sentences=2,
    )
    retriever.encoder = SimpleNamespace(
        encode=lambda texts, **kwargs: np.ones((len(texts), 4), dtype=float)
    )

    import json as _json

    def fake_json_dump(data, fh, **kwargs):
        fh.write(_json.dumps(data))

    import rag.retriever as retriever_module
    original_modules = sys.modules.copy()
    sys.modules["faiss"] = _FakeFaiss

    try:
        n_chunks = retriever.build_index()
    finally:
        sys.modules.update(original_modules)

    # Both the original doc and the correction should be indexed
    assert n_chunks >= 2, f"Expected ≥2 chunks (original + correction), got {n_chunks}"
