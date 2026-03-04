"""Tests for text chunking behavior in retriever."""

from types import SimpleNamespace

from rag.retriever import Retriever


def _make_retriever(chunk_size: int, overlap: int) -> Retriever:
    retriever = object.__new__(Retriever)
    retriever.config = SimpleNamespace(
        chunk_size_chars=chunk_size,
        chunk_overlap_chars=overlap,
    )
    return retriever


def test_short_text_remains_single_chunk():
    retriever = _make_retriever(chunk_size=700, overlap=120)
    text = "RAG combines retrieval and generation."

    chunks = retriever._chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_long_text_splits_with_overlap():
    retriever = _make_retriever(chunk_size=80, overlap=20)
    text = (
        "Sentence one gives context. Sentence two adds details. "
        "Sentence three extends discussion. Sentence four closes."
    )

    chunks = retriever._chunk_text(text)

    assert len(chunks) >= 2
    assert all(len(chunk["text"]) <= 80 for chunk in chunks)
    # Overlap should keep tail chars from first chunk in the next chunk.
    first_tail = chunks[0]["text"][-20:].strip()
    assert first_tail
    assert first_tail in chunks[1]["text"]
