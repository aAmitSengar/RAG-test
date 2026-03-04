"""Integration-style tests for pipeline output contract."""

from types import SimpleNamespace

from main import RAGPipeline


class _FakeRetriever:
    def retrieve(self, query, k=3):
        return [
            {
                "chunk_id": 10,
                "source_doc_id": 1,
                "text": "RAG retrieves relevant chunks first.",
                "score": 0.88,
                "distance": 0.2,
            },
            {
                "chunk_id": 12,
                "source_doc_id": 2,
                "text": "Generation uses the retrieved context.",
                "score": 0.76,
                "distance": 0.4,
            },
        ]


class _FakeGenerator:
    def generate_with_fallback(self, question, context):
        return {
            "answer": "RAG retrieves then generates [10] [12].",
            "citations": [10, 12],
        }


class _EmptyRetriever:
    def retrieve(self, query, k=3):
        return []


def _make_pipeline(citations_enabled=True):
    pipeline = object.__new__(RAGPipeline)
    pipeline.config = SimpleNamespace(citations_enabled=citations_enabled)
    return pipeline


def test_pipeline_returns_answer_chunks_and_citations():
    pipeline = _make_pipeline(citations_enabled=True)
    pipeline.retriever = _FakeRetriever()
    pipeline.generator = _FakeGenerator()

    result = pipeline.run("Explain RAG", k=2)

    assert result["query"] == "Explain RAG"
    assert len(result["retrieved_chunks"]) == 2
    assert result["citations"] == [10, 12]
    valid_chunk_ids = {chunk["chunk_id"] for chunk in result["retrieved_chunks"]}
    assert set(result["citations"]).issubset(valid_chunk_ids)


def test_pipeline_returns_insufficient_context_when_no_chunks():
    pipeline = _make_pipeline(citations_enabled=True)
    pipeline.retriever = _EmptyRetriever()
    pipeline.generator = _FakeGenerator()

    result = pipeline.run("No answer", k=2)

    assert result["retrieved_chunks"] == []
    assert result["citations"] == []
    assert "Insufficient context" in result["answer"]
