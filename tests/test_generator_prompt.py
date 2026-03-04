"""Tests for generator prompt and fallback behavior."""

from types import SimpleNamespace

from rag.generator import Generator


def _make_generator() -> Generator:
    generator = object.__new__(Generator)
    generator.config = SimpleNamespace(do_sample=False)
    generator.tokenizer = None
    generator.model = None
    return generator


def test_prompt_contains_grounding_instructions_and_citations():
    generator = _make_generator()
    context = [
        {"chunk_id": 4, "score": 0.91, "text": "RAG combines retrieval and generation."}
    ]

    prompt = generator._build_prompt("What is RAG?", context)

    assert "Answer using only the provided context" in prompt
    assert "If the context is insufficient" in prompt
    assert "[Chunk 4]" in prompt


def test_empty_context_returns_insufficient_response():
    generator = _make_generator()

    output = generator.generate_with_fallback("Unknown question", [])

    assert "Insufficient context" in output["answer"]
    assert output["citations"] == []


def test_low_quality_model_output_switches_to_fallback():
    generator = _make_generator()
    generator.generate = lambda question, context: "Answer using only the provided context"
    context = [{"chunk_id": 2, "score": 0.8, "text": "India became independent on 15 August 1947."}]

    output = generator.generate_with_fallback("What happened in 1947?", context)

    assert "1947" in output["answer"]
    assert "Sources:" in output["answer"]
    assert "[2]" in output["answer"]
