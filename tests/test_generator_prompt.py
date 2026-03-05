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


def test_yes_no_question_uses_interactive_fallback_style():
    generator = _make_generator()
    generator.generate = lambda question, context: "question:"
    context = [
        {
            "chunk_id": 21,
            "score": 0.9,
            "text": (
                "Trade between India and the Roman Empire intensified in the first two centuries CE, "
                "with exports of spices and textiles and inflows of Roman gold coins."
            ),
        }
    ]

    output = generator.generate_with_fallback("Was India rich?", context)

    assert output["answer"].startswith("Yes, based on the retrieved context.")
    assert "[21]" in output["answer"]


def test_when_question_is_refined_to_direct_date_answer():
    generator = _make_generator()
    generator.generate = lambda question, context: (
        "Mohandas Karamchand Gandhi biography... India became independent on 15 August 1947 "
        "after the Indian Independence Act. [Sources: [74], [63]]"
    )
    context = [
        {
            "chunk_id": 74,
            "score": 0.9,
            "text": "On 15 August 1947 India became independent.",
        },
        {
            "chunk_id": 63,
            "score": 0.8,
            "text": "The Indian Independence Act was passed in 1947.",
        },
    ]

    output = generator.generate_with_fallback("when india got freedom?", context)

    assert output["answer"].startswith("India got freedom on 15 August 1947.")
    assert "[74]" in output["answer"]
