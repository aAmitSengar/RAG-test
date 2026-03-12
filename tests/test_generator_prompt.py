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


def test_summary_question_returns_multiple_sentences():
    """Summary/explain questions should produce multi-sentence rich answers."""
    generator = _make_generator()
    generator.generate = lambda question, context: "question:"  # force fallback
    context = [
        {
            "chunk_id": 70,
            "score": 0.95,
            "text": (
                "Mohandas Karamchand Gandhi emerged as the foremost leader of the Indian freedom struggle. "
                "He championed non-violent resistance (Satyagraha) to oppose British colonial rule. "
                "His Salt March in 1930 became a pivotal moment in the independence movement."
            ),
        },
        {
            "chunk_id": 110,
            "score": 0.88,
            "text": (
                "The Indian National Congress, founded in 1885, united freedom fighters across the country. "
                "The Quit India Movement of 1942 demanded an end to British rule."
            ),
        },
        {
            "chunk_id": 72,
            "score": 0.82,
            "text": (
                "India achieved independence on 15 August 1947 after decades of struggle. "
                "Jawaharlal Nehru became the first Prime Minister of independent India."
            ),
        },
    ]

    output = generator.generate_with_fallback("summarize freedom?", context)

    # Should contain multiple sentences (not just one line)
    sentences = [s for s in output["answer"].split(".") if s.strip()]
    assert len(sentences) >= 3, f"Expected at least 3 sentences, got: {output['answer']}"
    # Should cite sources
    assert "Sources:" in output["answer"]
    assert "[70]" in output["answer"]


def test_timeline_question_returns_year_wise_chronological_points():
    generator = _make_generator()
    generator.generate = lambda question, context: (
        "In order to provide deeper historical density, we revisit epochs in detail."
    )
    context = [
        {
            "chunk_id": 96,
            "score": 0.93,
            "text": (
                "In 1857, a large rebellion challenged East India Company rule. "
                "In 1885, the Indian National Congress was formed. "
                "In 1947, India gained independence."
            ),
        },
        {
            "chunk_id": 88,
            "score": 0.81,
            "text": "In 1950, the Constitution came into effect and India became a republic.",
        },
    ]

    output = generator.generate_with_fallback(
        "In Indian history explain year wise change, please explain", context
    )

    assert output["answer"].startswith("Year-wise timeline:")
    assert "1857:" in output["answer"]
    assert "1947:" in output["answer"]
    assert "1950:" in output["answer"]
    assert output["answer"].find("1857:") < output["answer"].find("1947:")
