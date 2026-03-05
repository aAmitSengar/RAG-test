"""Evaluate RAG quality on a JSONL dataset."""

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Set

from main import RAGPipeline


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]{2,}\b", (text or "").lower())


def _exact_match(prediction: str, reference: str) -> float:
    return float((prediction or "").strip().lower() == (reference or "").strip().lower())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1

    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))

    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return (2 * precision * recall) / (precision + recall)


def _retrieval_recall(retrieved: Sequence[int], expected: Sequence[int]) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hit = len(expected_set.intersection(set(retrieved)))
    return hit / len(expected_set)


def _citation_precision(citations: Sequence[int], expected: Sequence[int]) -> float:
    citation_set: Set[int] = set(citations)
    if not citation_set:
        return 0.0
    expected_set = set(expected)
    hit = len(citation_set.intersection(expected_set))
    return hit / len(citation_set)


def _load_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate(dataset_path: Path, k: int) -> None:
    rag = RAGPipeline()
    rows = _load_jsonl(dataset_path)
    if not rows:
        raise ValueError(f"No evaluation examples in {dataset_path}")

    em_scores: List[float] = []
    f1_scores: List[float] = []
    recall_scores: List[float] = []
    citation_scores: List[float] = []

    for row in rows:
        question = row["question"]
        expected_answer = row.get("expected_answer", "")
        expected_chunks = row.get("expected_chunk_ids", [])

        result = rag.run(question, k=k)
        answer = str(result["answer"])
        retrieved_ids = [int(chunk["chunk_id"]) for chunk in result["retrieved_chunks"]]
        citations = [int(cid) for cid in result.get("citations", [])]

        em_scores.append(_exact_match(answer, expected_answer))
        f1_scores.append(_token_f1(answer, expected_answer))
        if expected_chunks:
            recall_scores.append(_retrieval_recall(retrieved_ids, expected_chunks))
            citation_scores.append(_citation_precision(citations, expected_chunks))

    print("\nRAG Evaluation Summary")
    print("=" * 48)
    print(f"Dataset: {dataset_path}")
    print(f"Examples: {len(rows)}")
    print(f"Exact Match: {mean(em_scores):.3f}")
    print(f"Token F1: {mean(f1_scores):.3f}")
    if recall_scores:
        print(f"Retrieval Recall@{k}: {mean(recall_scores):.3f}")
    if citation_scores:
        print(f"Citation Precision: {mean(citation_scores):.3f}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on JSONL dataset.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to JSONL file with fields: question, expected_answer, expected_chunk_ids",
    )
    parser.add_argument("--k", type=int, default=3, help="Retriever top-k for evaluation.")
    args = parser.parse_args()

    evaluate(Path(args.dataset), args.k)


if __name__ == "__main__":
    main()
