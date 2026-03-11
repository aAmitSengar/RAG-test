"""Feedback store for the RAG pipeline teaching/learning loop.

When a user marks an answer as wrong and provides the correct answer:
1. The correction is appended to ``data/corrections.txt`` as a Q→A document.
2. A structured log entry is written to ``data/feedback.jsonl``.
3. The caller is expected to rebuild the FAISS index so the correction is
   retrievable for future queries.

This is the standard RAG "human-in-the-loop" approach: rather than fine-tuning
the generation model (which requires GPU time and large datasets), we teach the
*retrieval* layer by inserting authoritative Q→A pairs directly into the
knowledge base.  The next time a semantically similar question is asked, the
corrected document surfaces in top-k chunks, and the generator produces a
grounded, accurate answer.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class FeedbackStore:
    """Persist answer corrections and expose them to the retrieval pipeline."""

    def __init__(self, data_dir: Path) -> None:
        data_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_file: Path = data_dir / "corrections.txt"
        self.feedback_log: Path = data_dir / "feedback.jsonl"

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def record(self, question: str, wrong_answer: str, correct_answer: str) -> None:
        """Persist a user correction.

        Appends to both the structured JSONL audit log and the plain-text
        corrections document that gets indexed by the retriever.
        """
        question = question.strip()
        correct_answer = correct_answer.strip()
        if not question or not correct_answer:
            raise ValueError("question and correct_answer must not be empty")

        # 1. Write to the JSONL audit log.
        entry: Dict[str, str] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "wrong_answer": wrong_answer.strip(),
            "correct_answer": correct_answer,
        }
        with open(self.feedback_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # 2. Append the Q→A pair to corrections.txt as a retrievable document.
        #    Each line becomes one "document" for the retriever's chunker.
        doc_line = f"Q: {question} A: {correct_answer}"
        with open(self.corrections_file, "a", encoding="utf-8") as fh:
            fh.write(doc_line + "\n")

        logger.info(
            "[FeedbackStore] correction recorded | question='%s'", question[:80]
        )

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def list_corrections(self) -> List[Dict[str, str]]:
        """Return all logged corrections, newest last."""
        if not self.feedback_log.exists():
            return []
        entries: List[Dict[str, str]] = []
        with open(self.feedback_log, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("[FeedbackStore] skipping malformed log line")
        return entries

    def correction_count(self) -> int:
        """Return the total number of recorded corrections."""
        return len(self.list_corrections())
