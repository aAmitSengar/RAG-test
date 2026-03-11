"""
RAG (Retrieval-Augmented Generation) Teaching Example
=====================================================

This module demonstrates a complete RAG pipeline with light conversational support:

1) RETRIEVER
   - Searches relevant documents using embeddings (FAISS + SentenceTransformer)
   - (Hybrid lexical + semantic if enabled in .env)

2) GENERATOR
   - Generates answers using the retrieved context (Seq2Seq model like T5)
   - Produces concise, grounded answers with optional citations

3) CONVERSATION-AWARE FOLLOW-UPS
   - Detects vague follow-ups like "tell me more", "continue", "aur batao"
   - Rewrites them using the previous topic so retrieval stays on track

4) CHAT HISTORY (SQLite)
   - Stores one chat per program run (user and assistant messages)

Teaching note:
This is a simple, readable teaching example showing core RAG workflow + follow-ups.
For production, consider LangChain/LlamaIndex, durable memory, auth, evals, and guardrails.
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root (so `rag` package is importable when running this file directly)
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.generator import Generator
from rag.retriever import Retriever
from rag.utils import setup_logging
from rag.chat_store import ChatStore  # <-- ensure rag/chat_store.py exists

logger = logging.getLogger(__name__)


# -----------------------------
# Conversation helpers
# -----------------------------

class ConversationState:
    """
    Minimal in-memory conversation state so follow-up queries become topic-aware.
      - last_question: previous user query
      - last_answer_summary: 1–2 sentence semantic summary of the last answer
    """
    def __init__(self):
        self.last_question: Optional[str] = None
        self.last_answer_summary: Optional[str] = None

    def reset(self):
        self.last_question = None
        self.last_answer_summary = None


class FollowupHeuristics:
    """
    Simple rules to detect vague follow-up queries and build a topical prompt.
    Extend with LLM rewriting later if needed.
    """
    FOLLOWUP_PATTERNS = {
        "tell me more",
        "continue",
        "go on",
        "more",
        "and then what",
        "what happened after that",
        "phir kya hua",
        "aur batao",
        "aur bhi batao",
        "aur",
        "aur kya",
    }

    @staticmethod
    def is_followup(query: str) -> bool:
        q = (query or "").strip().lower()
        if not q:
            return False
        if q in FollowupHeuristics.FOLLOWUP_PATTERNS:
            return True
        # Loose suffix checks (e.g., "… more", "… aur")
        return q.endswith(" more") or q.endswith(" aur")

    @staticmethod
    def build_followup_query(
        original_question: Optional[str],
        last_answer_summary: Optional[str]
    ) -> Optional[str]:
        """
        Convert a vague follow-up into a concrete, on-topic retrieval prompt.
        Priority:
          1) last_answer_summary (most precise)
          2) original_question  (better than a vague 'tell me more')
        """
        if last_answer_summary:
            return f"Give more historical details, evidence, and key dates about: {last_answer_summary}"
        if original_question:
            return f"Give more historical details, evidence, and key dates about the previous topic: {original_question}"
        return None


# -----------------------------
# Teaching Pipeline
# -----------------------------

class RAGPipeline:
    """Complete RAG pipeline combining retrieval, generation, conversation memory, and chat storage."""

    def __init__(self, debug: bool = False):
        """Initialize config, retriever, generator, chat store, and conversation memory."""
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)
        self.debug = debug
        try:
            # 1) Config
            self.config = Config()
            logger.info("Configuration loaded")
            logger.debug("%s", self.config)

            # 2) Core components
            self.retriever = Retriever(self.config)
            self.generator = Generator(self.config)

            # 3) Persistent chat history (SQLite) — one chat id per run
            self.chat_store = ChatStore()  # creates data/chat_history.db if needed
            self.current_chat_id = f"chat_{uuid.uuid4().hex[:8]}"
            self.chat_store.create_chat(self.current_chat_id, title="Teaching Demo Chat")

            # 4) Lightweight conversation memory
            self.state = ConversationState()

            logger.info("Pipeline initialized (chat_id=%s)", self.current_chat_id)
        except Exception as exc:
            logger.error("Failed to initialize pipeline: %s", exc)
            raise

    def _explain_step(self, title: str, detail: str) -> None:
        """Print an explanation and optionally pause when running in debug mode."""
        logger.info("%s: %s", title, detail)
        if self.debug:
            input("Press Enter to continue...")

    def _summarize_for_memory(self, answer_text: str, fallback: Optional[str] = None) -> str:
        """
        Produce a clean semantic summary so follow-ups stay on-topic.
        HARD-FILTER persona/language/instruction echoes.
        """
        import re
        if not answer_text:
            return fallback or ""

        sentences = re.split(r"(?<=[.!?])\s+", answer_text.strip())
        clean = []
        for s in sentences:
            t = s.strip()
            low = t.lower()
            if not t:
                continue
            if low.startswith("answer in "):
                continue
            if low.startswith("respond in "):
                continue
            if "do not repeat this instruction" in low:
                continue
            if low.startswith("answer:"):
                continue
            if low.startswith("you are a ") or low.startswith("you are an "):
                continue
            clean.append(t)

        if not clean:
            return fallback or ""

        first = clean[0]
        if len(first) > 160:
            first = first[:157].rstrip() + "..."
        return first


    def _make_effective_query(self, user_query: str) -> str:
        """
        Build the retrieval query to use:
        - If user_query is a vague follow-up, rewrite it using conversation state.
        - Otherwise, return user_query directly.
        """
        if FollowupHeuristics.is_followup(user_query):
            logger.info("[Pipeline] Follow-up detected for query: %s", user_query)
            rewritten = FollowupHeuristics.build_followup_query(
                self.state.last_question,
                self.state.last_answer_summary
            )
            if rewritten:
                return rewritten
        return user_query

    def run(self, query: str, k: Optional[int] = None, guided: bool = False) -> Dict[str, Any]:
        """
        Run retrieval + generation and return structured output.

        Args:
            query: The user question.
            k: Number of chunks to retrieve (defaults to config value).
            guided: When True, include step-by-step pipeline explanations in the response.

        Returns:
            {
                "query": <original_user_query>,
                "effective_query": <rewritten_query_used_for_retrieval>,
                "retrieved_chunks": <list[dict]>,
                "answer": <string>,
                "citations": <list[int]>,
                "chat_id": <str>,
                "pipeline_steps": <list[dict]>  # populated only when guided=True
            }
        """
        logger.info("=" * 60)
        logger.info("Running RAG Pipeline")
        logger.info("=" * 60)
        logger.info("User Query: %s", query)

        pipeline_steps: list = []

        try:
            # Build an effective (topic-anchored) retrieval query for vague follow-ups
            effective_query = self._make_effective_query(query)
            if effective_query != query:
                logger.info("Effective Retrieval Query: %s", effective_query)

            # --- Save user message (original text) ---
            self.chat_store.add_message(self.current_chat_id, role="user", content=query)

            # Step 1: Retrieval
            self._explain_step(
                "Step 1/2 Retrieval",
                "Encode the question and fetch the highest-relevance chunks."
            )
            if guided:
                pipeline_steps.append({
                    "step": "Step 1/2: Retrieval",
                    "description": "Encoding the question and fetching the highest-relevance chunks from the FAISS index.",
                })
            top_k = int(k if k is not None else self.config.retrieval_k)
            retrieved_chunks = self.retriever.retrieve(effective_query, k=top_k)

            logger.info("Retrieved %d chunk(s)", len(retrieved_chunks))
            for i, chunk in enumerate(retrieved_chunks, 1):
                preview = str(chunk["text"])
                preview = (preview[:100] + "...") if len(preview) > 100 else preview
                logger.info(
                    "  [%d] chunk=%s source=%s score=%.3f %s",
                    i,
                    chunk["chunk_id"],
                    chunk["source_doc_id"],
                    float(chunk["score"]),
                    preview,
                )

            if guided:
                pipeline_steps.append({
                    "step": "Retrieval Result",
                    "description": f"Retrieved {len(retrieved_chunks)} relevant chunk(s).",
                    "chunks": retrieved_chunks,
                })

            if not retrieved_chunks:
                answer = "Insufficient context to answer confidently."

                # --- Save assistant message (even if nothing retrieved) ---
                self.chat_store.add_message(self.current_chat_id, role="assistant", content=answer)

                # Update conversation memory for next turn
                self.state.last_question = query
                self.state.last_answer_summary = self._summarize_for_memory(answer, fallback=query)

                return {
                    "query": query,
                    "effective_query": effective_query,
                    "retrieved_chunks": [],
                    "answer": answer,
                    "citations": [],
                    "chat_id": self.current_chat_id,
                    "pipeline_steps": pipeline_steps,
                }

            # Step 2: Generation
            self._explain_step(
                "Step 2/2 Generation",
                "Build a grounded prompt from retrieved chunks and generate an answer."
            )
            if guided:
                pipeline_steps.append({
                    "step": "Step 2/2: Generation",
                    "description": "Building a grounded prompt from retrieved chunks and generating an answer.",
                })
            generated = self.generator.generate_with_fallback(query, retrieved_chunks)
            answer = str(generated.get("answer", "")).strip()
            citations = list(generated.get("citations", []))

            # If citations are enabled but absent, fall back to first few chunk ids
            if self.config.citations_enabled and not citations:
                citations = [int(chunk["chunk_id"]) for chunk in retrieved_chunks[:2]]

            # --- Save assistant message ---
            self.chat_store.add_message(self.current_chat_id, role="assistant", content=answer)

            # Update conversation memory for the NEXT turn
            self.state.last_question = query
            self.state.last_answer_summary = self._summarize_for_memory(answer, fallback=query)

            return {
                "query": query,
                "effective_query": effective_query,
                "retrieved_chunks": retrieved_chunks,
                "answer": answer,
                "citations": citations,
                "chat_id": self.current_chat_id,
                "pipeline_steps": pipeline_steps,
            }

        except Exception as exc:
            logger.error("Pipeline execution failed: %s", exc)
            raise

    def build_index(self) -> None:
        """Build FAISS index from documents. Call this first if index doesn't exist."""
        self._explain_step(
            "Index Build",
            "Split documents into chunks, embed them, and save FAISS index + metadata."
        )
        num_chunks = self.retriever.build_index()
        logger.info("Index built with %d chunk(s)", num_chunks)


# -----------------------------
# Utility functions
# -----------------------------

def _validate_setup(rag: "RAGPipeline") -> bool:
    """Validate that docs exist and are non-empty."""
    if not rag.config.docs_file.exists():
        logger.error("\n❌ ERROR: Documentation file not found at %s", rag.config.docs_file)
        logger.error("\nPlease create a 'docs.txt' file in the data/ folder with one document per line.")
        return False

    with open(rag.config.docs_file, "r", encoding="utf-8") as fh:
        docs = [line.strip() for line in fh if line.strip()]

    if not docs:
        logger.error("❌ ERROR: docs.txt is empty. Please add documents to search from.")
        return False

    logger.info("✓ Found %d document line(s) in %s", len(docs), rag.config.docs_file)
    return True


def _build_index_if_needed(rag: "RAGPipeline") -> bool:
    """Build FAISS index if it doesn't exist or metadata is missing."""
    if rag.config.index_file.exists() and rag.config.meta_file.exists():
        logger.info(
            "✓ FAISS index already exists at %s and metadata at %s\n",
            rag.config.index_file,
            rag.config.meta_file,
        )
        return True

    if rag.config.index_file.exists() and not rag.config.meta_file.exists():
        logger.warning(
            "Index exists but metadata file is missing (%s). Rebuilding index.",
            rag.config.meta_file,
        )

    logger.info("Building FAISS index from documents...")
    try:
        rag.build_index()
        return True
    except Exception as exc:
        logger.error("❌ Failed to build index: %s", exc)
        return False


def _run_single_query(rag: "RAGPipeline", query: str) -> dict:
    """Run a single RAG query and return results."""
    return rag.run(query, k=rag.config.retrieval_k)


def _print_result(result: dict, debug: bool = False) -> None:
    """Pretty print RAG result."""
    if not debug:
        print(f"\nAnswer: {result['answer']}\n")
        return

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULT")
    logger.info("=" * 70)
    logger.info("\nOriginal Question: %s", result["query"])
    if result.get("effective_query") and result["effective_query"] != result["query"]:
        logger.info("Effective Retrieval Query: %s", result["effective_query"])

    logger.info("\nRetrieved %d chunk(s):", len(result["retrieved_chunks"]))
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        preview = str(chunk["text"])
        preview = (preview[:80] + "...") if len(preview) > 80 else preview
        logger.info(
            "  [%d] chunk=%s source=%s score=%.3f %s",
            i,
            chunk["chunk_id"],
            chunk["source_doc_id"],
            float(chunk["score"]),
            preview,
        )

    if result.get("citations"):
        citation_text = ", ".join(f"[{cid}]" for cid in result["citations"])
        logger.info("\nCitations: %s", citation_text)

    logger.info("\nGenerated Answer:\n%s", result["answer"])
    logger.info("\n" + "=" * 70 + "\n")


# -----------------------------
# CLI entry point
# -----------------------------

def main():
    """Main RAG teaching example with optional interactive mode."""
    parser = argparse.ArgumentParser(description="Run the RAG teaching pipeline.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logs for retrieval/generation internals.",
    )
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.ERROR)

    try:
        rag = RAGPipeline(debug=args.debug)

        if not _validate_setup(rag):
            logger.info("\n⚠️  Setup incomplete. Please fix the issues above and try again.")
            sys.exit(1)

        if not _build_index_if_needed(rag):
            sys.exit(1)

        if args.debug:
            logger.info("\nInteractive mode")
            logger.info("Ask questions. Type 'quit' to exit.\n")

        while True:
            try:
                user_query = input("Ask a question (or 'quit' to exit): ").strip()

                if user_query.lower() in ("quit", "exit", "q"):
                    logger.info("\n✓ Goodbye!")
                    break

                if not user_query:
                    logger.warning("Please enter a question.")
                    continue

                result = _run_single_query(rag, user_query)
                _print_result(result, debug=args.debug)

            except KeyboardInterrupt:
                logger.info("\n\n✓ Interrupted. Goodbye!")
                break
            except Exception as exc:
                logger.error("❌ Error: %s", exc)
                continue

    except FileNotFoundError as exc:
        logger.error("❌ File not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("❌ Pipeline error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()