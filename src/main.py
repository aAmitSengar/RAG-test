"""
RAG (Retrieval-Augmented Generation) Teaching Example
=====================================================

This module demonstrates a complete RAG pipeline:
1. RETRIEVER: Search relevant documents using embeddings (FAISS + SentenceTransformer)
2. GENERATOR: Generate answers using the retrieved context (Seq2Seq model like T5)

The pipeline is broken down into:
- Config: Centralized configuration management
- Retriever: FAISS-based document retrieval
- Generator: Answer generation from context

This is a teaching example designed to show the core RAG workflow in a clear,
modular way. For production, consider using frameworks like LangChain or LlamaIndex.
"""

import logging
import sys
import argparse
from pathlib import Path
from typing import Any, Dict

# Add the 'src' directory to Python path to enable direct imports of 'rag' module.
# This setup is common in smaller projects and detailed further in README.md.
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.generator import Generator
from rag.retriever import Retriever
from rag.utils import setup_logging


logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""

    def __init__(self):
        """Initialize the RAG pipeline with configuration, retriever, and generator."""
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)

        try:
            self.config = Config()
            logger.info("Configuration loaded")
            logger.debug("%s", self.config)
            self.retriever = Retriever(self.config)
            self.generator = Generator(self.config)
            logger.info("Pipeline initialized")
        except Exception as exc:
            logger.error("Failed to initialize pipeline: %s", exc)
            raise

    def _explain_step(self, title: str, detail: str) -> None:
        """Print a short explanation and optionally pause in guided mode."""
        logger.info("%s: %s", title, detail)
        if getattr(self.config, "step_by_step_mode", False):
            input("Press Enter to continue...")

    def run(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Run retrieval + generation and return structured output."""
        logger.info("=" * 60)
        logger.info("Running RAG Pipeline")
        logger.info("=" * 60)
        logger.info("Query: %s", query)

        try:
            self._explain_step(
                "Step 1/2 Retrieval",
                "Encode the question and fetch the highest-relevance chunks.",
            )
            retrieved_chunks = self.retriever.retrieve(query, k=k)

            logger.info("Retrieved %d chunk(s)", len(retrieved_chunks))
            for i, chunk in enumerate(retrieved_chunks, 1):
                preview = str(chunk["text"])
                preview = preview[:100] + "..." if len(preview) > 100 else preview
                logger.info(
                    "  [%d] chunk=%s source=%s score=%.3f %s",
                    i,
                    chunk["chunk_id"],
                    chunk["source_doc_id"],
                    float(chunk["score"]),
                    preview,
                )

            if not retrieved_chunks:
                answer = "Insufficient context to answer confidently."
                return {
                    "query": query,
                    "retrieved_chunks": [],
                    "answer": answer,
                    "citations": [],
                }

            self._explain_step(
                "Step 2/2 Generation",
                "Build a grounded prompt from retrieved chunks and generate an answer.",
            )
            generated = self.generator.generate_with_fallback(query, retrieved_chunks)

            answer = str(generated["answer"])
            citations = list(generated["citations"])
            if self.config.citations_enabled and not citations:
                citations = [int(chunk["chunk_id"]) for chunk in retrieved_chunks[:2]]

            return {
                "query": query,
                "retrieved_chunks": retrieved_chunks,
                "answer": answer,
                "citations": citations,
            }

        except Exception as exc:
            logger.error("Pipeline execution failed: %s", exc)
            raise

    def build_index(self) -> None:
        """Build FAISS index from documents. Call this first if index doesn't exist."""
        self._explain_step(
            "Index Build",
            "Split documents into chunks, embed them, and save FAISS index + metadata.",
        )
        num_chunks = self.retriever.build_index()
        logger.info("Index built with %d chunk(s)", num_chunks)


def _validate_setup(rag: "RAGPipeline") -> bool:
    """Validate that all required files and docs are available."""
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
    logger.info("\nQuestion: %s", result["query"])
    logger.info("\nRetrieved %d chunk(s):", len(result["retrieved_chunks"]))
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        preview = str(chunk["text"])
        preview = preview[:80] + "..." if len(preview) > 80 else preview
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
        rag = RAGPipeline()

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
