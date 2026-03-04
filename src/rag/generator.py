"""Generator module for answer generation."""

import logging
import re
from typing import Dict, List, Sequence

from .config import Config

logger = logging.getLogger(__name__)


class Generator:
    """Sequence-to-sequence based answer generator."""

    def __init__(self, config: Config):
        """Initialize the generator."""
        self.config = config
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the generation model and tokenizer."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_source = self.config.gen_model
            local_only = self.config.gen_model_is_local

            logger.info("[Generator] loading generation model: %s", model_source)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    local_files_only=local_only,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_source,
                    local_files_only=local_only,
                )
                logger.info("[Generator] generation model ready")
                return
            except Exception as local_error:
                if not local_only:
                    raise

                fallback_model = self._fallback_model_name(model_source)
                logger.warning(
                    "Failed to load local generation model '%s': %s. "
                    "Falling back to remote model '%s'.",
                    model_source,
                    local_error,
                    fallback_model,
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    fallback_model,
                    local_files_only=False,
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    fallback_model,
                    local_files_only=False,
                )
                logger.info("[Generator] fallback generation model ready")
        except ImportError as exc:
            raise ImportError(
                "transformers is required. Install with: "
                "pip install transformers"
            ) from exc
        except Exception as exc:
            logger.error(
                "Error loading generation model: %s. "
                "Continuing in template-only mode.",
                exc,
            )
            self.tokenizer = None
            self.model = None

    @staticmethod
    def _fallback_model_name(model_source: str) -> str:
        """Resolve a reasonable Hugging Face model id for fallback loading."""
        return model_source.rstrip("/").split("/")[-1] or "t5-small"

    @staticmethod
    def _extract_citations(answer: str) -> List[int]:
        """Extract citation ids from model output format like [3], [12]."""
        hits = re.findall(r"\[(\d+)\]", answer or "")
        return sorted({int(hit) for hit in hits})

    @staticmethod
    def _is_low_quality_answer(answer: str) -> bool:
        """Detect common low-quality outputs from small seq2seq models."""
        cleaned = (answer or "").strip().lower()
        if not cleaned:
            return True

        bad_prefixes = (
            "answer using only the provided context",
            "you are a grounded qa assistant",
            "question:",
            "context:",
            "answer:",
        )
        if any(cleaned.startswith(prefix) for prefix in bad_prefixes):
            return True

        # Extremely short outputs are usually non-answers.
        if len(cleaned.split()) < 4:
            return True

        return False

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentence-like units."""
        parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _question_terms(question: str) -> List[str]:
        """Extract lightweight lexical terms from question for sentence matching."""
        terms = re.findall(r"\b[a-zA-Z]{4,}\b|\b\d{4}\b", (question or "").lower())
        stop = {"what", "when", "where", "which", "happened", "happend", "about"}
        return [t for t in terms if t not in stop]

    def _build_prompt(self, question: str, context: Sequence[Dict[str, int | float | str]]) -> str:
        """Build an instruction-oriented prompt with source ids."""
        context_lines = []
        for chunk in context:
            context_lines.append(
                f"[Chunk {chunk['chunk_id']}] (score={float(chunk['score']):.3f}) {chunk['text']}"
            )
        context_text = "\n".join(context_lines)

        return (
            "You are a grounded QA assistant.\n"
            "Answer using only the provided context.\n"
            "If the context is insufficient, answer exactly: Insufficient context to answer confidently.\n"
            "When you use evidence, cite chunk ids in square brackets, e.g. [12].\n\n"
            f"Question: {question}\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            "Answer:"
        )

    def generate(self, question: str, context: List[Dict[str, int | float | str]]) -> str:
        """Generate an answer based on question and structured context."""
        if not context:
            return "Insufficient context to answer confidently."

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Generation model is unavailable")

        try:
            prompt = self._build_prompt(question, context)
            logger.info("[Generator] generating answer from %d context chunk(s)", len(context))

            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=180,
                num_beams=5,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                do_sample=self.config.do_sample,
                early_stopping=True,
            )

            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            logger.info("[Generator] answer generated")
            return answer or "Insufficient context to answer confidently."

        except Exception as exc:
            logger.error("Error during generation: %s", exc)
            raise

    def generate_with_fallback(
        self, question: str, context: List[Dict[str, int | float | str]]
    ) -> Dict[str, str | List[int]]:
        """Generate answer with fallback to template if model fails."""
        try:
            answer = self.generate(question, context)
            if self._is_low_quality_answer(answer):
                logger.warning(
                    "[Generator] Low-quality model output detected, switching to extractive fallback."
                )
                answer = self._template_answer(question, context)
        except Exception as exc:
            logger.warning("[Generator] Generation failed, using template fallback: %s", exc)
            answer = self._template_answer(question, context)

        citations = self._extract_citations(answer)
        return {"answer": answer, "citations": citations}

    @staticmethod
    def _template_answer(question: str, context: List[Dict[str, int | float | str]]) -> str:
        """Generate a grounded extractive answer when model output is weak."""
        if not context:
            return "Insufficient context to answer confidently."

        top_chunks = context[:3]
        chunks = [int(item["chunk_id"]) for item in top_chunks]
        terms = Generator._question_terms(question)

        candidates: List[str] = []
        for item in top_chunks:
            for sent in Generator._split_sentences(str(item["text"])):
                candidates.append(sent)

        def score(sentence: str) -> int:
            s = sentence.lower()
            hit_count = sum(1 for term in terms if term in s)
            year_bonus = 2 if re.search(r"\b\d{4}\b", s) else 0
            return hit_count + year_bonus

        ranked = sorted(candidates, key=score, reverse=True)
        selected: List[str] = []
        seen = set()
        for sent in ranked:
            key = sent.lower()
            if key in seen:
                continue
            seen.add(key)
            selected.append(sent)
            if len(selected) >= 2:
                break

        if not selected:
            selected = [str(top_chunks[0]["text"]).strip()]

        answer = " ".join(selected).strip()
        if answer and answer[-1] not in ".!?":
            answer += "."
        return f"{answer} [Sources: {', '.join(f'[{c}]' for c in chunks)}]"
