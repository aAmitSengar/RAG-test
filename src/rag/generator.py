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
            "Respond in a conversational style with short sentences.\n"
            "Do not copy long passages from the context.\n"
            "Start with a direct answer first, then a brief reason.\n"
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

        answer = self._refine_answer(question, answer, context)
        citations = self._extract_citations(answer)
        return {"answer": answer, "citations": citations}

    @staticmethod
    def _is_yes_no_question(question: str) -> bool:
        """Detect simple yes/no question forms."""
        text = (question or "").strip().lower()
        return bool(
            re.match(
                r"^(is|are|was|were|do|does|did|can|could|should|would|will|has|have|had)\b",
                text,
            )
        )

    @staticmethod
    def _compress_sentence(text: str, max_words: int = 22) -> str:
        """Keep output concise to avoid long extractive copy."""
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return ""
        words = cleaned.split()
        if len(words) <= max_words:
            return cleaned
        return " ".join(words[:max_words]).rstrip(",;:") + "..."

    @staticmethod
    def _infer_yes_no_from_evidence(sentence: str) -> str:
        """Infer a lightweight yes/no stance from top evidence."""
        text = (sentence or "").lower()
        negation_markers = (" not ", " no ", " never ", " none ", " without ", " lacked ", " lack ")
        if any(marker in f" {text} " for marker in negation_markers):
            return "No"
        return "Yes"

    @staticmethod
    def _is_when_question(question: str) -> bool:
        """Detect time/date seeking questions."""
        q = (question or "").strip().lower()
        return bool(
            re.search(r"\bwhen\b|\bwhat\s+date\b|\bwhich\s+year\b|\bwhat\s+year\b", q)
        )

    @staticmethod
    def _strip_citation_markers(text: str) -> str:
        """Remove citation-style markers from body text."""
        cleaned = re.sub(
            r"\[Sources:\s*(?:\[\d+\](?:,\s*)?)+\]",
            "",
            (text or ""),
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\[\d+\]", "", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _extract_date(text: str) -> str:
        """Extract the most specific date available from text."""
        if not text:
            return ""
        month = (
            r"January|February|March|April|May|June|July|August|September|October|"
            r"November|December"
        )
        patterns = [
            rf"\b\d{{1,2}}\s+(?:{month})\s+\d{{4}}\b",
            rf"\b(?:{month})\s+\d{{1,2}},?\s+\d{{4}}\b",
            r"\b\d{4}\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                return match.group(0)
        return ""

    @staticmethod
    def _source_suffix(citations: List[int], context: List[Dict[str, int | float | str]]) -> str:
        """Build sources suffix with stable fallback ids."""
        source_ids = citations or [int(chunk["chunk_id"]) for chunk in context[:3]]
        source_ids = list(dict.fromkeys(source_ids))
        return f"[Sources: {', '.join(f'[{cid}]' for cid in source_ids)}]"

    @staticmethod
    def _refine_answer(
        question: str, answer: str, context: List[Dict[str, int | float | str]]
    ) -> str:
        """Refine answer to concise interactive style."""
        if not answer:
            return "Insufficient context to answer confidently."

        citations = Generator._extract_citations(answer)
        body = Generator._strip_citation_markers(answer)
        if not body:
            body = "Insufficient context to answer confidently."

        if Generator._is_when_question(question):
            candidate_text = " ".join(
                [body] + [str(item["text"]) for item in context[:3]]
            )
            date = Generator._extract_date(candidate_text)
            q_lower = (question or "").lower()
            if date:
                if "india" in q_lower and ("freedom" in q_lower or "independ" in q_lower):
                    body = f"India got freedom on {date}."
                else:
                    body = f"It happened on {date}."
            else:
                body = Generator._compress_sentence(body, max_words=20)
                if body and body[-1] not in ".!?":
                    body += "."
        else:
            sentences = Generator._split_sentences(body)
            compact = sentences[0] if sentences else body
            body = Generator._compress_sentence(compact, max_words=24)
            if body and body[-1] not in ".!?":
                body += "."

        return f"{body} {Generator._source_suffix(citations, context)}"

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

        concise = [Generator._compress_sentence(sent) for sent in selected if sent.strip()]
        concise = [sent for sent in concise if sent]
        if not concise:
            concise = ["I found relevant context, but it is limited."]

        if Generator._is_yes_no_question(question):
            verdict = Generator._infer_yes_no_from_evidence(concise[0])
            reason = concise[0]
            if reason and reason[-1] not in ".!?":
                reason += "."
            answer = f"{verdict}, based on the retrieved context. {reason}"
        else:
            body = " ".join(concise[:2]).strip()
            if body and body[-1] not in ".!?":
                body += "."
            answer = body

        return f"{answer} [Sources: {', '.join(f'[{c}]' for c in chunks)}]"
