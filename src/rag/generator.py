
"""
Generator module for answer generation.

FINAL VERSION:
- General, event-aware date selection
- Detailed WHEN answers (date + explanation)
- Evidence-scoped citations
- No static facts, no hacks, no syntax bugs
"""

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config

logger = logging.getLogger(__name__)

# -------------------------------------------------
# General verb heuristics (domain-agnostic)
# -------------------------------------------------

EVENT_VERBS = {
    "became", "become", "got", "gained", "achieved",
    "declared", "ended", "started", "formed",
    "passed", "enacted", "signed", "established", "created"
}

BIO_VERBS = {
    "born", "died", "trained", "studied",
    "lawyer", "activist", "reformer", "leader"
}


class Generator:
    """Seq2Seq-based RAG answer generator with event-aware reasoning."""

    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load_model()

    # -------------------------------------------------
    # Model loading
    # -------------------------------------------------

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            import torch

            model_id = self.config.gen_model
            local_only = (
                self.config.gen_model_is_local
                or getattr(self.config, "use_local_only", False)
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, local_files_only=local_only
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_id, local_files_only=local_only
            )

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            logger.info("[Generator] Model loaded on %s", self.device)

        except Exception as exc:
            logger.error("[Generator] Model loading failed: %s", exc)
            self.model = None
            self.tokenizer = None

    # -------------------------------------------------
    # Prompt
    # -------------------------------------------------

    def _build_prompt(
        self, question: str, context: Sequence[Dict[str, object]]
    ) -> str:
        persona = getattr(
            self.config, "persona", "You are a knowledgeable teacher."
        )
        language = getattr(self.config, "answer_language", "English")

        ctx = [
            f"[Chunk {c['chunk_id']}]\n{c['text']}"
            for c in context
        ]

        return (
            f"{persona}\n"
            f"Respond in {language}. Do NOT repeat instructions.\n\n"
            "Rules:\n"
            "- Answer ONLY using the context below.\n"
            "- Explain in your own words.\n"
            "- Cite evidence like [12].\n\n"
            f"Question:\n{question}\n\n"
            f"Context:\n{'\n\n'.join(ctx)}\n\n"
            "Answer:\n"
        )

    # -------------------------------------------------
    # Generation
    # -------------------------------------------------

    def generate(
        self, question: str, context: List[Dict[str, object]]
    ) -> str:
        if not context:
            return "Insufficient context to answer confidently."
        if not self.model or not self.tokenizer:
            raise RuntimeError("Generation model unavailable")

        prompt = self._build_prompt(question, context)

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=1024
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=getattr(self.config, "max_new_tokens", 300),
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=1.1,
            early_stopping=True,
        )

        text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()

        return self._clean_instruction_echoes(text)

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def generate_with_fallback(
        self, question: str, context: List[Dict[str, object]]
    ) -> Dict[str, object]:

        try:
            answer = self.generate(question, context)
            if self._is_low_quality(answer):
                answer = self._template_answer(context)
        except Exception:
            answer = self._template_answer(context)

        final_answer, citations = self._refine_answer(
            question, answer, context
        )

        return {
            "answer": final_answer,
            "citations": citations,
        }

    # -------------------------------------------------
    # Refinement (KEY LOGIC)
    # -------------------------------------------------

    def _refine_answer(
        self,
        question: str,
        answer: str,
        context: List[Dict[str, object]],
    ) -> Tuple[str, List[int]]:

        body = self._strip_citations(answer)

        # ---------- WHEN / DATE QUESTIONS ----------
        if self._is_when_question(question):
            selection = self._select_event_sentence(question, context)
            if selection:
                date, sentence, chunk_id = selection

                explanation = self._expand_explanation(
                    sentence, context, chunk_id
                )

                body = (
                    f"It happened in {date}. {explanation}"
                    if explanation
                    else f"It happened in {date}."
                )
                citations = [chunk_id]
            else:
                body = self._compress(body)
                citations = self._fallback_citations(context)

        # ---------- DEFAULT ----------
        else:
            body = self._compress(body)
            citations = self._fallback_citations(context)

        suffix = self._source_suffix(citations)
        return f"{body.strip()} {suffix}".strip(), citations

    # -------------------------------------------------
    # Event-aware selection (date + sentence + chunk)
    # -------------------------------------------------

    def _select_event_sentence(
        self,
        question: str,
        context: List[Dict[str, object]],
    ) -> Optional[Tuple[str, str, int]]:

        q_terms = set(self._terms(question))
        best_score = -1.0
        best_result = None

        for chunk in context:
            chunk_score = float(chunk.get("score", 0.0))
            chunk_id = int(chunk["chunk_id"])
            text = str(chunk["text"])

            for sent in self._sentences(text):
                date = self._extract_date(sent)
                if not date:
                    continue

                s_lower = sent.lower()
                s_terms = set(self._terms(sent))

                overlap = len(q_terms & s_terms)
                event_bonus = 3 if any(v in s_lower for v in EVENT_VERBS) else 0
                bio_penalty = -3 if any(v in s_lower for v in BIO_VERBS) else 0

                score = overlap * 2 + chunk_score + event_bonus + bio_penalty

                if score > best_score:
                    best_score = score
                    best_result = (date, sent, chunk_id)

        return best_result

    # -------------------------------------------------
    # Explanation expansion (same chunk only)
    # -------------------------------------------------

    def _expand_explanation(
        self,
        anchor_sentence: str,
        context: List[Dict[str, object]],
        chunk_id: int,
        max_sentences: int = 2,
    ) -> str:
        for chunk in context:
            if int(chunk["chunk_id"]) != chunk_id:
                continue

            sentences = self._sentences(str(chunk["text"]))
            if anchor_sentence not in sentences:
                return ""

            idx = sentences.index(anchor_sentence)
            expanded = sentences[idx : idx + max_sentences]
            return " ".join(expanded)

        return ""

    # -------------------------------------------------
    # Utilities
    # -------------------------------------------------

    @staticmethod
    def _terms(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        stop = {"what", "when", "where", "which", "about", "year"}
        return [w for w in words if w not in stop]

    @staticmethod
    def _extract_date(text: str) -> Optional[str]:
        match = re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", text)
        return match.group(0) if match else None

    @staticmethod
    def _is_when_question(question: str) -> bool:
        return bool(
            re.search(
                r"\bwhen\b|\bwhat year\b|\bwhat date\b",
                question.lower(),
            )
        )

    @staticmethod
    def _sentences(text: str) -> List[str]:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]

    @staticmethod
    def _compress(text: str, max_words: int = 28) -> str:
        words = text.split()
        return (
            text
            if len(words) <= max_words
            else " ".join(words[:max_words]) + "..."
        )

    @staticmethod
    def _strip_citations(text: str) -> str:
        return re.sub(r"\[\d+\]", "", text).strip()

    @staticmethod
    def _clean_instruction_echoes(text: str) -> str:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(
            p for p in parts
            if not p.lower().startswith(
                ("answer in ", "respond in ", "you are ")
            )
        ).strip()

    @staticmethod
    def _is_low_quality(text: str) -> bool:
        return not text or len(text.split()) < 2

    @staticmethod
    def _template_answer(context: List[Dict[str, object]]) -> str:
        for chunk in context:
            text = str(chunk.get("text", "")).strip()
            if text:
                return Generator._sentences(text)[0]
        return "Insufficient context to answer confidently."

    @staticmethod
    def _fallback_citations(context: List[Dict[str, object]]) -> List[int]:
        return [int(c["chunk_id"]) for c in context[:2]]

    def _source_suffix(self, citations: List[int]) -> str:
        if not getattr(self.config, "citations_enabled", True):
            return ""
        citations = list(dict.fromkeys(citations))
        return f"[Sources: {', '.join(f'[{i}]' for i in citations)}]"
