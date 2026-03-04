"""Generator module for answer generation."""

import logging
from typing import List

from .config import Config

logger = logging.getLogger(__name__)


class Generator:
    """Sequence-to-sequence based answer generator."""

    def __init__(self, config: Config):
        """
        Initialize the generator.
        
        Args:
            config: Configuration object containing model paths.
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        logger.info("[Generator] Initializing generator component")
        self._load_model()

    def _load_model(self):
        """Load the generation model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            model_source = self.config.gen_model
            local_only = self.config.gen_model_is_local

            logger.info("[Generator][Step 1/3] Loading tokenizer and generation model")
            logger.info(f"[Generator] Model source: {model_source}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_source,
                    local_files_only=local_only
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_source,
                    local_files_only=local_only
                )
                logger.info("[Generator] ✓ Generator model loaded successfully")
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
                    local_files_only=False
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    fallback_model,
                    local_files_only=False
                )
                logger.info("[Generator] ✓ Fallback generation model loaded successfully")
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: "
                "pip install transformers"
            )
        except Exception as e:
            logger.error(
                "Error loading generation model: %s. "
                "Continuing in template-only mode.",
                e,
            )
            self.tokenizer = None
            self.model = None

    @staticmethod
    def _fallback_model_name(model_source: str) -> str:
        """Resolve a reasonable Hugging Face model id for fallback loading."""
        return model_source.rstrip("/").split("/")[-1] or "t5-small"

    def generate(self, question: str, context: List[str]) -> str:
        """
        Generate an answer based on question and context.
        
        Args:
            question: The question to answer.
            context: List of context documents.
            
        Returns:
            Generated answer string.
        """
        if not context:
            return "No context provided to generate an answer."

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Generation model is unavailable")

        try:
            # STEP 1: Build a single prompt from question + retrieved context.
            # This is where retrieval output is fused into generation input.
            context_text = "\n".join(context)
            prompt = f"Question: {question}\n\nContext: {context_text}\n\nAnswer:"
            
            logger.info("[Generator][Step 1/3] Prompt prepared")
            logger.info("[Generator][Step 2/3] Running model inference")
            
            # STEP 2: Tokenize prompt and run seq2seq decoding.
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
            
            # STEP 3: Decode generated token ids back into text.
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("[Generator][Step 3/3] ✓ Answer generated successfully")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    def generate_with_fallback(self, question: str, context: List[str]) -> str:
        """
        Generate answer with fallback to template if model fails.
        
        Args:
            question: The question to answer.
            context: List of context documents.
            
        Returns:
            Generated or template-based answer.
        """
        try:
            return self.generate(question, context)
        except Exception as e:
            logger.warning(f"[Generator] Generation failed, using template fallback: {e}")
            return self._template_answer(question, context)

    @staticmethod
    def _template_answer(question: str, context: List[str]) -> str:
        """Generate a template-based answer."""
        context_text = "\n".join(f"- {c}" for c in context)
        return (
            f"Question: {question}\n\n"
            f"Based on the retrieved documents:\n"
            f"{context_text}\n\n"
            f"These documents contain relevant information to answer your question."
        )
