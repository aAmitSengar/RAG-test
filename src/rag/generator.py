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
        self._load_model()

    def _load_model(self):
        """Load the generation model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            logger.info(f"Loading generation model: {self.config.gen_model}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.gen_model,
                local_files_only=self.config.gen_model_is_local
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.gen_model,
                local_files_only=self.config.gen_model_is_local
            )
            logger.info("✓ Generator model loaded successfully")
        except ImportError:
            raise ImportError(
                "transformers is required. Install with: "
                "pip install transformers"
            )
        except Exception as e:
            logger.error(f"Error loading generation model: {e}")
            raise

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

        try:
            # Prepare prompt
            context_text = "\n".join(context)
            prompt = f"Question: {question}\n\nContext: {context_text}\n\nAnswer:"
            
            logger.info("Generating answer...")
            
            # Tokenize and generate
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
            outputs = self.model.generate(
                **inputs,
                max_length=150,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info("✓ Answer generated successfully")
            
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
            logger.warning(f"Generation failed, using template: {e}")
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
