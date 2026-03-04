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
from pathlib import Path

# Add the 'src' directory to Python path to enable direct imports of 'rag' module.
# This setup is common in smaller projects and detailed further in README.md.
sys.path.insert(0, str(Path(__file__).parent))

from rag.config import Config
from rag.retriever import Retriever
from rag.generator import Generator
from rag.utils import setup_logging


# Configure logging for better visibility into what's happening
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    This class orchestrates the entire RAG workflow:
    1. Builds the FAISS index from documents
    2. Retrieves relevant documents for a query
    3. Generates an answer using the retrieved context
    """
    
    def __init__(self):
        """Initialize the RAG pipeline with configuration, retriever, and generator."""
        logger.info("=" * 60)
        logger.info("Initializing RAG Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load configuration
            self.config = Config()
            logger.info(f"Configuration loaded:\n{self.config}")
            
            # Step 2: Initialize retriever (embedding model for document search)
            self.retriever = Retriever(self.config)
            
            # Step 3: Initialize generator (LLM for answer generation)
            self.generator = Generator(self.config)
            
            logger.info("✓ Pipeline initialized successfully\n")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def run(self, query: str, k: int = 3):
        """
        Run the complete RAG pipeline.
        
        Args:
            query: The question to answer
            k: Number of documents to retrieve (default: 3)
            
        Returns:
            Dictionary with retrieved documents and generated answer
        """
        logger.info("=" * 60)
        logger.info(f"Running RAG Pipeline")
        logger.info("=" * 60)
        logger.info(f"\nQuery: {query}\n")
        
        try:
            # STEP 1: RETRIEVAL
            # ---------------------
            # The retriever encodes the query using a sentence transformer
            # and searches the FAISS index to find k most similar documents
            logger.info("STEP 1: Retrieving relevant documents...")
            logger.info("-" * 60)
            retrieved_docs = self.retriever.retrieve(query, k=k)
            
            logger.info(f"Retrieved {len(retrieved_docs)} document(s):")
            for i, doc in enumerate(retrieved_docs, 1):
                # Truncate long documents for logging
                doc_preview = doc[:100] + "..." if len(doc) > 100 else doc
                logger.info(f"  [{i}] {doc_preview}")
            
            # STEP 2: GENERATION
            # ---------------------
            # The generator takes the original question and retrieved context
            # and uses a seq2seq model (like T5) to generate a natural answer
            logger.info("\nSTEP 2: Generating answer from retrieved context...")
            logger.info("-" * 60)
            answer = self.generator.generate_with_fallback(query, retrieved_docs)
            
            logger.info(f"Generated Answer:\n{answer}\n")
            
            return {
                "query": query,
                "retrieved_documents": retrieved_docs,
                "answer": answer
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def build_index(self):
        """Build FAISS index from documents. Call this first if index doesn't exist."""
        logger.info("Building FAISS index from documents...")
        num_docs = self.retriever.build_index()
        logger.info(f"✓ Index built with {num_docs} documents\n")


def _validate_setup(rag: 'RAGPipeline') -> bool:
    """
    Validate that all required files and models are available.
    
    Args:
        rag: The RAG pipeline instance.
        
    Returns:
        True if setup is valid, False otherwise.
    """
    # Check if docs.txt exists
    if not rag.config.docs_file.exists():
        logger.error("\n❌ ERROR: Documentation file not found at %s", rag.config.docs_file)
        logger.error("\nPlease create a 'docs.txt' file in the data/ folder with one document per line.")
        logger.error("\nExample docs.txt format:")
        logger.error("-" * 60)
        logger.error("RAG combines retrieval with generation for better answers.")
        logger.error("FAISS enables efficient similarity search at scale.")
        logger.error("Embeddings represent text as numerical vectors in space.")
        logger.error("-" * 60)
        return False
    
    # Check if docs have actual content
    with open(rag.config.docs_file, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    
    if not docs:
        logger.error("❌ ERROR: docs.txt is empty. Please add documents to search from.")
        return False
    
    logger.info(f"✓ Found {len(docs)} documents in {rag.config.docs_file}")
    return True


def _build_index_if_needed(rag: 'RAGPipeline') -> bool:
    """
    Build FAISS index if it doesn't exist.
    
    Args:
        rag: The RAG pipeline instance.
        
    Returns:
        True if index is ready, False if there was an error.
    """
    if rag.config.index_file.exists():
        logger.info(f"✓ FAISS index already exists at {rag.config.index_file}\n")
        return True
    
    logger.info("Building FAISS index from documents...")
    try:
        rag.build_index()
        return True
    except Exception as e:
        logger.error(f"❌ Failed to build index: {e}")
        return False


def _run_single_query(rag: 'RAGPipeline', query: str) -> dict:
    """
    Run a single RAG query and return results.
    
    Args:
        rag: The RAG pipeline instance.
        query: The question to answer.
        
    Returns:
        Dictionary with query, retrieved_documents, and answer.
    """
    return rag.run(query, k=rag.config.retrieval_k)


def _print_result(result: dict) -> None:
    """
    Pretty print RAG result.
    
    Args:
        result: Result dictionary from RAG pipeline.
    """
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULT")
    logger.info("=" * 70)
    logger.info(f"\nQuestion: {result['query']}")
    logger.info(f"\nRetrieved {len(result['retrieved_documents'])} document(s):")
    for i, doc in enumerate(result['retrieved_documents'], 1):
        doc_preview = doc[:80] + "..." if len(doc) > 80 else doc
        logger.info(f"  [{i}] {doc_preview}")
    logger.info(f"\nGenerated Answer:\n{result['answer']}")
    logger.info("\n" + "=" * 70 + "\n")


def main():
    """
    Main RAG teaching example.
    
    This demonstrates the complete RAG pipeline:
    1. Initialize components (retriever and generator)
    2. Build or load the FAISS index
    3. Run queries and generate answers
    4. Optionally enable interactive mode for multiple queries
    """
    try:
        # Step 1: Initialize pipeline
        rag = RAGPipeline()
        
        # Step 2: Validate setup
        if not _validate_setup(rag):
            logger.info("\n⚠️  Setup incomplete. Please fix the issues above and try again.")
            sys.exit(1)
        
        # Step 3: Build or load index
        if not _build_index_if_needed(rag):
            sys.exit(1)
        
        # Step 4: Run sample queries
        sample_queries = [
            "What is RAG?",
            "How does retrieval work?",
            "What are embeddings used for?"
        ]
        
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING SAMPLE QUERIES")
        logger.info("=" * 70 + "\n")
        
        for query in sample_queries:
            try:
                result = _run_single_query(rag, query)
                _print_result(result)
            except Exception as e:
                logger.error(f"❌ Query failed: {e}")
                continue
        
        # Step 5: Optional interactive mode
        logger.info("\n" + "=" * 70)
        logger.info("INTERACTIVE MODE (optional)")
        logger.info("=" * 70)
        logger.info("You can now ask questions interactively.")
        logger.info("Type 'quit' or 'exit' to stop.\n")
        
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
                _print_result(result)
                
            except KeyboardInterrupt:
                logger.info("\n\n✓ Interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                continue
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
