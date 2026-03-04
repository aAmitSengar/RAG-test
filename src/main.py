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


def main():
    """
    Main teaching example showing how to use RAG.
    
    This demonstrates:
    1. Initializing the RAG pipeline
    2. Building the index (if needed)
    3. Running a complete RAG query
    """
    
    try:
        # Initialize the pipeline
        rag = RAGPipeline()
        
        # Check if docs.txt exists and has data
        if not rag.config.docs_file.exists():
            logger.error(f"\n❌ ERROR: Documentation file not found at {rag.config.docs_file}")
            logger.error("Please create a 'docs.txt' file in the data/ folder with one document per line.")
            logger.error("\nExample docs.txt:")
            logger.error("-" * 60)
            logger.error("RAG is a technique that combines retrieval with generation.")
            logger.error("FAISS is a library for efficient similarity search.")
            logger.error("Embeddings represent text as numerical vectors.")
            logger.error("-" * 60)
            return
        
        # Build index if it doesn't exist
        if not rag.config.index_file.exists():
            logger.info("📁 FAISS index not found. Building from documents...")
            rag.build_index()
        
        # Run RAG on a sample question
        # ============================
        # You can modify this query or make it interactive
        sample_query = "What is RAG architecture?"
        
        result = rag.run(sample_query, k=3)
        
        # Print formatted result
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RESULT")
        logger.info("=" * 60)
        logger.info(f"\nQuestion: {result['query']}")
        logger.info(f"\nContext Documents: {len(result['retrieved_documents'])}")
        logger.info(f"\nAnswer:\n{result['answer']}")
        logger.info("\n" + "=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
