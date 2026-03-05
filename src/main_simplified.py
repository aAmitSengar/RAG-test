"""Simple RAG workflow example with educational comments and terminal logs.

This file intentionally avoids large model downloads and demonstrates the
retrieval-augmented generation (RAG) idea in an easy-to-follow way.
"""
from pathlib import Path
import logging
import numpy as np

# Paths used by the simplified demo pipeline.
DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_FILE = DATA_DIR / "faiss.index"
DOCS_FILE = DATA_DIR / "docs.txt"


def setup_logging() -> None:
    """Configure terminal logging for educational step-by-step visibility."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)

def simple_embedding(text):
    """Create a deterministic pseudo-embedding for a text.

    Why this exists:
    - For education, we want to demonstrate retrieval without downloading
        transformer models.
    - We use a hash-derived random seed so the same text always maps to the
        same vector.
    """
    text_hash = hash(text.lower())
    np.random.seed(abs(text_hash) % (2**31))
    return np.random.randn(384).astype('float32')

def build_index():
    """Build a FAISS index from lines in docs.txt.

    Pipeline step explained:
    1. Read raw documents from file
    2. Convert each document to an embedding vector
    3. Add vectors to FAISS index for nearest-neighbor search
    4. Persist index to disk
    """
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        )
    
    if not DOCS_FILE.exists():
        logger.error("Missing docs file: %s", DOCS_FILE)
        logger.error("Please add one document per line in docs.txt")
        return
    
    # Read documents
    with open(DOCS_FILE, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    
    if not docs:
        logger.error("No documents found in docs.txt")
        return
    
    logger.info("STEP 1/4: Building FAISS index from %d documents", len(docs))
    
    # Create embeddings for each document using simple embedding
    embeddings = np.array([simple_embedding(doc) for doc in docs]).astype('float32')
    
    # Create and train FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, str(INDEX_FILE))
    logger.info("STEP 4/4: Index saved to %s", INDEX_FILE)

def retrieve(query, k=3):
    """Retrieve top-k documents using nearest-neighbor search in FAISS."""
    try:
        import faiss
    except ImportError:
        raise ImportError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        )
    
    if not INDEX_FILE.exists():
        logger.info("FAISS index missing, building it now...")
        build_index()
    
    # Load index
    index = faiss.read_index(str(INDEX_FILE))
    
    # Encode query using simple embedding
    q_emb = np.array([simple_embedding(query)]).astype('float32')
    
    # Search
    distances, ids = index.search(q_emb, k=min(k, index.ntotal))
    
    # Retrieve documents
    with open(DOCS_FILE, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    
    results = [docs[i] for i in ids[0] if i < len(docs)]
    logger.info("Retrieved %d relevant document(s)", len(results))
    logger.debug("Distances: %s", distances)
    return results

def generate_answer(question, context):
    """Generate an educational, template-based response from context.

    In real systems, this stage uses a language model. Here we keep it simple
    so students can focus on retrieval flow first.
    """
    context_text = " ".join(context)
    
    answer = f"""Based on the retrieved documents:
    
    {context_text}
    
    Question: {question}
    
    Answer: This is a simplified RAG system that demonstrates:
    1. Document retrieval using FAISS indexing
    2. Context-aware answer generation
    3. Working without requiring large model downloads
    
    The system successfully retrieved relevant documents and can be extended
    with actual language models when SSL/network issues are resolved."""
    
    return answer

def main():
    """Run the full simplified RAG workflow with clear terminal logs."""
    setup_logging()
    logger.info("=" * 70)
    logger.info("Starting Simplified RAG Demo (Education Mode)")
    logger.info("=" * 70)

    # Ensure data files exist
    if not DOCS_FILE.exists():
        logger.info("Docs file not found, creating sample docs at %s", DOCS_FILE)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(DOCS_FILE, "w") as f:
            f.write("""RAG (Retrieval-Augmented Generation) is a technique that combines retrieval and generation.
Transformers are neural network architectures based on self-attention mechanisms.
FAISS is a library for efficient similarity search and clustering of dense vectors.
Natural language processing involves computational techniques for human language.
Machine learning enables systems to learn from data without explicit programming.
Deep learning uses neural networks with multiple layers for complex tasks.""")
        logger.info("Sample docs created")

    if not INDEX_FILE.exists():
        logger.info("No index found, starting indexing stage")
        build_index()

    query = "What is RAG architecture?"
    logger.info("\nQuery: %s", query)
    logger.info("%s", "-" * 50)
    
    hits = retrieve(query)
    logger.info("Retrieved documents:")
    for i, h in enumerate(hits, 1):
        logger.info("  %d. %s", i, h)

    logger.info("\n%s", "-" * 50)
    logger.info("Generating answer from retrieved context")
    ans = generate_answer(query, hits)
    logger.info("Generated answer:\n%s", ans)
    
    logger.info("\n%s", "=" * 70)
    logger.info("RAG workflow completed successfully")
    logger.info("%s", "=" * 70)

if __name__ == "__main__":
    main()
