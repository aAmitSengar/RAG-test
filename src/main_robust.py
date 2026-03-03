"""RAG workflow - With fallback to offline mode if models are unavailable."""
import os
from pathlib import Path

# Load environment variables (HF token, model overrides, etc.)
from dotenv import load_dotenv
load_dotenv()

import sys

# Try online mode first, fall back to offline if needed
try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import faiss
    import numpy as np
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    print("Warning: transformers not available, using simplified mode")

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_FILE = DATA_DIR / "faiss.index"
DOCS_FILE = DATA_DIR / "docs.txt"

# Model paths
LOCAL_MODELS = {
    "embedding": Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2",
    "generation": Path(__file__).parent.parent / "models" / "t5-small"
}

# Global encoder and model
encoder = None
generator_model = None
generator_tokenizer = None


def load_embedding_model():
    """Load the embedding model with fallback strategy."""
    global encoder
    if encoder is not None:
        return encoder
    
    if not HAVE_TRANSFORMERS:
        raise ImportError("sentence_transformers not available")
    
    model_path = LOCAL_MODELS["embedding"]
    
    # Check if local model exists and is valid
    if model_path.exists() and (model_path / "pytorch_model.bin").exists():
        try:
            print(f"Loading embedding model from {model_path}...")
            encoder = SentenceTransformer(str(model_path), local_files_only=True)
            print("✓ Embedding model loaded successfully")
            return encoder
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            print("Attempting to download from Hugging Face...")
    
    # Try downloading from Hugging Face
    try:
        print("Downloading embedding model from Hugging Face...")
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", local_files_only=False)
        print("✓ Embedding model downloaded successfully")
        return encoder
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise RuntimeError("Failed to load embedding model. Check network/SSL configuration or use main_simplified.py")


def load_generation_model():
    """Load the generation model with fallback strategy."""
    global generator_model, generator_tokenizer
    if generator_model is not None:
        return generator_model, generator_tokenizer
    
    if not HAVE_TRANSFORMERS:
        raise ImportError("transformers not available")
    
    model_path = LOCAL_MODELS["generation"]
    
    # Check if local model exists and is valid
    if model_path.exists() and (model_path / "pytorch_model.bin").exists():
        try:
            print(f"Loading generation model from {model_path}...")
            generator_tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
            generator_model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)
            print("✓ Generation model loaded successfully")
            return generator_model, generator_tokenizer
        except Exception as e:
            print(f"✗ Error loading local model: {e}")
            print("Attempting to download from Hugging Face...")
    
    # Try downloading from Hugging Face
    try:
        print("Downloading generation model from Hugging Face...")
        generator_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small", local_files_only=False)
        generator_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", local_files_only=False)
        print("✓ Generation model downloaded successfully")
        return generator_model, generator_tokenizer
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise RuntimeError("Failed to load generation model. Check network/SSL configuration or use main_simplified.py")


def build_index(model_name: str | None = None):
    """Read documents from data/ and build a FAISS index."""
    encoder = load_embedding_model()
    
    texts = []
    with open(DOCS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)

    embeddings = encoder.encode(texts, convert_to_numpy=True)
    embeddings = np.asarray(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Indexed {len(texts)} documents.")


def retrieve(query: str, k: int = 3):
    """Return top-k text spans relevant to the query."""
    index = faiss.read_index(str(INDEX_FILE))
    encoder = load_embedding_model()
    
    q_emb = encoder.encode([query], convert_to_numpy=True)
    distances, ids = index.search(q_emb, k)
    with open(DOCS_FILE, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    results = [docs[i] for i in ids[0]]
    return results


def generate_answer(question: str, context: list[str]):
    """Simple generator using a seq2seq LM (e.g. T5)"""
    model, tokenizer = load_generation_model()

    prompt = "\n\n".join([question] + context)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def main():
    # ensure data files exist
    if not DOCS_FILE.exists():
        print("Please add a docs.txt file under data/ with one document per line.")
        return

    try:
        if not INDEX_FILE.exists():
            print("Building FAISS index...")
            build_index()

        query = "What is RAG architecture?"
        print(f"\nQuery: {query}")
        
        print("\nRetrieving passages...")
        hits = retrieve(query)
        print("Retrieved passages:")
        for h in hits:
            print('-', h)

        print("\nGenerating answer...")
        ans = generate_answer(query, hits)
        print("\nGenerated answer:\n", ans)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nFallback: Use 'python src/main_simplified.py' for offline mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
