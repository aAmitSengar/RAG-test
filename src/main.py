"""Simple RAG workflow example."""
import os
from pathlib import Path

# load environment variables (HF token, model overrides, etc.)
from dotenv import load_dotenv
load_dotenv()

# Fix SSL on macOS by setting environment variable
import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

# force offline mode to bypass SSL/network issues
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
# os.environ["HF_HUB_OFFLINE"] = "1"

# you can adapt these imports to your own retriever/generator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_FILE = DATA_DIR / "faiss.index"
DOCS_FILE = DATA_DIR / "docs.txt"


def build_index(model_name: str | None = None):
    """Read documents from data/ and build a FAISS index.

    The model_name can be a Hugging Face identifier (e.g. "all-MiniLM-L6-v2")
    or a local directory containing the pre-downloaded model.  An alternative
    environment variable `EMB_MODEL` is checked when `model_name` is None.
    If that is also unset, the function will look for a sibling folder
    `all-MiniLM-L6-v2` in the project and use it automatically.
    """
    # determine which embedding model to use
    if model_name is None:
        model_name = os.getenv("EMB_MODEL")
        if model_name is None:
            # prefer a local clone if present
            local_emb_folder = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2"
            if local_emb_folder.exists():
                model_name = str(local_emb_folder)
            else:
                model_name = "all-MiniLM-L6-v2"

    # try to instantiate; if download fails, and path is local, try again
    try:
        is_local = Path(model_name).exists()
        encoder = SentenceTransformer(model_name, local_files_only=is_local)
    except Exception as e:
        print(f"\nError loading sentence-transformers model '{model_name}':", e)
        if not Path(model_name).exists():
            print("The name above may require an internet connection."
                  " To work offline, download the model folder manually" \
                  " and set EMB_MODEL to its path.")
        print("Also ensure your system has valid SSL certificates (see README).")
        raise

    texts = []
    with open(DOCS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)

    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"Indexed {len(texts)} documents.")


def retrieve(query: str, k: int = 3):
    """Return top-k text spans relevant to the query."""
    index = faiss.read_index(str(INDEX_FILE))
    
    # Use the same local logic as build_index
    local_emb_folder = Path(__file__).parent.parent / "models" / "all-MiniLM-L6-v2"
    model_name = str(local_emb_folder) if local_emb_folder.exists() else "all-MiniLM-L6-v2"
    is_local = local_emb_folder.exists()
    
    encoder = SentenceTransformer(model_name, local_files_only=is_local)
    q_emb = encoder.encode([query], convert_to_numpy=True)
    distances, ids = index.search(q_emb, k)
    with open(DOCS_FILE, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    results = [docs[i] for i in ids[0]]
    return results


def generate_answer(question: str, context: list[str]):
    """Simple generator using a seq2seq LM (e.g. T5)"""
    model_name = os.getenv("GEN_MODEL")
    if model_name is None:
        # check for a local folder clone first
        local_gen = Path(__file__).parent.parent / "models" / "t5-small"
        if local_gen.exists():
            model_name = str(local_gen)
        else:
            model_name = "t5-small"

    try:
        is_local = Path(model_name).exists()
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=is_local)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=is_local)
    except Exception as e:
        print(f"\nError loading generation model '{model_name}':", e)
        print("Verify network/SSL configuration or set GEN_MODEL to a local path.")
        raise

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

    if not INDEX_FILE.exists():
        build_index()

    query = "What is RAG architecture?"
    hits = retrieve(query)
    print("Retrieved passages:")
    for h in hits:
        print('-', h)

    # Try to generate answer, but gracefully handle model loading errors
    try:
        ans = generate_answer(query, hits)
        print("\nGenerated answer:\n", ans)
    except Exception as e:
        print(f"\nNote: Could not generate answer due to model loading issue.")
        print(f"The embedding and retrieval parts are working correctly!")
        print(f"Error details: {type(e).__name__}: {str(e)[:100]}...")
        print("\nTo complete RAG functionality, you need to download the t5-small model.")
        print("Consider running: python src/download_models.py")
        print("Or set GEN_MODEL environment variable to a local model path.")


if __name__ == "__main__":
    main()
