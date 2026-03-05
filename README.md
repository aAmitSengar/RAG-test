# RAG-test: A Simple Retrieval-Augmented Generation (RAG) Example

This project provides a basic, runnable example of a RAG pipeline using Python.
It demonstrates how to combine document retrieval with a language model
to answer questions based on a given set of documents.

## Features

- **Modular Design**: Separates concerns into `Config`, `Retriever`, and `Generator` classes.
- **FAISS Integration**: Uses FAISS for efficient similarity search over document embeddings.
- **Hybrid Retrieval**: Combines dense FAISS scores with sparse BM25-style lexical scores.
- **Query Rewriting**: Expands user queries with lightweight domain synonyms before retrieval.
- **Context Compression**: Applies query-aware extractive compression before generation.
- **Sentence Transformers**: Leverages pre-trained models for creating document and query embeddings.
- **Hugging Face Transformers**: Integrates with Hugging Face `transformers` library for answer generation.
- **Local Model Support**: Automatically uses locally downloaded models if available, enabling offline usage.
- **Clear Logging**: Provides informative logs for better understanding of the RAG workflow.

## Project Structure

```
. # Project Root
├── data/
│   ├── docs.txt             # Your raw documents (one per line)
│   └── faiss.index          # (Generated) FAISS index of document embeddings
├── models/
│   ├── all-MiniLM-L6-v2/    # (Optional) Local Sentence Transformer model
│   └── t5-small/            # (Optional) Local T5 generation model
├── src/
│   ├── main.py              # Main entry point for the RAG pipeline
│   ├── download_models.py   # Script to download models locally
│   └── rag/
│       ├── __init__.py      # Package initializer
│       ├── config.py        # Configuration management
│       ├── generator.py     # Handles answer generation
│       ├── retriever.py     # Handles document retrieval and index building
│       └── utils.py         # Utility functions (e.g., logging)
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Setup and Installation

1.  **Clone the Repository (if you haven't already):**

    ```bash
    git clone <repository_url>
    cd RAG-test
    ```

2.  **Create a Python Virtual Environment:**

    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**

    Install the required Python packages using `pip`:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models (Optional, but Recommended for Offline Use):**

    The application can download models on the fly, but for faster startup
    and offline functionality, you can pre-download them:

    ```bash
    python src/download_models.py
    ```

5.  **Prepare Your Documents:**

    Create a file named `docs.txt` in the `data/` directory.
    Each line in this file should contain a single document or text snippet.
    The RAG pipeline will build an index from these documents.

    Example `data/docs.txt`:
    ```
    Retrieval-Augmented Generation (RAG) is an AI framework for improving the specificity and factuality of responses from Large Language Models (LLMs).
    FAISS is a library for efficient similarity search and clustering of dense vectors.
    Sentence Transformers is a Python framework for state-of-art sentence, text and image embeddings.
    T5 (Text-to-Text Transfer Transformer) is a Transformer-based model that frames all NLP problems as a text-to-text problem.
    Embeddings are numerical representations of text that capture semantic meaning.
    ```

## How to Run

Once you have set up the environment and prepared your `docs.txt` file, you can run the RAG pipeline:

```bash
python src/main.py
```

Upon first run, if `data/faiss.index` does not exist, the script will automatically
build the FAISS index from `data/docs.txt`. It will then perform a sample query
and print the retrieved documents and the generated answer.

## Configuration

You can configure the RAG pipeline using environment variables:

-   `EMB_MODEL`: Path to a local embedding model or a Hugging Face model identifier.
    (e.g., `EMB_MODEL=/path/to/my/all-MiniLM-L6-v2` or `EMB_MODEL=sentence-transformers/all-MiniLM-L6-v2`)
-   `GEN_MODEL`: Path to a local generation model or a Hugging Face model identifier.
    (e.g., `GEN_MODEL=/path/to/my/t5-small` or `GEN_MODEL=google/flan-t5-small`)
-   `RETRIEVAL_K`: Number of documents to retrieve (default: 3).
-   `FETCH_K`: Number of retrieval candidates before filtering (default: 10).
-   `MIN_RELEVANCE_SCORE`: Minimum normalized score for keeping a chunk (default: 0.35).
-   `MAX_CONTEXT_CHARS`: Maximum total context size sent to the generator (default: 2500).
-   `CHUNK_SIZE_CHARS`: Target chunk size while building the index (default: 700).
-   `CHUNK_OVERLAP_CHARS`: Overlap between adjacent chunks (default: 120).
-   `CITATIONS_ENABLED`: Set to `true` to include citation ids in output when possible.
-   `STEP_BY_STEP_MODE`: Set to `true` to pause and explain each major stage.
-   `USE_LOCAL_ONLY`: Set to `true` to force loading models only from local paths.
-   `HYBRID_SEARCH_ENABLED`: Set to `true` to fuse dense+sparse retrieval signals (default: `true`).
-   `HYBRID_DENSE_WEIGHT`: Weight for dense FAISS relevance in fusion (default: `0.65`).
-   `HYBRID_SPARSE_WEIGHT`: Weight for sparse BM25-style relevance in fusion (default: `0.35`).
-   `QUERY_REWRITE_ENABLED`: Set to `true` to rewrite/expand query terms before retrieval (default: `true`).
-   `CONTEXT_COMPRESSION_ENABLED`: Set to `true` to compress context chunks query-aware (default: `true`).
-   `COMPRESSION_MAX_SENTENCES`: Sentences kept per chunk during compression (default: `2`).
-   `COMPRESSION_MAX_CHARS`: Max chars per compressed chunk (default: `420`).

Example of setting environment variables (for a single command):

```bash
EMB_MODEL=./models/all-MiniLM-L6-v2 GEN_MODEL=./models/t5-small python src/main.py
```

Teaching mode:

```bash
STEP_BY_STEP_MODE=true python src/main.py
```

Hybrid retrieval tuning example:

```bash
HYBRID_DENSE_WEIGHT=0.55 HYBRID_SPARSE_WEIGHT=0.45 python src/main.py
```

## Evaluation

Run dataset-based evaluation:

```bash
python src/eval_rag.py --dataset data/eval.jsonl --k 3
```

Expected JSONL fields per row:

```json
{"question":"What is RAG?","expected_answer":"Retrieval-Augmented Generation is ...","expected_chunk_ids":[2,7]}
```

Reported metrics include:
- Exact Match
- Token F1
- Retrieval Recall@k (when `expected_chunk_ids` is provided)
- Citation Precision (when `expected_chunk_ids` is provided)

## Troubleshooting

-   **SSL Certificate Errors (especially on macOS)**: The `config.py` module attempts to fix this by setting the `SSL_CERT_FILE` environment variable using `certifi`. Ensure `certifi` is installed (`pip install certifi`). If issues persist, refer to Python's SSL documentation.
-   **Model Download Errors**: If models fail to download, ensure you have an active internet connection or pre-download them using `python src/download_models.py`.
-   **`docs.txt` missing**: The `main.py` script will prompt you to create `data/docs.txt` if it's not found.
-   **`faiss.index` missing**: The index will be built automatically on the first run if it doesn't exist.

Feel free to explore and modify the code to experiment with different models, datasets, and RAG strategies!

## Web UI (React)

You can also ask questions from a browser UI.

1. Start API backend:

```bash
python3 src/web_api.py
```

2. Start UI:

```bash
cd Clients/UI
npm install
npm run dev
```

Then open `http://127.0.0.1:5173`.
