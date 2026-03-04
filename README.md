# RAG-test

A small RAG (Retrieval-Augmented Generation) testing app using FAISS, sentence-transformers, and T5.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your settings.  
Most models used here are **public** and work without a token, so you can leave `HF_TOKEN` empty for normal use.

> **Getting a "token expired or invalid: 403" error?**  
> This means your `HF_TOKEN` in `.env` is expired or set to a bad value.  
> Public models (all-MiniLM-L6-v2, t5-small) do **not** need a token — set  
> `HF_TOKEN=` (leave it empty) and the app will connect anonymously.  
> For private/gated models, generate a fresh token at  
> <https://huggingface.co/settings/tokens> and paste it in.

### 3. Download models (optional – for offline use)

```bash
python src/download_models.py
```

This saves `all-MiniLM-L6-v2` and `t5-small` under `models/` so the app works completely offline.

### 4. Run

```bash
# Full RAG pipeline (requires model downloads or internet access)
python src/main.py

# Robust version with automatic fallback
python src/main_robust.py

# Simplified offline mode (no model downloads needed)
python src/main_simplified.py
```

## Troubleshooting

| Error | Fix |
|-------|-----|
| `token expired or invalid: 403` | Set `HF_TOKEN=` (empty) in `.env` for public models, or refresh the token at <https://huggingface.co/settings/tokens> for private/gated models (see step 2) |
| SSL errors on macOS | `pip install certifi` – already handled automatically |
| Model not found locally | Run `python src/download_models.py` |
