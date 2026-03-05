# RAG UI (React)

This UI lets you ask questions from browser and receive answers from the Python RAG backend.

## 1) Start backend API

From project root:

```bash
python3 src/web_api.py
```

API runs at `http://127.0.0.1:8000`.

## 2) Start UI

In another terminal:

```bash
cd Clients/UI
npm install
npm run dev
```

UI runs at `http://127.0.0.1:5173`.

## 3) Optional API URL override

Create `.env` in `Clients/UI`:

```bash
VITE_API_BASE=http://127.0.0.1:8000
```

## 4) Debug backend logs

```bash
python3 src/web_api.py --debug
```
