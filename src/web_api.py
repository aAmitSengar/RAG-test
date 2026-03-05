"""FastAPI wrapper for the RAG pipeline."""

import argparse
import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import RAGPipeline, _build_index_if_needed, _validate_setup
from rag.utils import setup_logging

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[int]
    retrieved_chunks: List[Dict[str, Any]]


app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline: RAGPipeline | None = None


@app.on_event("startup")
def startup_event() -> None:
    global pipeline

    # Avoid interactive pauses in API mode.
    os.environ.setdefault("STEP_BY_STEP_MODE", "false")

    pipeline = RAGPipeline()
    if not _validate_setup(pipeline):
        raise RuntimeError("Invalid setup: docs.txt missing or empty")
    if not _build_index_if_needed(pipeline):
        raise RuntimeError("Failed to build/load index")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        k = payload.k if payload.k is not None else pipeline.config.retrieval_k
        result = pipeline.run(question, k=k)
        return AskResponse(**result)
    except Exception as exc:
        logger.exception("Ask request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logs")
    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.debug else logging.ERROR)

    import uvicorn

    uvicorn.run("web_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
