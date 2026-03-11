"""FastAPI wrapper for the RAG pipeline."""

import argparse
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import RAGPipeline, _build_index_if_needed, _validate_setup
from rag.utils import setup_logging

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)
    guided: bool = Field(default=False)


class PipelineStep(BaseModel):
    step: str
    description: str
    chunks: Optional[List[Dict[str, Any]]] = None


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[int]
    retrieved_chunks: List[Dict[str, Any]]
    pipeline_steps: List[PipelineStep] = []


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1)
    rating: int = Field(..., description="1 = helpful, -1 = not helpful")
    chat_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: int
    status: str


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
        result = pipeline.run(question, k=k, guided=payload.guided)
        return AskResponse(
            query=result["query"],
            answer=result["answer"],
            citations=result["citations"],
            retrieved_chunks=result["retrieved_chunks"],
            pipeline_steps=[PipelineStep(**s) for s in result.get("pipeline_steps", [])],
        )
    except Exception as exc:
        logger.exception("Ask request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if payload.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (helpful) or -1 (not helpful)")

    try:
        fid = pipeline.chat_store.store_feedback(
            question=payload.question,
            answer=payload.answer,
            rating=payload.rating,
            chat_id=payload.chat_id,
        )
        return FeedbackResponse(id=fid, status="recorded")
    except Exception as exc:
        logger.exception("Feedback request failed")
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
