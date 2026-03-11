"""FastAPI wrapper for the RAG pipeline."""

import argparse
import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import RAGPipeline, _build_index_if_needed, _validate_setup
from rag.feedback import FeedbackStore
from rag.utils import is_auth_error, setup_logging

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: List[int]
    retrieved_chunks: List[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    wrong_answer: str = Field(default="", max_length=4000)
    correct_answer: str = Field(..., min_length=1, max_length=4000)


class FeedbackResponse(BaseModel):
    status: str
    corrections_total: int
    message: str


app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline: RAGPipeline | None = None
feedback_store: FeedbackStore | None = None


@app.on_event("startup")
def startup_event() -> None:
    global pipeline, feedback_store

    # Avoid interactive pauses in API mode.
    os.environ.setdefault("STEP_BY_STEP_MODE", "false")

    try:
        pipeline = RAGPipeline()
    except Exception as exc:
        if is_auth_error(exc):
            logger.error(
                "Pipeline initialization failed due to an authentication error: %s. "
                "Set a valid HF_TOKEN in your .env file. "
                "Generate a new token at: https://huggingface.co/settings/tokens",
                exc,
            )
        raise

    if not _validate_setup(pipeline):
        raise RuntimeError("Invalid setup: docs.txt missing or empty")
    if not _build_index_if_needed(pipeline):
        raise RuntimeError("Failed to build/load index")

    feedback_store = FeedbackStore(pipeline.config.data_dir)


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
        detail = str(exc)
        # Provide an actionable hint for Hugging Face token errors.
        if is_auth_error(exc):
            detail = (
                f"{detail} — If you are seeing a token/authentication error, "
                "set a valid HF_TOKEN in your .env file. "
                "Generate a new token at: https://huggingface.co/settings/tokens"
            )
        raise HTTPException(status_code=500, detail=detail) from exc


@app.post("/api/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Accept a user correction and teach the RAG system.

    The correct answer is added to the retrieval knowledge base and the FAISS
    index is rebuilt in-place so that future queries benefit immediately.
    This is the RAG equivalent of "learning from mistakes" — no model
    fine-tuning required.
    """
    if pipeline is None or feedback_store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        feedback_store.record(
            question=payload.question,
            wrong_answer=payload.wrong_answer,
            correct_answer=payload.correct_answer,
        )
        # Rebuild the index so the correction is immediately searchable.
        pipeline.build_index()
        total = feedback_store.correction_count()
        logger.info("[Feedback] index rebuilt after correction; total corrections=%d", total)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Feedback submission failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return FeedbackResponse(
        status="ok",
        corrections_total=total,
        message=(
            "Thank you! The correction has been added to the knowledge base. "
            "The system will use it to answer similar questions better in future."
        ),
    )


@app.get("/api/feedback/list")
def list_feedback() -> Dict[str, Any]:
    """Return all recorded corrections (what the system has learned)."""
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    corrections = feedback_store.list_corrections()
    return {"total": len(corrections), "corrections": corrections}


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
