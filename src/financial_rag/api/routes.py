"""API route handlers."""

from fastapi import APIRouter, HTTPException, Request

from financial_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ModelsResponse,
)
from financial_rag.generation.ollama import AVAILABLE_MODELS

router = APIRouter()


def _get_pipeline(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    return pipeline


@router.post("/ask", response_model=AskResponse)
def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question using the RAG pipeline."""
    pipeline = _get_pipeline(request)
    response = pipeline.ask(
        question=body.question,
        top_k=body.top_k,
        source_filter=body.source_filter,
    )
    return AskResponse(
        answer=response.answer,
        query=response.query,
        citations=response.citations,
        retrieval_scores=response.retrieval_scores,
        chunks_used=response.chunks_used,
        model=response.model,
        retrieval_ms=response.retrieval_ms,
        generation_ms=response.generation_ms,
        total_ms=response.total_ms,
        is_grounded=response.is_grounded,
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return pipeline status and store statistics."""
    pipeline = _get_pipeline(request)
    store_chunks = pipeline._retriever.store_size
    model = type(pipeline._generator).__name__
    return HealthResponse(
        status="ok",
        model=model,
        store_chunks=store_chunks,
        store_path=str(request.app.state.store_path),
    )


@router.get("/models", response_model=ModelsResponse)
def models(request: Request) -> ModelsResponse:
    """List available Ollama models and the currently active one."""
    pipeline = _get_pipeline(request)
    current = getattr(pipeline._generator, "_model", "unknown")
    return ModelsResponse(available=AVAILABLE_MODELS, current=current)
