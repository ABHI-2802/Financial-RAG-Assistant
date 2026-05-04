"""API route handlers."""

import json
import time
from collections.abc import Generator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from financial_rag.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    ModelsResponse,
)
from financial_rag.generation.ollama import AVAILABLE_MODELS, _SYSTEM_PROMPT

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


@router.post("/ask/stream")
def ask_stream(body: AskRequest, request: Request) -> StreamingResponse:
    """Stream answer tokens via Server-Sent Events."""
    pipeline = _get_pipeline(request)
    generator = pipeline._generator
    retriever = pipeline._retriever

    # Retrieve synchronously, then stream generation
    t0 = time.perf_counter()
    retrieval = retriever.retrieve(
        body.question,
        top_k=body.top_k,
        **({"source_filter": body.source_filter} if body.source_filter else {}),
    )
    retrieval_ms = (time.perf_counter() - t0) * 1000

    def event_stream() -> Generator[str, None, None]:
        import ollama
        import re

        if retrieval.is_empty:
            no_ctx = "No encontré información relevante en los documentos."
            yield f"data: {json.dumps({'token': no_ctx})}\n\n"
            yield f"data: {json.dumps({'done': True, 'answer': no_ctx, 'query': body.question, 'citations': [], 'retrieval_scores': [], 'chunks_used': 0, 'model': getattr(generator, '_model', 'unknown'), 'retrieval_ms': retrieval_ms, 'generation_ms': 0, 'total_ms': retrieval_ms, 'is_grounded': False})}\n\n"
            return

        # Format context
        lines = ["Contexto:"]
        for i, r in enumerate(retrieval.results, start=1):
            lines.append(f"[{i}] {r.citation}")
            lines.append(r.chunk.content.strip())
            lines.append("")
        context_text = "\n".join(lines)

        model_name = getattr(generator, "_model", "qwen3:8b")
        think = getattr(generator, "_think", False)

        t1 = time.perf_counter()
        full_text = ""
        in_think = False

        stream = ollama.chat(
            model=model_name,
            think=think,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"{context_text}\n\nPregunta: {body.question}"},
            ],
            stream=True,
        )

        for chunk in stream:
            token = chunk.message.content or ""
            full_text += token

            # Skip <think> blocks in streamed output
            if "<think>" in full_text:
                in_think = True
            if "</think>" in full_text:
                in_think = False
                full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
                continue
            if in_think:
                continue

            if token:
                yield f"data: {json.dumps({'token': token})}\n\n"

        generation_ms = (time.perf_counter() - t1) * 1000
        clean_answer = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()

        done_payload = {
            "done": True,
            "answer": clean_answer,
            "query": body.question,
            "citations": retrieval.citations,
            "retrieval_scores": [r.score for r in retrieval.results],
            "chunks_used": retrieval.total,
            "model": model_name,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": retrieval_ms + generation_ms,
            "is_grounded": retrieval.total > 0,
        }
        yield f"data: {json.dumps(done_payload)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
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
