"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from financial_rag.api.routes import router

_DEFAULT_STORE = "data/processed/vector_store"
_DEFAULT_MODEL = "qwen3:8b"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the RAG pipeline once at startup; tear down on shutdown.

    If app.state.pipeline is already set (e.g. injected in tests),
    skip creation so tests can supply their own mock.
    """
    if app.state.pipeline is None:
        store_path = getattr(app.state, "store_path", _DEFAULT_STORE)
        model = getattr(app.state, "model", _DEFAULT_MODEL)
        from financial_rag.pipeline.factory import create_pipeline
        app.state.pipeline = create_pipeline(store_path=store_path, model=model)
        print(f"  [api] Pipeline ready — model={model}, store={store_path}")
    yield
    app.state.pipeline = None
    print("  [api] Pipeline shut down.")


def create_app(
    store_path: str = _DEFAULT_STORE,
    model: str = _DEFAULT_MODEL,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        store_path: Path to the FAISS index (without extension).
        model: Ollama model tag for generation.

    Returns:
        Configured FastAPI app ready to serve.
    """
    app = FastAPI(
        title="Financial RAG Assistant",
        description="RAG API for Peruvian bank annual reports.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Store config so lifespan can read it
    app.state.store_path = store_path
    app.state.model = model
    app.state.pipeline = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
