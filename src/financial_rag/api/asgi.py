"""ASGI entry point for production: uvicorn financial_rag.api.asgi:app

Bridges the application factory with environment-driven config so that
Docker / uvicorn can reference a stable module-level object.
"""

from financial_rag.api.app import create_app
from financial_rag.config import settings

app = create_app(
    store_path=settings.vector_store_path,
    model=settings.ollama_model,
)
