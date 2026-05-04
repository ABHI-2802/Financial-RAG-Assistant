"""Pydantic schemas for API request and response bodies."""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    source_filter: str | None = Field(
        default=None,
        description="Restrict retrieval to sources containing this string (e.g. 'interbank').",
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "¿Cuál fue la utilidad neta de Interbank en 2024?",
        "top_k": 5,
        "source_filter": "interbank",
    }}}


class CitationItem(BaseModel):
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    query: str
    citations: list[str]
    retrieval_scores: list[float]
    chunks_used: int
    model: str
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    is_grounded: bool


class HealthResponse(BaseModel):
    status: str
    model: str
    store_chunks: int
    store_path: str


class ModelsResponse(BaseModel):
    available: list[str]
    current: str
