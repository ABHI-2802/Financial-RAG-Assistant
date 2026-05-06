"""Pydantic schemas for API request and response bodies."""

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")
    source_filter: str | None = Field(
        default=None,
        description="Restrict retrieval to sources containing this string (e.g. 'interbank').",
    )
    model: str | None = Field(
        default=None,
        description="Ollama model tag to use for this request. Defaults to the pipeline's startup model.",
    )
    history: list[ConversationMessage] = Field(
        default_factory=list,
        description="Prior conversation turns to inject for multi-turn support.",
    )

    model_config = {"json_schema_extra": {"example": {
        "question": "¿Y cómo compara eso con Scotiabank?",
        "top_k": 5,
        "source_filter": None,
        "model": "qwen3:14b",
        "history": [
            {"role": "user", "content": "¿Cuál fue la utilidad neta de Interbank en 2024?"},
            {"role": "assistant", "content": "La utilidad neta de Interbank fue S/ 1,234M [1]."},
        ],
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
