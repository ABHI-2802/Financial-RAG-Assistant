"""Unit tests for generation models and MockGenerator."""

import pytest

from financial_rag.chunking.models import Chunk
from financial_rag.embeddings.mock import MockEmbedder
from financial_rag.generation.factory import create_generator
from financial_rag.generation.mock import MockGenerator
from financial_rag.generation.models import GenerationResult
from financial_rag.generation.ollama import AVAILABLE_MODELS
from financial_rag.ingestion.models import Document
from financial_rag.retrieval.models import RetrievalResult
from financial_rag.retrieval.vector_store import FAISSVectorStore
from financial_rag.retrieval.retriever import VectorRetriever


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_retrieval_result(
    query: str = "¿Cuál fue la utilidad neta?",
    texts: list[str] | None = None,
) -> RetrievalResult:
    texts = texts or ["La utilidad neta de Interbank fue S/ 1,200 millones.", "El ROE fue 18%."]
    embedder = MockEmbedder(dim=16)
    store = FAISSVectorStore(dimension=16)
    chunks = [
        Chunk.from_document(
            Document(content=t, source=f"interbank.pdf", page=i),
            content=t, chunk_index=i, total_chunks=len(texts),
        )
        for i, t in enumerate(texts)
    ]
    store.add_chunks(chunks, embedder)
    retriever = VectorRetriever(store=store, embedder=embedder, score_threshold=-1.0)
    return retriever.retrieve(query, top_k=len(texts))


def empty_retrieval_result(query: str = "pregunta sin respuesta") -> RetrievalResult:
    return RetrievalResult(query=query, results=[])


# ── GenerationResult ──────────────────────────────────────────────────────────

class TestGenerationResult:
    def test_total_tokens(self):
        result = GenerationResult(
            answer="Respuesta.", query="pregunta", citations=[],
            model="mock", input_tokens=100, output_tokens=50,
        )
        assert result.total_tokens == 150

    def test_total_tokens_defaults_zero(self):
        result = GenerationResult(answer="R", query="Q", citations=[], model="mock")
        assert result.total_tokens == 0

    def test_repr_contains_model(self):
        result = GenerationResult(answer="R", query="Q", citations=["c1"], model="qwen3:8b")
        assert "qwen3:8b" in repr(result)

    def test_repr_contains_citation_count(self):
        result = GenerationResult(
            answer="R", query="Q", citations=["c1", "c2"], model="mock"
        )
        assert "2" in repr(result)


# ── MockGenerator ─────────────────────────────────────────────────────────────

class TestMockGenerator:
    def test_returns_generation_result(self):
        gen = MockGenerator()
        result = gen.generate(make_retrieval_result())
        assert isinstance(result, GenerationResult)

    def test_captures_query(self):
        gen = MockGenerator()
        rr = make_retrieval_result(query="¿Cuál fue el ROE?")
        result = gen.generate(rr)
        assert result.query == "¿Cuál fue el ROE?"

    def test_captures_citations(self):
        gen = MockGenerator()
        rr = make_retrieval_result()
        result = gen.generate(rr)
        assert result.citations == rr.citations

    def test_custom_answer(self):
        gen = MockGenerator(answer="Respuesta personalizada.")
        result = gen.generate(make_retrieval_result())
        assert result.answer == "Respuesta personalizada."

    def test_model_is_mock(self):
        gen = MockGenerator()
        result = gen.generate(make_retrieval_result())
        assert result.model == "mock"

    def test_empty_retrieval_result(self):
        gen = MockGenerator()
        result = gen.generate(empty_retrieval_result())
        assert result.citations == []
        assert isinstance(result.answer, str)

    def test_token_counts_are_positive(self):
        gen = MockGenerator()
        result = gen.generate(make_retrieval_result())
        assert result.input_tokens >= 0
        assert result.output_tokens >= 0

    def test_total_tokens_consistent(self):
        gen = MockGenerator()
        result = gen.generate(make_retrieval_result())
        assert result.total_tokens == result.input_tokens + result.output_tokens


# ── Factory ───────────────────────────────────────────────────────────────────

class TestFactory:
    def test_create_generator_returns_ollama_generator(self):
        from financial_rag.generation.ollama import OllamaGenerator
        gen = create_generator(model="qwen3:8b")
        assert isinstance(gen, OllamaGenerator)

    def test_create_generator_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_generator(model="gpt-99:fake")

    def test_all_known_models_accepted(self):
        for model in AVAILABLE_MODELS:
            gen = create_generator(model=model)
            assert gen is not None
