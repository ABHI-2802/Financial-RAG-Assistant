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

    def test_create_generator_claude_backend(self):
        from unittest.mock import MagicMock, patch
        from financial_rag.generation.claude import ClaudeGenerator
        with patch("financial_rag.generation.claude._anthropic_lib") as mock_lib:
            mock_lib.Anthropic.return_value = MagicMock()
            gen = create_generator(model="claude-sonnet-4-6", backend="claude")
        assert isinstance(gen, ClaudeGenerator)

    def test_create_generator_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_generator(model="qwen3:8b", backend="openai")

    def test_create_generator_unknown_claude_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_generator(model="gpt-4o", backend="claude")


# ── ClaudeGenerator ───────────────────────────────────────────────────────────

class TestClaudeGenerator:
    def _make_generator(self, answer: str = "Respuesta de prueba."):
        from unittest.mock import MagicMock
        from financial_rag.generation.claude import ClaudeGenerator

        mock_client = MagicMock()

        # generate() path
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=answer)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_client.messages.create.return_value = mock_response

        # stream() path — context manager yields text tokens
        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_stream_cm)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)
        mock_stream_cm.text_stream = iter([answer])
        mock_client.messages.stream.return_value = mock_stream_cm

        return ClaudeGenerator(_client=mock_client)

    def test_returns_generation_result(self):
        gen = self._make_generator()
        result = gen.generate(make_retrieval_result())
        assert isinstance(result, GenerationResult)

    def test_answer_populated(self):
        gen = self._make_generator(answer="Utilidad neta fue S/ 1,200 millones.")
        result = gen.generate(make_retrieval_result())
        assert result.answer == "Utilidad neta fue S/ 1,200 millones."

    def test_query_captured(self):
        gen = self._make_generator()
        rr = make_retrieval_result(query="¿Cuál fue el ROE?")
        result = gen.generate(rr)
        assert result.query == "¿Cuál fue el ROE?"

    def test_citations_captured(self):
        gen = self._make_generator()
        rr = make_retrieval_result()
        result = gen.generate(rr)
        assert result.citations == rr.citations

    def test_token_counts(self):
        gen = self._make_generator()
        result = gen.generate(make_retrieval_result())
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150

    def test_empty_retrieval_returns_no_context_answer(self):
        from financial_rag.generation.claude import _NO_CONTEXT_ANSWER
        gen = self._make_generator()
        result = gen.generate(empty_retrieval_result())
        assert result.answer == _NO_CONTEXT_ANSWER
        assert result.citations == []

    def test_empty_retrieval_skips_api_call(self):
        from unittest.mock import MagicMock
        from financial_rag.generation.claude import ClaudeGenerator
        mock_client = MagicMock()
        gen = ClaudeGenerator(_client=mock_client)
        gen.generate(empty_retrieval_result())
        mock_client.messages.create.assert_not_called()

    def test_stream_yields_tokens(self):
        gen = self._make_generator(answer="token de respuesta")
        tokens = list(gen.stream(make_retrieval_result()))
        assert len(tokens) > 0
        assert "token de respuesta" in "".join(tokens)

    def test_stream_empty_yields_no_context(self):
        from financial_rag.generation.claude import _NO_CONTEXT_ANSWER
        gen = self._make_generator()
        tokens = list(gen.stream(empty_retrieval_result()))
        assert tokens == [_NO_CONTEXT_ANSWER]

    def test_stream_empty_skips_api_call(self):
        from unittest.mock import MagicMock
        from financial_rag.generation.claude import ClaudeGenerator
        mock_client = MagicMock()
        gen = ClaudeGenerator(_client=mock_client)
        list(gen.stream(empty_retrieval_result()))
        mock_client.messages.stream.assert_not_called()

    def test_repr_contains_class_and_model(self):
        gen = self._make_generator()
        r = repr(gen)
        assert "ClaudeGenerator" in r
        assert "claude-sonnet-4-6" in r

    def test_history_injected_into_messages(self):
        from unittest.mock import MagicMock, call
        from financial_rag.generation.claude import ClaudeGenerator
        from financial_rag.generation.models import ConversationTurn
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        gen = ClaudeGenerator(_client=mock_client)
        history = [
            ConversationTurn(role="user", content="pregunta anterior"),
            ConversationTurn(role="assistant", content="respuesta anterior"),
        ]
        gen.generate(make_retrieval_result(), history=history)

        _, kwargs = mock_client.messages.create.call_args
        roles = [m["role"] for m in kwargs["messages"]]
        assert roles[0] == "user"
        assert roles[1] == "assistant"
        assert roles[-1] == "user"  # current question last
