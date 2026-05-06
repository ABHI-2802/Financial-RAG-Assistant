"""Unit tests for the re-ranker components and pipeline integration."""

import pytest

from financial_rag.chunking.splitter import Chunk
from financial_rag.retrieval.reranker import MockReranker
from financial_rag.retrieval.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(rank: int, score: float, content: str = "text") -> SearchResult:
    chunk = Chunk(
        content=content,
        source="test.pdf",
        page=rank,
        chunk_index=rank,
        total_chunks=5,
        metadata={},
    )
    return SearchResult(chunk=chunk, score=score, rank=rank)


# ── MockReranker ──────────────────────────────────────────────────────────────

class TestMockReranker:
    def test_returns_top_k(self):
        results = [_make_result(i, 0.9 - i * 0.1) for i in range(5)]
        reranker = MockReranker()
        out = reranker.rerank("query", results, top_k=3)
        assert len(out) == 3

    def test_preserves_order(self):
        results = [_make_result(i, 0.9 - i * 0.1, content=f"chunk {i}") for i in range(4)]
        reranker = MockReranker()
        out = reranker.rerank("query", results, top_k=3)
        assert [r.chunk.content for r in out] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_empty_input_returns_empty(self):
        reranker = MockReranker()
        assert reranker.rerank("query", [], top_k=5) == []

    def test_top_k_larger_than_results(self):
        results = [_make_result(i, 0.5) for i in range(2)]
        reranker = MockReranker()
        out = reranker.rerank("query", results, top_k=10)
        assert len(out) == 2

    def test_ranks_updated(self):
        results = [_make_result(i, 0.5) for i in range(3)]
        reranker = MockReranker()
        out = reranker.rerank("query", results, top_k=3)
        assert [r.rank for r in out] == [0, 1, 2]


# ── Pipeline with reranker ────────────────────────────────────────────────────

class TestPipelineWithReranker:
    def test_pipeline_repr_includes_reranker(self):
        from financial_rag.generation.mock import MockGenerator
        from financial_rag.retrieval.retriever import VectorRetriever
        from financial_rag.pipeline.pipeline import RAGPipeline
        from unittest.mock import MagicMock

        retriever = MagicMock(spec=VectorRetriever)
        generator = MockGenerator()
        reranker = MockReranker()
        pipeline = RAGPipeline(retriever=retriever, generator=generator, reranker=reranker)

        assert "MockReranker" in repr(pipeline)

    def test_pipeline_without_reranker_repr(self):
        from financial_rag.generation.mock import MockGenerator
        from financial_rag.pipeline.pipeline import RAGPipeline
        from unittest.mock import MagicMock
        from financial_rag.retrieval.retriever import VectorRetriever

        retriever = MagicMock(spec=VectorRetriever)
        pipeline = RAGPipeline(retriever=retriever, generator=MockGenerator())

        assert "None" in repr(pipeline)

    def test_reranker_receives_correct_candidate_count(self):
        """Pipeline fetches top_k * RERANK_FACTOR candidates before reranking."""
        from financial_rag.generation.mock import MockGenerator
        from financial_rag.pipeline.pipeline import RAGPipeline
        from financial_rag.retrieval.models import RetrievalResult
        from unittest.mock import MagicMock, patch

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = RetrievalResult(query="q", results=[])

        mock_reranker = MagicMock(spec=MockReranker)
        mock_reranker.rerank.return_value = []

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            generator=MockGenerator(),
            top_k=5,
            reranker=mock_reranker,
        )
        pipeline.ask("¿Cuántos empleados?")

        called_top_k = mock_retriever.retrieve.call_args[1]["top_k"]
        assert called_top_k == 5 * RAGPipeline.RERANK_FACTOR


# ── BenchmarkConfig with reranker ─────────────────────────────────────────────

class TestBenchmarkConfigReranker:
    def test_use_reranker_defaults_false(self):
        from financial_rag.evaluation.runner import BenchmarkConfig
        cfg = BenchmarkConfig(model="qwen3:14b", top_k=5)
        assert cfg.use_reranker is False

    def test_use_reranker_can_be_set(self):
        from financial_rag.evaluation.runner import BenchmarkConfig
        cfg = BenchmarkConfig(model="qwen3:14b", top_k=5, use_reranker=True)
        assert cfg.use_reranker is True
