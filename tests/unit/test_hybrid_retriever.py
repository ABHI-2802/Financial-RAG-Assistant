"""Unit tests for the HybridRetriever (BM25 + FAISS with RRF)."""

import pytest
from unittest.mock import MagicMock

from financial_rag.chunking.models import Chunk
from financial_rag.retrieval.hybrid import HybridRetriever, _BM25Index
from financial_rag.retrieval.vector_store import SearchResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(idx: int, content: str, source: str = "test.pdf") -> Chunk:
    return Chunk(
        content=content,
        source=source,
        page=idx,
        chunk_index=idx,
        total_chunks=10,
        metadata={},
    )


def _make_search_result(chunk: Chunk, score: float, rank: int) -> SearchResult:
    return SearchResult(chunk=chunk, score=score, rank=rank)


def _make_mock_store(chunks: list[Chunk]) -> MagicMock:
    store = MagicMock()
    store.chunks = chunks
    store.size = len(chunks)
    store.search.return_value = [
        _make_search_result(c, score=0.9 - i * 0.1, rank=i)
        for i, c in enumerate(chunks)
    ]
    return store


# ── _BM25Index ────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_returns_top_k_results(self):
        chunks = [_make_chunk(i, f"utilidad neta interbank {i}") for i in range(10)]
        idx = _BM25Index(chunks=chunks)
        results = idx.query("utilidad neta", top_k=3)
        assert len(results) == 3

    def test_scores_are_sorted_descending(self):
        chunks = [_make_chunk(i, f"utilidad neta interbank {i}") for i in range(5)]
        idx = _BM25Index(chunks=chunks)
        results = idx.query("utilidad", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_ranks_higher(self):
        chunks = [
            _make_chunk(0, "empleados de interbank 2024"),
            _make_chunk(1, "política de dividendos scotiabank"),
            _make_chunk(2, "número de empleados en el banco"),
        ]
        idx = _BM25Index(chunks=chunks)
        results = idx.query("empleados interbank", top_k=3)
        top_chunk_idx = results[0][0]
        assert top_chunk_idx == 0  # exact match chunk ranks first

    def test_empty_corpus(self):
        idx = _BM25Index(chunks=[])
        results = idx.query("cualquier cosa", top_k=5)
        assert results == []


# ── HybridRetriever ───────────────────────────────────────────────────────────

class TestHybridRetriever:
    def _make_retriever(self, chunks: list[Chunk]) -> HybridRetriever:
        store = _make_mock_store(chunks)
        embedder = MagicMock()
        return HybridRetriever(store=store, embedder=embedder)

    def test_returns_retrieval_result(self):
        from financial_rag.retrieval.models import RetrievalResult
        chunks = [_make_chunk(i, f"texto del chunk {i}") for i in range(5)]
        retriever = self._make_retriever(chunks)
        result = retriever.retrieve("query", top_k=3)
        assert isinstance(result, RetrievalResult)

    def test_returns_at_most_top_k(self):
        chunks = [_make_chunk(i, f"chunk {i}") for i in range(10)]
        retriever = self._make_retriever(chunks)
        result = retriever.retrieve("query", top_k=4)
        assert len(result.results) <= 4

    def test_ranks_are_sequential(self):
        chunks = [_make_chunk(i, f"chunk {i}") for i in range(5)]
        retriever = self._make_retriever(chunks)
        result = retriever.retrieve("query", top_k=3)
        assert [r.rank for r in result.results] == list(range(len(result.results)))

    def test_source_filter_applied(self):
        chunks = [
            _make_chunk(0, "datos interbank", source="interbank.pdf"),
            _make_chunk(1, "datos scotiabank", source="scotiabank.pdf"),
            _make_chunk(2, "más interbank", source="interbank.pdf"),
        ]
        store = _make_mock_store(chunks)
        store.search.return_value = [
            _make_search_result(c, score=0.9 - i * 0.1, rank=i)
            for i, c in enumerate(chunks)
        ]
        retriever = HybridRetriever(store=store, embedder=MagicMock())
        result = retriever.retrieve("query", top_k=5, source_filter="scotiabank")
        assert all("scotiabank" in r.chunk.source.lower() for r in result.results)

    def test_repr(self):
        chunks = [_make_chunk(0, "texto")]
        retriever = self._make_retriever(chunks)
        assert "HybridRetriever" in repr(retriever)
        assert "rrf_k" in repr(retriever)

    def test_store_size_property(self):
        chunks = [_make_chunk(i, f"chunk {i}") for i in range(7)]
        retriever = self._make_retriever(chunks)
        assert retriever.store_size == 7
