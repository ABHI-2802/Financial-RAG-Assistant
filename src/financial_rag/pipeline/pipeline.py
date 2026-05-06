"""RAG pipeline — chains retriever and generator into a single ask() call."""

import time

from financial_rag.generation.base import BaseGenerator
from financial_rag.pipeline.models import RAGResponse
from financial_rag.retrieval.base import BaseRetriever
from financial_rag.retrieval.reranker import BaseReranker


class RAGPipeline:
    """End-to-end RAG pipeline: retrieve relevant chunks, then generate an answer.

    This is the single entry point for the application layer (API, UI).
    All latency measurement, logging hooks, and future caching live here.

    Args:
        retriever: Any BaseRetriever implementation.
        generator: Any BaseGenerator implementation.
        top_k: Default number of chunks to pass to the generator.
        reranker: Optional cross-encoder reranker. When provided, retrieves
            top_k * RERANK_FACTOR candidates from FAISS and reranks to top_k.
    """

    RERANK_FACTOR = 3  # retrieve 3× candidates, rerank down to top_k

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        top_k: int = 5,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._retriever = retriever
        self._generator = generator
        self._top_k = top_k
        self._reranker = reranker

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> RAGResponse:
        """Answer a question using retrieved context.

        Args:
            question: Natural language question.
            top_k: Override the default number of chunks to retrieve.
            source_filter: Restrict retrieval to a specific source
                (e.g. "interbank", "scotiabank"). Passed to the retriever
                if it supports it — silently ignored otherwise.

        Returns:
            RAGResponse with the answer, citations, scores, and latency breakdown.
        """
        k = top_k if top_k is not None else self._top_k

        # ── Retrieval ──────────────────────────────────────────────────────
        t0 = time.perf_counter()
        candidate_k = k * self.RERANK_FACTOR if self._reranker else k
        retrieve_kwargs: dict = {"top_k": candidate_k}
        if source_filter is not None:
            retrieve_kwargs["source_filter"] = source_filter
        retrieval = self._retriever.retrieve(question, **retrieve_kwargs)

        if self._reranker and not retrieval.is_empty:
            reranked = self._reranker.rerank(question, retrieval.results, top_k=k)
            retrieval = type(retrieval)(query=retrieval.query, results=reranked)

        retrieval_ms = (time.perf_counter() - t0) * 1000

        # ── Generation ─────────────────────────────────────────────────────
        t1 = time.perf_counter()
        generation = self._generator.generate(retrieval)
        generation_ms = (time.perf_counter() - t1) * 1000

        return RAGResponse(
            answer=generation.answer,
            query=question,
            citations=generation.citations,
            retrieval_scores=[r.score for r in retrieval.results],
            chunks_used=retrieval.total,
            model=generation.model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            chunk_texts=[r.chunk.content for r in retrieval.results],
        )

    def __repr__(self) -> str:
        return (
            f"RAGPipeline("
            f"retriever={type(self._retriever).__name__}, "
            f"generator={type(self._generator).__name__}, "
            f"reranker={type(self._reranker).__name__ if self._reranker else None}, "
            f"top_k={self._top_k})"
        )
