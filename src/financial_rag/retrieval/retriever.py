"""Vector similarity retriever backed by FAISSVectorStore."""

from financial_rag.embeddings.base import BaseEmbedder
from financial_rag.retrieval.base import BaseRetriever
from financial_rag.retrieval.models import RetrievalResult
from financial_rag.retrieval.vector_store import FAISSVectorStore, SearchResult


class VectorRetriever(BaseRetriever):
    """Retrieves chunks from a FAISS vector store using cosine similarity.

    Args:
        store: Pre-built or loaded FAISSVectorStore.
        embedder: Must be the same model used to build the store.
        score_threshold: Minimum cosine similarity to include a result.
    """

    def __init__(
        self,
        store: FAISSVectorStore,
        embedder: BaseEmbedder,
        score_threshold: float = 0.0,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        """Find the most relevant chunks for a query.

        Args:
            query: Natural language question.
            top_k: Maximum number of results to return.
            source_filter: If set, only return chunks whose source path contains
                this string (case-insensitive). Useful for restricting to a
                specific bank: "interbank", "scotiabank", etc.

        Returns:
            RetrievalResult with ranked chunks.
        """
        # Over-fetch when filtering so we still get top_k after dropping results.
        fetch_k = top_k * 3 if source_filter else top_k

        raw: list[SearchResult] = self._store.search(
            query=query,
            embedder=self._embedder,
            top_k=fetch_k,
            score_threshold=self._score_threshold,
        )

        if source_filter:
            raw = [
                r for r in raw
                if source_filter.lower() in r.chunk.source.lower()
            ]

        filtered = raw[:top_k]

        # Re-number ranks after filtering (gaps would confuse downstream code).
        for new_rank, result in enumerate(filtered):
            result.rank = new_rank

        return RetrievalResult(query=query, results=filtered)

    @property
    def store_size(self) -> int:
        return self._store.size

    def __repr__(self) -> str:
        return (
            f"VectorRetriever(store_size={self.store_size}, "
            f"embedder={type(self._embedder).__name__}, "
            f"score_threshold={self._score_threshold})"
        )
