"""Hybrid BM25 + FAISS retriever with Reciprocal Rank Fusion.

Why hybrid search:
- FAISS (dense) captures semantic similarity — good for paraphrased questions.
- BM25 (sparse) captures exact keyword matches — good for proper nouns, figures,
  and Spanish-specific terms that embeddings may not encode well.
- Reciprocal Rank Fusion (RRF) combines both rankings without needing to
  normalize scores across different scales.

RRF formula:  score(d) = Σ  1 / (k + rank(d, system_i))
              where k=60 is the standard constant that dampens high-rank outliers.
"""

from dataclasses import dataclass

from financial_rag.embeddings.base import BaseEmbedder
from financial_rag.retrieval.base import BaseRetriever
from financial_rag.retrieval.models import RetrievalResult
from financial_rag.retrieval.vector_store import FAISSVectorStore, SearchResult

_RRF_K = 60  # standard RRF constant


@dataclass
class _BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi tied to the chunk list."""

    chunks: list  # list[Chunk] — parallel to the BM25 corpus
    _bm25: object = None

    def __post_init__(self) -> None:
        if not self.chunks:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(c.content) for c in self.chunks]
        self._bm25 = BM25Okapi(tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def query(self, text: str, top_k: int) -> list[tuple[int, float]]:
        """Return (chunk_index, bm25_score) for the top_k chunks."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(self._tokenize(text))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class HybridRetriever(BaseRetriever):
    """Two-stage retriever: BM25 + FAISS fused with Reciprocal Rank Fusion.

    Args:
        store: Loaded FAISSVectorStore (also the source of the BM25 corpus).
        embedder: Same model used to build the FAISS index.
        score_threshold: Minimum FAISS cosine similarity (applied before fusion).
        rrf_k: RRF constant (default 60 — standard value from the original paper).
    """

    def __init__(
        self,
        store: FAISSVectorStore,
        embedder: BaseEmbedder,
        score_threshold: float = 0.0,
        rrf_k: int = _RRF_K,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._score_threshold = score_threshold
        self._rrf_k = rrf_k
        self._bm25 = _BM25Index(chunks=store.chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
    ) -> RetrievalResult:
        candidate_k = top_k * 3

        # ── Dense (FAISS) retrieval ───────────────────────────────────────
        faiss_results: list[SearchResult] = self._store.search(
            query=query,
            embedder=self._embedder,
            top_k=candidate_k,
            score_threshold=self._score_threshold,
        )

        # ── Sparse (BM25) retrieval ───────────────────────────────────────
        bm25_hits = self._bm25.query(query, top_k=candidate_k)

        # ── Reciprocal Rank Fusion ────────────────────────────────────────
        rrf_scores: dict[int, float] = {}

        for rank, sr in enumerate(faiss_results):
            chunk_idx = sr.chunk.chunk_index
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (self._rrf_k + rank)

        for rank, (chunk_idx, _) in enumerate(bm25_hits):
            rrf_scores[chunk_idx] = rrf_scores.get(chunk_idx, 0.0) + 1.0 / (self._rrf_k + rank)

        # Build a lookup from chunk_index → SearchResult for already-fetched chunks
        faiss_lookup: dict[int, SearchResult] = {
            sr.chunk.chunk_index: sr for sr in faiss_results
        }
        bm25_lookup: dict[int, object] = {
            idx: self._bm25.chunks[idx] for idx, _ in bm25_hits
        }

        sorted_idx = sorted(rrf_scores, key=lambda i: rrf_scores[i], reverse=True)

        results: list[SearchResult] = []
        for new_rank, chunk_idx in enumerate(sorted_idx[:top_k]):
            if chunk_idx in faiss_lookup:
                sr = faiss_lookup[chunk_idx]
                chunk = sr.chunk
            else:
                chunk = bm25_lookup[chunk_idx]

            if source_filter and source_filter.lower() not in chunk.source.lower():
                continue

            results.append(SearchResult(chunk=chunk, score=rrf_scores[chunk_idx], rank=new_rank))

        # Re-number ranks after optional source filter
        for i, r in enumerate(results):
            r.rank = i

        return RetrievalResult(query=query, results=results[:top_k])

    @property
    def store_size(self) -> int:
        return self._store.size

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(store_size={self.store_size}, "
            f"embedder={type(self._embedder).__name__}, "
            f"rrf_k={self._rrf_k})"
        )
