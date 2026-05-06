"""Factory to build a VectorRetriever from a saved index."""

from pathlib import Path

from financial_rag.embeddings.base import BaseEmbedder
from financial_rag.retrieval.base import BaseRetriever
from financial_rag.retrieval.retriever import VectorRetriever
from financial_rag.retrieval.vector_store import FAISSVectorStore


def create_retriever(
    store_path: str | Path,
    embedder: BaseEmbedder | None = None,
    score_threshold: float = 0.0,
    use_hybrid: bool = False,
) -> BaseRetriever:
    """Load a saved FAISS index and wrap it in a retriever.

    Args:
        store_path: Base path to the FAISS index (without extension).
        embedder: Pre-built embedder instance. Defaults to all-MiniLM-L6-v2.
        score_threshold: Minimum cosine similarity for a result (FAISS only).
        use_hybrid: If True, returns a HybridRetriever (BM25 + FAISS with RRF).

    Returns:
        Ready-to-query retriever (VectorRetriever or HybridRetriever).
    """
    store = FAISSVectorStore.load(store_path)

    if embedder is None:
        from financial_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    if use_hybrid:
        from financial_rag.retrieval.hybrid import HybridRetriever
        return HybridRetriever(store=store, embedder=embedder, score_threshold=score_threshold)

    return VectorRetriever(store=store, embedder=embedder, score_threshold=score_threshold)
