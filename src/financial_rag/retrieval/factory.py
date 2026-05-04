"""Factory to build a VectorRetriever from a saved index."""

from pathlib import Path

from financial_rag.embeddings.base import BaseEmbedder
from financial_rag.retrieval.retriever import VectorRetriever
from financial_rag.retrieval.vector_store import FAISSVectorStore


def create_retriever(
    store_path: str | Path,
    embedder: BaseEmbedder | None = None,
    score_threshold: float = 0.0,
) -> VectorRetriever:
    """Load a saved FAISS index and wrap it in a VectorRetriever.

    Args:
        store_path: Base path to the FAISS index (without extension).
            E.g. "data/processed/vector_store"
        embedder: Pre-built embedder instance. If None, defaults to
            SentenceTransformerEmbedder with all-MiniLM-L6-v2 — the same
            model used during indexing.
        score_threshold: Minimum cosine similarity for a result to be returned.

    Returns:
        Ready-to-query VectorRetriever.
    """
    store = FAISSVectorStore.load(store_path)

    if embedder is None:
        from financial_rag.embeddings.sentence_transformer import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()

    return VectorRetriever(store=store, embedder=embedder, score_threshold=score_threshold)
