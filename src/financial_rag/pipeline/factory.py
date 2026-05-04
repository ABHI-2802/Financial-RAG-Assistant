"""Factory to build a ready-to-use RAGPipeline from disk."""

from pathlib import Path

from financial_rag.generation.ollama import OllamaGenerator
from financial_rag.pipeline.pipeline import RAGPipeline
from financial_rag.retrieval.factory import create_retriever


def create_pipeline(
    store_path: str | Path = "data/processed/vector_store",
    model: str = "qwen3:8b",
    top_k: int = 5,
    score_threshold: float = 0.0,
    think: bool = False,
) -> RAGPipeline:
    """Load a saved FAISS index and wire it into a full RAG pipeline.

    Args:
        store_path: Base path to the FAISS index (without extension).
        model: Ollama model tag for generation.
            Options: qwen3:4b (fast), qwen3:8b (balanced), qwen3:14b (quality).
        top_k: Default number of chunks to retrieve per query.
        score_threshold: Minimum cosine similarity to include a chunk.
        think: Enable Qwen3 extended thinking mode.

    Returns:
        RAGPipeline ready to answer questions.
    """
    retriever = create_retriever(store_path, score_threshold=score_threshold)
    generator = OllamaGenerator(model=model, think=think)
    return RAGPipeline(retriever=retriever, generator=generator, top_k=top_k)
