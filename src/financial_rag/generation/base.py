"""Abstract base class for all generation backends."""

from abc import ABC, abstractmethod

from financial_rag.generation.models import GenerationResult
from financial_rag.retrieval.models import RetrievalResult


class BaseGenerator(ABC):
    """Contract for generation components.

    A generator takes a RetrievalResult (query + ranked chunks) and produces
    a grounded answer. Swap the LLM backend (Ollama, API, mock) without
    changing anything else in the RAG pipeline.
    """

    @abstractmethod
    def generate(self, retrieval_result: RetrievalResult) -> GenerationResult:
        """Generate an answer grounded in the retrieved chunks.

        Args:
            retrieval_result: Output from a Retriever, containing the query
                and ranked chunks to use as context.

        Returns:
            GenerationResult with the answer, citations, and token usage.
        """
        ...
