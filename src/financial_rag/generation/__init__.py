"""Generation — LLM-backed answer synthesis from retrieved chunks."""

from financial_rag.generation.models import GenerationResult
from financial_rag.generation.base import BaseGenerator
from financial_rag.generation.ollama import OllamaGenerator, AVAILABLE_MODELS
from financial_rag.generation.mock import MockGenerator
from financial_rag.generation.factory import create_generator

__all__ = [
    "GenerationResult",
    "BaseGenerator",
    "OllamaGenerator",
    "MockGenerator",
    "AVAILABLE_MODELS",
    "create_generator",
]
