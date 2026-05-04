"""Smoke tests for project setup and configuration."""

from financial_rag import __version__
from financial_rag.config import Settings


def test_version():
    assert __version__ == "0.1.0"


def test_settings_defaults():
    s = Settings()
    assert s.llm_backend == "mock"
    assert s.chunk_size == 512
    assert s.chunk_overlap == 64
    assert s.retrieval_top_k == 5


def test_settings_override(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    s = Settings()
    assert s.chunk_size == 256
    assert s.llm_backend == "ollama"
