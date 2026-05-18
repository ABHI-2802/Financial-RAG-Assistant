"""Factory to build generators from configuration."""

from __future__ import annotations

from financial_rag.generation.base import BaseGenerator
from financial_rag.generation.ollama import AVAILABLE_MODELS, OllamaGenerator


def create_generator(
    model: str = "qwen3:8b",
    think: bool = False,
    backend: str = "ollama",
    api_key: str | None = None,
) -> BaseGenerator:
    """Build a generator ready to use.

    Args:
        model: Model identifier.
            Ollama: qwen3:4b, qwen3:8b, qwen3:14b, qwen2.5-coder:32b, mistral:latest
            Claude: claude-sonnet-4-6, claude-haiku-4-5, claude-opus-4-7
        think: Enable Qwen3 extended thinking (Ollama backend only).
        backend: "ollama" (local, no API key) or "claude" (Anthropic API).
        api_key: Anthropic API key (Claude backend only).
            If None, reads ANTHROPIC_API_KEY env var.

    Returns:
        Ready-to-use generator implementing BaseGenerator.

    Raises:
        ValueError: If the model or backend is unknown.
    """
    if backend == "ollama":
        if model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. Available: {AVAILABLE_MODELS}"
            )
        return OllamaGenerator(model=model, think=think)

    if backend == "claude":
        from financial_rag.generation.claude import AVAILABLE_CLAUDE_MODELS, ClaudeGenerator
        if model not in AVAILABLE_CLAUDE_MODELS:
            raise ValueError(
                f"Unknown model: '{model}'. Available: {AVAILABLE_CLAUDE_MODELS}"
            )
        return ClaudeGenerator(model=model, api_key=api_key)

    raise ValueError(f"Unknown backend: '{backend}'. Choose 'ollama' or 'claude'.")
