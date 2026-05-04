"""Factory to build generators from configuration."""

from financial_rag.generation.ollama import AVAILABLE_MODELS, OllamaGenerator


def create_generator(
    model: str = "qwen3:8b",
    think: bool = False,
) -> OllamaGenerator:
    """Build an OllamaGenerator ready to use.

    Args:
        model: Ollama model tag. Must be pulled with `ollama pull <model>`.
            Available: qwen3:4b, qwen3:8b, qwen3:14b, qwen2.5-coder:32b, mistral:latest
        think: Enable Qwen3 extended thinking mode for harder questions.

    Returns:
        Ready-to-use OllamaGenerator.

    Raises:
        ValueError: If the model tag is not in the known list.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: '{model}'. Available: {AVAILABLE_MODELS}"
        )
    return OllamaGenerator(model=model, think=think)
