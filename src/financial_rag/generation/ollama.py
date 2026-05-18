"""Ollama-backed generator — runs fully local, no API key needed."""

import re
from collections.abc import Iterator

import ollama

from financial_rag.generation.base import BaseGenerator
from financial_rag.generation.models import ConversationTurn, GenerationResult
from financial_rag.retrieval.models import RetrievalResult

_SYSTEM_PROMPT = """\
You are a financial analyst assistant for Peruvian bank annual reports.

Your only job is to answer the question using the numbered context blocks provided.
Hard rules — every violation lowers your faithfulness score:
1. Every sentence must be traceable to a [N] block. Cite inline with [N].
2. Do NOT add general knowledge, background, or context not present in the blocks.
3. Do NOT infer, extrapolate, or synthesize beyond what is literally stated.
4. Do NOT round numbers, paraphrase statistics, or add qualifiers not in the source.
5. If the blocks lack information to answer fully, state exactly what is missing.
6. Respond in the same language as the question.\
"""

_NO_CONTEXT_ANSWER = (
    "No encontré información relevante en los documentos para responder esta pregunta."
)

AVAILABLE_MODELS = [
    "qwen3:4b",
    "qwen3:8b",
    "qwen3:14b",
    "qwen2.5-coder:32b",
    "mistral:latest",
]


class OllamaGenerator(BaseGenerator):
    """Generates grounded answers using a local Ollama model.

    Args:
        model: Ollama model tag. Must be pulled before use.
            Options: qwen3:4b (fast), qwen3:8b (balanced), qwen3:14b (quality).
        think: If True, enables Qwen3 thinking mode (slower but more accurate
            for complex financial reasoning). Only supported by Qwen3 models.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        think: bool = False,
    ) -> None:
        self._model = model
        self._think = think

    def generate(
        self,
        retrieval_result: RetrievalResult,
        history: list[ConversationTurn] | None = None,
    ) -> GenerationResult:
        if retrieval_result.is_empty:
            return GenerationResult(
                answer=_NO_CONTEXT_ANSWER,
                query=retrieval_result.query,
                citations=[],
                model=self._model,
            )

        response = ollama.chat(
            model=self._model,
            think=self._think,
            messages=self._build_messages(retrieval_result, history),
        )

        answer = self._clean_answer(response.message.content)
        return GenerationResult(
            answer=answer,
            query=retrieval_result.query,
            citations=retrieval_result.citations,
            model=self._model,
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )

    def stream(
        self,
        retrieval_result: RetrievalResult,
        history: list[ConversationTurn] | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        """Yield clean answer tokens, filtering out <think> blocks.

        Args:
            retrieval_result: Retrieved chunks and query.
            history: Prior conversation turns.
            model: Override the instance model for this request.
        """
        if retrieval_result.is_empty:
            yield _NO_CONTEXT_ANSWER
            return

        model_name = model or self._model
        full_text = ""
        in_think = False

        for chunk in ollama.chat(
            model=model_name,
            think=self._think,
            messages=self._build_messages(retrieval_result, history),
            stream=True,
        ):
            token = chunk.message.content or ""
            full_text += token

            if "<think>" in full_text:
                in_think = True
            if "</think>" in full_text:
                in_think = False
                full_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL)
                continue
            if in_think:
                continue

            if token:
                yield token

    def _build_messages(
        self,
        retrieval_result: RetrievalResult,
        history: list[ConversationTurn] | None = None,
    ) -> list[dict]:
        context_text = self._format_context(retrieval_result)
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in (history or []):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": f"{context_text}\n\nPregunta: {retrieval_result.query}"})
        return messages

    def _clean_answer(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def _format_context(self, retrieval_result: RetrievalResult) -> str:
        lines = ["Contexto:"]
        for i, result in enumerate(retrieval_result.results, start=1):
            lines.append(f"[{i}] {result.citation}")
            lines.append(result.chunk.content.strip())
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"OllamaGenerator(model={self._model!r}, think={self._think})"
