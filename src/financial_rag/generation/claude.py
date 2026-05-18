"""Anthropic Claude API-backed generator."""

from __future__ import annotations

from collections.abc import Iterator

from financial_rag.generation.base import BaseGenerator
from financial_rag.generation.models import ConversationTurn, GenerationResult
from financial_rag.retrieval.models import RetrievalResult

try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _anthropic_lib = None  # type: ignore[assignment]
    _ANTHROPIC_AVAILABLE = False

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

_DEFAULT_MODEL = "claude-sonnet-4-6"
_MAX_TOKENS = 2048

AVAILABLE_CLAUDE_MODELS = [
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
    "claude-opus-4-7",
]


class ClaudeGenerator(BaseGenerator):
    """Generates grounded answers using the Anthropic Claude API.

    Uses prompt caching on the system prompt to reduce cost and latency on
    repeated calls. The cached system prompt is reused for the lifetime of
    the process (ephemeral cache, up to 5 minutes TTL on the Anthropic side).

    Args:
        model: Claude model ID (default: claude-sonnet-4-6).
        api_key: Anthropic API key. Reads ANTHROPIC_API_KEY env var if None.
        max_tokens: Maximum output tokens per response.
        _client: Inject a pre-built client (for unit tests — avoids API calls).
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        api_key: str | None = None,
        max_tokens: int = _MAX_TOKENS,
        _client: object | None = None,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE and _client is None:
            raise ImportError(
                "The 'anthropic' package is required for ClaudeGenerator. "
                "Install it with: pip install 'financial-rag-assistant[claude]'"
            )
        self._model = model
        self._max_tokens = max_tokens
        self._client = _client or _anthropic_lib.Anthropic(api_key=api_key)
        # Cache the system prompt — tokens are reused across requests
        self._system = [
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ]

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

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._build_messages(retrieval_result, history),
        )

        return GenerationResult(
            answer=response.content[0].text,
            query=retrieval_result.query,
            citations=retrieval_result.citations,
            model=self._model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def stream(
        self,
        retrieval_result: RetrievalResult,
        history: list[ConversationTurn] | None = None,
        model: str | None = None,
    ) -> Iterator[str]:
        if retrieval_result.is_empty:
            yield _NO_CONTEXT_ANSWER
            return

        with self._client.messages.stream(
            model=model or self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._build_messages(retrieval_result, history),
        ) as stream:
            yield from stream.text_stream

    def _build_messages(
        self,
        retrieval_result: RetrievalResult,
        history: list[ConversationTurn] | None = None,
    ) -> list[dict]:
        messages: list[dict] = []
        for turn in (history or []):
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({
            "role": "user",
            "content": f"{self._format_context(retrieval_result)}\n\nPregunta: {retrieval_result.query}",
        })
        return messages

    def _format_context(self, retrieval_result: RetrievalResult) -> str:
        lines = ["Contexto:"]
        for i, result in enumerate(retrieval_result.results, start=1):
            lines.append(f"[{i}] {result.citation}")
            lines.append(result.chunk.content.strip())
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"ClaudeGenerator(model={self._model!r}, max_tokens={self._max_tokens})"
