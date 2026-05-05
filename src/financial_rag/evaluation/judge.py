"""LLM-as-judge for evaluating answer faithfulness."""

import re
import ollama


_JUDGE_PROMPT = """\
Evaluate if the answer is faithful to the provided source citations.

Question: {question}

Citations used:
{citations}

Answer given:
{answer}

Rate the faithfulness on this scale:
0 = Answer contradicts sources or invents information not in citations
1 = Answer is partially grounded but contains unsupported claims
2 = Answer is mostly grounded with minor gaps
3 = Answer is fully grounded — every claim is supported by the citations,
    or correctly states that information is not available

Rules:
- If citations are empty and the answer says "no information found", rate 3.
- If citations are empty but the answer makes specific claims, rate 0.
- Respond with ONLY a single digit: 0, 1, 2, or 3. Nothing else.\
"""


def judge_faithfulness(
    question: str,
    answer: str,
    citations: list[str],
    judge_model: str = "qwen3:4b",
) -> tuple[int, str]:
    """Score answer faithfulness using a small LLM as judge.

    Args:
        question: The original question.
        answer: The generated answer to evaluate.
        citations: List of citation strings from the retriever.
        judge_model: Ollama model to use as judge (use a fast small model).

    Returns:
        Tuple of (score 0-3, raw judge response for debugging).
    """
    citations_text = (
        "\n".join(f"- {c}" for c in citations) if citations else "(none)"
    )

    prompt = _JUDGE_PROMPT.format(
        question=question,
        citations=citations_text,
        answer=answer[:800],  # truncate very long answers
    )

    try:
        response = ollama.chat(
            model=judge_model,
            think=False,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.message.content.strip()
        # Strip thinking tags if present
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Extract first digit found
        match = re.search(r"[0-3]", raw)
        score = int(match.group()) if match else 1
        return score, raw
    except Exception as e:
        return 1, f"judge error: {e}"
