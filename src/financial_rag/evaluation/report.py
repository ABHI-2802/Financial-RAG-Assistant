"""Generate reports from benchmark results."""

import csv
import json
from datetime import datetime
from pathlib import Path

from financial_rag.evaluation.metrics import EvalSummary


def print_summary_table(summaries: list[EvalSummary]) -> None:
    """Print a comparison table to stdout."""
    header = (
        f"{'Model':<22} {'top_k':>5} {'thresh':>6} "
        f"{'faith%':>7} {'hit%':>6} {'grnd%':>6} "
        f"{'rej%':>5} {'retr ms':>8} {'gen ms':>8} {'total ms':>9}"
    )
    print(f"\n{'═'*len(header)}")
    print("BENCHMARK RESULTS")
    print(f"{'═'*len(header)}")
    print(header)
    print(f"{'─'*len(header)}")

    # Sort by faithfulness desc, then total_ms asc
    sorted_s = sorted(summaries, key=lambda s: (-s.avg_faithfulness, s.avg_total_ms))

    for s in sorted_s:
        print(
            f"{s.model:<22} {s.top_k:>5} {s.score_threshold:>6.2f} "
            f"{s.avg_faithfulness*100:>6.0f}% {s.source_hit_rate*100:>5.0f}% "
            f"{s.grounded_rate*100:>5.0f}% {s.out_of_scope_rejection_rate*100:>4.0f}% "
            f"{s.avg_retrieval_ms:>8.0f} {s.avg_generation_ms:>8.0f} {s.avg_total_ms:>9.0f}"
        )

    print(f"{'─'*len(header)}")
    best = sorted_s[0]
    print(
        f"\n★  Best config: {best.model}  top_k={best.top_k}  "
        f"faithfulness={best.avg_faithfulness*100:.0f}%  "
        f"total={best.avg_total_ms:.0f}ms"
    )


def save_csv(summaries: list[EvalSummary], path: str | Path) -> None:
    """Save per-question detail as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in summaries:
        for r in s.results:
            rows.append({
                "model": r.model,
                "top_k": r.top_k,
                "score_threshold": r.score_threshold,
                "question_id": r.question_id,
                "question_type": r.question_type,
                "faithfulness": r.faithfulness,
                "faithfulness_pct": round(r.faithfulness_pct, 3),
                "source_hit": int(r.source_hit),
                "is_grounded": int(r.is_grounded),
                "top_score": round(r.top_score, 4),
                "chunks_used": r.chunks_used,
                "retrieval_ms": round(r.retrieval_ms, 1),
                "generation_ms": round(r.generation_ms, 1),
                "total_ms": round(r.total_ms, 1),
            })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  [report] CSV saved → {path}")


def save_json(summaries: list[EvalSummary], path: str | Path) -> None:
    """Save full results as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "summaries": [
            {
                "model": s.model,
                "top_k": s.top_k,
                "score_threshold": s.score_threshold,
                "n_questions": s.n,
                "avg_faithfulness_pct": round(s.avg_faithfulness * 100, 1),
                "source_hit_rate_pct": round(s.source_hit_rate * 100, 1),
                "grounded_rate_pct": round(s.grounded_rate * 100, 1),
                "avg_retrieval_ms": round(s.avg_retrieval_ms, 1),
                "avg_generation_ms": round(s.avg_generation_ms, 1),
                "avg_total_ms": round(s.avg_total_ms, 1),
            }
            for s in summaries
        ],
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  [report] JSON saved → {path}")
