"""Unit tests for BenchmarkConfig and BenchmarkRunner."""

import pytest

from financial_rag.evaluation.runner import BenchmarkConfig, BenchmarkRunner
from financial_rag.evaluation.metrics import EvalSummary
from financial_rag.evaluation.questions import EvalQuestion


# ── BenchmarkConfig defaults ──────────────────────────────────────────────────

class TestBenchmarkConfig:
    def test_think_defaults_to_false(self):
        cfg = BenchmarkConfig(model="qwen3:4b", top_k=5)
        assert cfg.think is False

    def test_think_can_be_set_true(self):
        cfg = BenchmarkConfig(model="qwen3:14b", top_k=5, think=True)
        assert cfg.think is True

    def test_judge_model_default(self):
        cfg = BenchmarkConfig(model="qwen3:4b", top_k=5)
        assert cfg.judge_model == "qwen3:4b"

    def test_score_threshold_default(self):
        cfg = BenchmarkConfig(model="qwen3:4b", top_k=5)
        assert cfg.score_threshold == 0.0


# ── BenchmarkRunner construction ──────────────────────────────────────────────

class TestBenchmarkRunnerInit:
    def test_accepts_custom_questions(self):
        q = EvalQuestion(
            id="t_01",
            question="¿Cuál fue la utilidad neta?",
            question_type="factual",
            expected_source="interbank",
        )
        runner = BenchmarkRunner(questions=[q], configs=[BenchmarkConfig(model="x", top_k=3)])
        assert runner._questions == [q]

    def test_accepts_custom_configs(self):
        cfgs = [BenchmarkConfig(model="qwen3:4b", top_k=5, think=True)]
        runner = BenchmarkRunner(configs=cfgs, questions=[])
        assert runner._configs[0].think is True

    def test_default_configs_think_is_false(self):
        runner = BenchmarkRunner()
        assert all(not c.think for c in runner._configs)
