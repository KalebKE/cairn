# Copyright 2025-2026 OMEGA Memory Maintainers
# SPDX-License-Identifier: Apache-2.0
"""Tests for the retrieval evaluation pipeline."""

import json
import os
from unittest.mock import patch

import pytest

from omega.evaluation.retrieval_eval import (
    EvalReport,
    _extract_key_terms,
    _ndcg,
    format_report,
    generate_keyword_probe,
    run_evaluation,
    sample_memories,
)
from omega.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_omega_dir):
    """Create a SQLiteStore with seed data for evaluation tests."""
    db_path = str(tmp_omega_dir / "omega.db")
    s = SQLiteStore(db_path)

    # Seed diverse memories
    test_memories = [
        ("Authentication bug in login flow causes session timeout", "error_pattern"),
        ("Decided to use PostgreSQL for the main database backend", "decision"),
        ("User prefers dark mode in all applications", "user_preference"),
        ("Lesson: always validate JWT tokens on the server side", "lesson_learned"),
        ("Constraint: never deploy to production on Fridays", "constraint"),
        ("Python 3.11 introduced significant performance improvements", "user_fact"),
        ("Error pattern: database connection pool exhaustion under load", "error_pattern"),
        ("Decision: migrated from REST to GraphQL for the API layer", "decision"),
        ("Lesson: use connection pooling to prevent database bottlenecks", "lesson_learned"),
        ("Preference: use pytest over unittest for all test suites", "user_preference"),
        ("Refactored the caching layer to use Redis instead of memcached", "decision"),
        ("Bug fix: race condition in concurrent file writes", "error_pattern"),
        ("Constraint: all API responses must include correlation IDs", "constraint"),
        ("Lesson: semantic search works better with longer content chunks", "lesson_learned"),
        ("Decision: adopted TypeScript for all frontend projects", "decision"),
        ("User fact: the staging environment runs on Kubernetes", "user_fact"),
        ("Error: webhook delivery fails silently when endpoint returns 503", "error_pattern"),
        ("Preference: commit messages should follow conventional commits format", "user_preference"),
        ("Lesson: batch database writes for better throughput", "lesson_learned"),
        ("Decision: switched from SQLAlchemy to raw SQL for performance", "decision"),
    ]

    for content, event_type in test_memories:
        s.store(
            content=content,
            metadata={"event_type": event_type, "tags": []},
            session_id="test-session",
        )

    return s


# ---------------------------------------------------------------------------
# _extract_key_terms
# ---------------------------------------------------------------------------


class TestExtractKeyTerms:
    def test_extracts_meaningful_terms(self):
        terms = _extract_key_terms("Authentication bug in login flow causes session timeout")
        assert "authentication" in terms
        assert "login" in terms
        assert "bug" in terms
        # Stop words and OMEGA noise words should be filtered
        assert "in" not in terms
        assert "session" not in terms  # Filtered as OMEGA noise word

    def test_filters_stop_words(self):
        terms = _extract_key_terms("the quick brown fox jumps over the lazy dog")
        assert "the" not in terms
        assert "over" not in terms

    def test_filters_omega_noise_words(self):
        terms = _extract_key_terms("committed files changes to the session memory")
        # All these are in the noise word list
        assert "committed" not in terms
        assert "files" not in terms
        assert "session" not in terms
        assert "memory" not in terms

    def test_max_terms_limit(self):
        terms = _extract_key_terms(
            "alpha bravo charlie delta echo foxtrot golf hotel india juliet",
            max_terms=3,
        )
        assert len(terms) <= 3

    def test_deduplicates(self):
        terms = _extract_key_terms("postgres postgresql postgres database")
        assert terms.count("postgres") <= 1

    def test_empty_content(self):
        terms = _extract_key_terms("")
        assert terms == []

    def test_short_terms_filtered(self):
        terms = _extract_key_terms("I am a go to my DB")
        # "go" and "am" are 2 chars, should be filtered
        assert "go" not in terms
        assert "am" not in terms


# ---------------------------------------------------------------------------
# generate_keyword_probe
# ---------------------------------------------------------------------------


class TestGenerateKeywordProbe:
    def test_generates_probe_from_content(self):
        probe = generate_keyword_probe(
            "mem-123",
            "Authentication bug in login flow causes session timeout",
            "error_pattern",
        )
        assert probe is not None
        assert probe.source_memory_id == "mem-123"
        assert probe.source_event_type == "error_pattern"
        assert probe.query_method == "keyword"
        assert len(probe.query_text) > 0

    def test_returns_none_for_short_content(self):
        probe = generate_keyword_probe("mem-456", "ok", "memory")
        assert probe is None

    def test_returns_none_for_all_stop_words(self):
        probe = generate_keyword_probe("mem-789", "the is a an to of in", "memory")
        assert probe is None


# ---------------------------------------------------------------------------
# sample_memories
# ---------------------------------------------------------------------------


class TestSampleMemories:
    def test_samples_correct_count(self, store):
        samples = sample_memories(store, sample_size=10, seed=42)
        assert len(samples) == 10

    def test_samples_diverse_types(self, store):
        samples = sample_memories(store, sample_size=15, seed=42)
        types = {s["event_type"] for s in samples}
        # Should have at least 3 different types
        assert len(types) >= 3

    def test_deterministic_with_seed(self, store):
        s1 = sample_memories(store, sample_size=10, seed=42)
        s2 = sample_memories(store, sample_size=10, seed=42)
        assert [m["id"] for m in s1] == [m["id"] for m in s2]

    def test_different_seed_gives_different_sample(self, store):
        s1 = sample_memories(store, sample_size=10, seed=42)
        s2 = sample_memories(store, sample_size=10, seed=99)
        ids1 = {m["id"] for m in s1}
        ids2 = {m["id"] for m in s2}
        # Very unlikely to be identical with different seeds
        assert ids1 != ids2

    def test_handles_empty_store(self, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "empty.db")
        empty_store = SQLiteStore(db_path)
        samples = sample_memories(empty_store, sample_size=10)
        assert samples == []

    def test_sample_size_larger_than_store(self, store):
        # Store has 20 memories, ask for 50
        samples = sample_memories(store, sample_size=50, seed=42)
        assert len(samples) <= 20
        assert len(samples) > 0

    def test_excludes_superseded(self, store):
        # Store a memory then supersede it
        node_id = store.store(
            content="This old decision about using MongoDB was wrong",
            metadata={"event_type": "decision", "superseded": True, "tags": []},
            session_id="test-session",
        )
        samples = sample_memories(store, sample_size=50, seed=42)
        sample_ids = {s["id"] for s in samples}
        assert node_id not in sample_ids


# ---------------------------------------------------------------------------
# NDCG metric
# ---------------------------------------------------------------------------


class TestNDCG:
    def test_perfect_ranking(self):
        scores = [3, 2, 1, 0]
        assert _ndcg(scores, k=4) == pytest.approx(1.0)

    def test_worst_ranking(self):
        scores = [0, 0, 0, 3]
        assert _ndcg(scores, k=4) < 1.0
        assert _ndcg(scores, k=4) > 0.0

    def test_empty_scores(self):
        assert _ndcg([], k=5) == 0.0

    def test_all_zeros(self):
        assert _ndcg([0, 0, 0], k=3) == 0.0


# ---------------------------------------------------------------------------
# run_evaluation (basic mode, no LLM)
# ---------------------------------------------------------------------------


class TestRunEvaluation:
    def test_basic_mode_returns_report(self, store, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "omega.db")
        report = run_evaluation(
            db_path=db_path,
            sample_size=10,
            top_k=5,
            judge=False,
            seed=42,
        )
        assert isinstance(report, EvalReport)
        assert report.mode == "basic"
        assert report.sample_size <= 10
        assert report.total_memories == 20
        assert 0.0 <= report.hit_rate <= 1.0
        assert 0.0 <= report.mrr <= 1.0
        assert report.duration_seconds >= 0
        # Basic mode should not have judge metrics
        assert report.precision_at_k is None
        assert report.ndcg_at_k is None
        assert report.llm_calls == 0

    def test_report_has_probe_results(self, store, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "omega.db")
        report = run_evaluation(db_path=db_path, sample_size=5, top_k=3, seed=42)
        assert len(report.probe_results) > 0
        for pr in report.probe_results:
            assert "query" in pr
            assert "source_id" in pr
            assert "hit" in pr
            assert "results" in pr

    def test_report_has_type_breakdown(self, store, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "omega.db")
        report = run_evaluation(db_path=db_path, sample_size=15, top_k=5, seed=42)
        assert len(report.by_event_type) > 0
        for etype, metrics in report.by_event_type.items():
            assert "count" in metrics
            assert "hit_rate" in metrics
            assert "mrr" in metrics

    def test_saves_json_output(self, store, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "omega.db")
        output_path = str(tmp_omega_dir / "report.json")
        run_evaluation(
            db_path=db_path,
            sample_size=5,
            top_k=3,
            seed=42,
            output_path=output_path,
        )
        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "hit_rate" in data
        assert "mrr" in data
        assert "probe_results" in data

    def test_empty_store_returns_empty_report(self, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "empty.db")
        SQLiteStore(db_path)  # Create empty DB
        report = run_evaluation(db_path=db_path, sample_size=10)
        assert report.sample_size == 0
        assert report.hit_rate == 0.0

    def test_deterministic_results(self, store, tmp_omega_dir):
        db_path = str(tmp_omega_dir / "omega.db")
        r1 = run_evaluation(db_path=db_path, sample_size=10, seed=42)
        r2 = run_evaluation(db_path=db_path, sample_size=10, seed=42)
        assert r1.hit_rate == r2.hit_rate
        assert r1.mrr == r2.mrr


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_formats_basic_report(self):
        report = EvalReport(
            timestamp="2026-02-22T10:00:00",
            sample_size=20,
            top_k=5,
            mode="basic",
            total_memories=500,
            hit_rate=0.75,
            mrr=0.625,
            by_event_type={
                "decision": {"count": 8, "hit_rate": 0.875, "mrr": 0.75},
                "lesson_learned": {"count": 5, "hit_rate": 0.6, "mrr": 0.5},
            },
            probe_results=[],
        )
        text = format_report(report)
        assert "Retrieval Evaluation Report" in text
        assert "75.0%" in text
        assert "0.625" in text
        assert "decision" in text

    def test_formats_judge_report(self):
        report = EvalReport(
            timestamp="2026-02-22T10:00:00",
            sample_size=20,
            top_k=5,
            mode="judge",
            total_memories=500,
            hit_rate=0.8,
            mrr=0.7,
            precision_at_k=0.65,
            ndcg_at_k=0.72,
            avg_relevance=2.1,
            llm_calls=120,
            llm_input_tokens=50000,
            llm_output_tokens=600,
            probe_results=[],
        )
        text = format_report(report)
        assert "Precision" in text
        assert "NDCG" in text
        assert "Cost" in text
        assert "120" in text

    def test_formats_misses(self):
        report = EvalReport(
            timestamp="2026-02-22T10:00:00",
            sample_size=2,
            top_k=5,
            mode="basic",
            total_memories=100,
            hit_rate=0.5,
            mrr=0.25,
            probe_results=[
                {
                    "query": "test query",
                    "query_method": "keyword",
                    "source_id": "mem-abc",
                    "source_type": "decision",
                    "hit": False,
                    "source_rank": None,
                    "results": [
                        {
                            "id": "mem-xyz",
                            "rank": 1,
                            "score": 0.5,
                            "relevance": None,
                            "is_source": False,
                            "preview": "other",
                        }
                    ],
                },
                {
                    "query": "another query",
                    "query_method": "keyword",
                    "source_id": "mem-def",
                    "source_type": "lesson_learned",
                    "hit": True,
                    "source_rank": 1,
                    "results": [],
                },
            ],
        )
        text = format_report(report)
        assert "Misses" in text
        assert "mem-abc" in text


# ---------------------------------------------------------------------------
# LLM-based functions (mocked)
# ---------------------------------------------------------------------------


class TestLLMProbeGeneration:
    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_generate_llm_probe(self, mock_llm):
        from omega.evaluation.retrieval_eval import generate_llm_probe

        mock_llm.return_value = ("How to fix database connection pool issues", 150, 12)
        probe, in_tok, out_tok = generate_llm_probe(
            "mem-123", "Database connection pool exhaustion under load", "error_pattern"
        )
        assert probe is not None
        assert probe.query_method == "llm"
        assert "database" in probe.query_text.lower() or "connection" in probe.query_text.lower()
        assert in_tok == 150
        assert out_tok == 12

    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_generate_llm_probe_handles_failure(self, mock_llm):
        from omega.evaluation.retrieval_eval import generate_llm_probe

        mock_llm.side_effect = Exception("API error")
        probe, in_tok, out_tok = generate_llm_probe("mem-123", "Some content", "decision")
        assert probe is None
        assert in_tok == 0

    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_judge_relevance(self, mock_llm):
        from omega.evaluation.retrieval_eval import judge_relevance

        mock_llm.return_value = ("3", 100, 3)
        score, in_tok, out_tok = judge_relevance("database pooling", "Connection pool config")
        assert score == 3
        assert in_tok == 100

    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_judge_relevance_handles_bad_output(self, mock_llm):
        from omega.evaluation.retrieval_eval import judge_relevance

        mock_llm.return_value = ("not a number", 100, 5)
        score, _, _ = judge_relevance("query", "content")
        assert score == 1  # Defaults to 1

    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_judge_relevance_clamps_score(self, mock_llm):
        from omega.evaluation.retrieval_eval import judge_relevance

        mock_llm.return_value = ("9", 100, 3)
        score, _, _ = judge_relevance("query", "content")
        assert score == 3  # Clamped to max


# ---------------------------------------------------------------------------
# run_evaluation (judge mode, mocked LLM)
# ---------------------------------------------------------------------------


class TestRunEvaluationJudgeMode:
    @patch("omega.evaluation.retrieval_eval._call_llm")
    def test_judge_mode_computes_extra_metrics(self, mock_llm, store, tmp_omega_dir):
        # Mock LLM to return probe queries and relevance scores
        call_count = [0]

        def fake_llm(prompt, **kwargs):
            call_count[0] += 1
            if "search query" in prompt:
                # Probe generation
                return ("database connection pooling fix", 150, 12)
            else:
                # Relevance judging
                return ("2", 80, 3)

        mock_llm.side_effect = fake_llm

        db_path = str(tmp_omega_dir / "omega.db")
        report = run_evaluation(
            db_path=db_path,
            sample_size=5,
            top_k=3,
            judge=True,
            seed=42,
        )
        assert report.mode == "judge"
        assert report.llm_calls > 0
        assert report.precision_at_k is not None
        assert report.ndcg_at_k is not None
        assert report.avg_relevance is not None
