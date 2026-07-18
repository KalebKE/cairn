"""Retrieval knowledge-fit regression tests (fork Phase 2).

Pins the ranking semantics that make retrieval favor durable knowledge:

1. The strong-signal FTS short-circuit must apply the same metadata scoring
   (type weight x feedback x priority x decay) as the fusion phase — upstream
   returned raw text scores, letting a lexically-sharp task_completion outrank
   a decision that would have won under fusion.
2. Episodic task_completion must decay FASTER than decisions (upstream gave
   them the same half-life despite being 40x more numerous exhaust).
3. Access-aware decay slowdown must cover lessons/insights, not just decisions
   — a frequently-recalled lesson should persist longer.
4. Temporal inference must not fire on incidental month/weekday words
   ("we may need...", "the march of progress").
"""
from datetime import datetime, timedelta, timezone

import pytest


def _iso_days_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


class TestMetadataScoreFactor:
    """_metadata_score_factor is the shared scoring helper used by both the
    fusion phase and the strong-signal short-circuit."""

    def _node(self, store, content, event_type):
        nid = store.store(content=content, metadata={"event_type": event_type})
        return store.get_node(nid, track_access=False)

    def test_decision_outweighs_task_completion(self, store):
        d = self._node(store, "Decision: adopt server-issued sync tokens", "decision")
        t = self._node(store, "Completed: rotated the sync tokens on staging", "task_completion")
        assert store._metadata_score_factor(d) > store._metadata_score_factor(t), (
            "same-age decision must outweigh task_completion (2.0 vs 1.4 type weight)"
        )

    def test_short_circuit_path_uses_metadata_scoring(self, store, monkeypatch):
        """Wire check: when the strong-signal shortcut fires, scores must pass
        through _metadata_score_factor (not raw FTS relevance)."""
        store.store(
            content="zephyrblat calibration manifold quorum settings for ingestion",
            metadata={"event_type": "decision"},
        )
        store.store(
            content="zephyrblat overview note",
            metadata={"event_type": "task_completion"},
        )
        calls = {"n": 0}
        orig = store._metadata_score_factor

        def counting(node):
            calls["n"] += 1
            return orig(node)

        monkeypatch.setattr(store, "_metadata_score_factor", counting)
        store.query("zephyrblat calibration manifold quorum settings ingestion", limit=5)
        assert calls["n"] > 0, (
            "query scoring must route through _metadata_score_factor "
            "(both fusion AND the strong-signal shortcut)"
        )


class TestKnowledgeDecay:
    def test_task_completion_decays_faster_than_decision(self, store):
        created = _iso_days_ago(60)
        tc = store._compute_decay_factor("task_completion", None, created, access_count=0)
        dec = store._compute_decay_factor("decision", None, created, access_count=0)
        assert tc < dec, "episodic exhaust must decay faster than decisions"

    # A real access-aware slowdown produces a decisive gap; a >0 comparison
    # alone can pass on datetime.now() drift between the two calls.
    _MARGIN = 0.05

    def test_lesson_access_slows_decay(self, store):
        last = _iso_days_ago(100)
        well_used = store._compute_decay_factor("lesson_learned", last, None, access_count=10)
        barely_used = store._compute_decay_factor("lesson_learned", last, None, access_count=1)
        assert well_used > barely_used + self._MARGIN, (
            "a frequently-recalled lesson must persist longer (access-aware decay)"
        )

    def test_insight_access_slows_decay(self, store):
        last = _iso_days_ago(30)
        well_used = store._compute_decay_factor("advisor_insight", last, None, access_count=10)
        barely_used = store._compute_decay_factor("advisor_insight", last, None, access_count=1)
        assert well_used > barely_used + self._MARGIN

    def test_decision_access_slowdown_unchanged(self, store):
        last = _iso_days_ago(60)
        well_used = store._compute_decay_factor("decision", last, None, access_count=10)
        barely_used = store._compute_decay_factor("decision", last, None, access_count=1)
        assert well_used > barely_used + self._MARGIN


class TestTemporalInferenceWordBoundary:
    def _infer(self, text):
        from omega.bridge import _infer_temporal_range
        return _infer_temporal_range(text)

    @pytest.mark.parametrize("text", [
        "we may need to refactor the ingestion parser",
        "the march of technology is relentless in this codebase",
        "handle the mayor of casterbridge edge case",
        "monday-morning quarterbacking the outage postmortem",
    ])
    def test_incidental_words_do_not_trigger(self, text):
        assert self._infer(text) is None, f"false temporal trigger on: {text!r}"

    @pytest.mark.parametrize("text", [
        "what did we decide in may",
        "decisions from march 2026",
        "lessons since june",
        "what shipped last monday",
    ])
    def test_real_temporal_references_trigger(self, text):
        assert self._infer(text) is not None, f"missed temporal reference: {text!r}"
