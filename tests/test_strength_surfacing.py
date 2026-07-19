"""Tests for strength surfacing in query results."""
import pytest
from cairn.sqlite_store import SQLiteStore


class TestStrengthField:
    """Test that MemoryResult includes strength score."""

    def test_memory_result_has_strength_slot(self, store):
        node_id = store.store(
            content="Temporal decay is computed at query time",
            metadata={"event_type": "decision"},
        )
        results = store.query("temporal decay")
        assert len(results) > 0
        assert hasattr(results[0], "strength"), "MemoryResult missing 'strength' attribute"

    def test_strength_is_float_between_0_and_1(self, store):
        store.store(content="Alpha memory", metadata={"event_type": "decision"})
        store.store(content="Beta memory", metadata={"event_type": "lesson_learned"})
        store.store(content="Gamma memory", metadata={"event_type": "session_summary"})
        results = store.query("memory")
        for r in results:
            assert 0.0 <= r.strength <= 1.0, f"strength {r.strength} not in [0, 1]"

    def test_strength_default_is_zero(self):
        from cairn.sqlite_store import MemoryResult
        mr = MemoryResult(id="test-123", content="test")
        assert mr.strength == 0.0


class TestStrengthComputation:
    """Test that strength is computed from decay, feedback, type weight."""

    def test_decisions_have_higher_strength_than_task_completions(self, store):
        # Verify type_weight difference: decision=2.0 vs task_completion=1.4.
        # Use `store` fixture with direct SQL to bypass store-time dedup,
        # then query and compare normalized strength values.
        import hashlib
        import json as _json
        import unicodedata
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        for etype, content in [
            ("task_completion", "Finished migrating Postgres indexes to improve read throughput"),
            ("decision", "Decided to migrate Postgres indexes for better read throughput"),
        ]:
            nid = f"strength-{etype}-test"
            chash = hashlib.sha256(content.encode()).hexdigest()
            canon = unicodedata.normalize("NFKC", content).lower()
            canon_hash = hashlib.sha256(canon.encode()).hexdigest()
            store._conn.execute(
                """INSERT OR REPLACE INTO memories
                   (node_id, content, metadata, created_at, access_count, last_accessed,
                    content_hash, canonical_hash, event_type, priority)
                   VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, 3)""",
                (nid, content, _json.dumps({"event_type": etype, "priority": 3}),
                 now, now, chash, canon_hash, etype),
            )
        store._conn.commit()
        store._invalidate_query_cache()

        results = store.query("Postgres indexes migrate read throughput")
        decision = next((r for r in results if (r.metadata or {}).get("event_type") == "decision"), None)
        task = next((r for r in results if (r.metadata or {}).get("event_type") == "task_completion"), None)
        assert decision is not None and task is not None, (
            f"Both types expected, got: {[(r.metadata or {}).get('event_type') for r in results]}"
        )
        assert decision.strength >= task.strength, (
            f"Decision strength ({decision.strength}) should be >= task_completion ({task.strength})"
        )

    def test_strength_nonzero_for_query_results(self, store):
        store.store(content="Important architecture decision", metadata={"event_type": "decision"})
        results = store.query("architecture")
        assert len(results) > 0
        assert results[0].strength > 0.0, "Top result should have nonzero strength"

    def test_negative_feedback_reduces_strength(self, store):
        nid1 = store.store(
            content="Lesson learned: pytest fixtures should use tmpdir for isolation",
            metadata={"event_type": "lesson_learned", "feedback_score": 3},
        )
        nid2 = store.store(
            content="Lesson learned: database migrations require schema version bumps",
            metadata={"event_type": "lesson_learned", "feedback_score": -3},
        )
        # Query broadly enough to retrieve both distinct lessons
        results = store.query("lesson learned")
        good = next((r for r in results if r.id == nid1), None)
        bad = next((r for r in results if r.id == nid2), None)
        assert good is not None and bad is not None, (
            f"Expected both results. good={good}, bad={bad}, "
            f"result_ids={[r.id for r in results]}, nid1={nid1}, nid2={nid2}"
        )
        assert good.strength > bad.strength


@pytest.mark.usefixtures("_reset_bridge")
class TestStrengthInBridgeOutput:

    def test_query_markdown_includes_strength(self, tmp_cairn_dir):
        from cairn.bridge import store, query
        store(content="Architecture: we use event sourcing", event_type="decision")
        result = query(query_text="event sourcing")
        assert "str:" in result.lower(), f"Strength not found in query output: {result[:300]}"

    def test_query_structured_includes_strength(self, tmp_cairn_dir):
        from cairn.bridge import store, query_structured
        store(content="Architecture: we use event sourcing", event_type="decision")
        results = query_structured(query_text="event sourcing")
        assert len(results) > 0
        assert "strength" in results[0], f"Missing 'strength' key. Keys: {results[0].keys()}"
        assert isinstance(results[0]["strength"], float)


@pytest.mark.usefixtures("_reset_bridge")
class TestStrengthMinFilter:

    def test_strength_min_filters_weak_results(self, tmp_cairn_dir):
        from cairn.bridge import store, query
        store(content="Critical architecture decision about database", event_type="decision")
        store(content="Random task completion note about database indexing", event_type="task_completion")
        all_results = query(query_text="database")
        filtered = query(query_text="database", strength_min=0.8)
        assert len(filtered.split("##")) <= len(all_results.split("##"))

    def test_strength_min_zero_returns_all(self, tmp_cairn_dir):
        from cairn.bridge import store, query
        store(content="Test memory for strength filtering", event_type="decision")
        all_results = query(query_text="strength filtering")
        filtered = query(query_text="strength filtering", strength_min=0.0)
        assert filtered == all_results
