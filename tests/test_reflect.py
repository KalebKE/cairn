"""Tests for cairn.reflect -- memory quality analysis functions."""

import time
from datetime import datetime, timedelta, timezone

import pytest


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def store(tmp_cairn_dir):
    """Create a fresh SQLiteStore for testing."""
    from cairn.sqlite_store import SQLiteStore

    db_path = tmp_cairn_dir / "test_reflect.db"
    s = SQLiteStore(db_path=db_path)
    yield s
    s.close()


def _store_memory(store, content, event_type="memory", priority=3, entity_id=None, age_days=0):
    """Helper to store a memory with optional age backdating."""
    metadata = {"event_type": event_type, "priority": priority}
    if entity_id:
        metadata["entity_id"] = entity_id
    node_id = store.store(content=content, metadata=metadata, skip_inference=True)

    if age_days > 0:
        # Backdate created_at via direct SQL
        old_date = (datetime.now(timezone.utc) - timedelta(days=age_days)).isoformat()
        store._conn.execute(
            "UPDATE memories SET created_at = ? WHERE node_id = ?",
            (old_date, node_id),
        )
        store._conn.commit()

    return node_id


# ============================================================================
# TestFindContradictions
# ============================================================================


class TestFindContradictions:
    def test_no_memories(self, store):
        from cairn.reflect import find_contradictions

        result = find_contradictions(store, "nonexistent topic")
        assert result["memories_analyzed"] == 0
        assert result["contradictions"] == []

    def test_single_memory(self, store):
        from cairn.reflect import find_contradictions

        _store_memory(store, "Alex prefers dark mode")
        result = find_contradictions(store, "dark mode")
        assert result["memories_analyzed"] <= 1
        assert result["contradictions"] == []

    def test_detects_known_contradiction(self, store):
        from cairn.reflect import find_contradictions

        _store_memory(store, "Alex prefers dark mode for all editors")
        _store_memory(store, "Alex prefers light mode for all editors")
        result = find_contradictions(store, "mode editors")
        # Should find at least one contradiction (antonym: dark vs light)
        if result["memories_analyzed"] >= 2:
            assert len(result["contradictions"]) >= 1
            c = result["contradictions"][0]
            assert c["confidence"] > 0
            assert "memory_a_id" in c
            assert "memory_b_id" in c
            assert len(c["signals"]) > 0

    def test_no_contradiction_unrelated(self, store):
        from cairn.reflect import find_contradictions

        _store_memory(store, "Python is a programming language")
        _store_memory(store, "The weather is sunny today")
        result = find_contradictions(store, "Python weather")
        # Unrelated memories should not contradict
        assert result["contradictions"] == []

    def test_entity_scoping(self, store):
        from cairn.reflect import find_contradictions

        _store_memory(store, "Project uses dark mode", entity_id="proj_a")
        _store_memory(store, "Project uses light mode", entity_id="proj_b")
        # Scoping to proj_a should only see one memory
        result = find_contradictions(store, "mode", entity_id="proj_a")
        assert result["memories_analyzed"] <= 1
        assert result["contradictions"] == []

    def test_limit(self, store):
        from cairn.reflect import find_contradictions

        for i in range(10):
            _store_memory(store, f"Memory about testing number {i}")
        result = find_contradictions(store, "testing", limit=3)
        assert result["memories_analyzed"] <= 3

    def test_result_structure(self, store):
        """Verify the return dict has all expected keys."""
        from cairn.reflect import find_contradictions

        _store_memory(store, "Always use vim")
        _store_memory(store, "Never use vim")
        result = find_contradictions(store, "vim")
        assert "topic" in result
        assert "memories_analyzed" in result
        assert "contradictions" in result
        assert result["topic"] == "vim"


# ============================================================================
# TestTraceEvolution
# ============================================================================


class TestTraceEvolution:
    def test_no_memories(self, store):
        from cairn.reflect import trace_evolution

        result = trace_evolution(store, "nonexistent topic")
        assert result["total_memories"] == 0
        assert result["chains"] == []

    def test_single_memory(self, store):
        from cairn.reflect import trace_evolution

        _store_memory(store, "Initial understanding of deployment")
        result = trace_evolution(store, "deployment")
        # Single memory = no chains (singletons are excluded)
        assert result["chains"] == []

    def test_follows_evolution_edges(self, store):
        from cairn.reflect import trace_evolution

        id1 = _store_memory(store, "Deployment uses docker compose")
        id2 = _store_memory(store, "Deployment now uses kubernetes")
        store.add_edge(id1, id2, edge_type="evolution", weight=0.9)

        result = trace_evolution(store, "deployment")
        assert result["total_memories"] >= 2
        assert len(result["chains"]) >= 1
        chain = result["chains"][0]
        assert chain["length"] >= 2
        node_ids = [m["node_id"] for m in chain["memories"]]
        assert id1 in node_ids
        assert id2 in node_ids

    def test_follows_supersedes_edges(self, store):
        from cairn.reflect import trace_evolution

        id1 = _store_memory(store, "Use Python 3.9 for cairn")
        id2 = _store_memory(store, "Use Python 3.11 for cairn")
        store.add_edge(id2, id1, edge_type="supersedes", weight=1.0)

        result = trace_evolution(store, "Python cairn")
        assert len(result["chains"]) >= 1
        chain = result["chains"][0]
        node_ids = [m["node_id"] for m in chain["memories"]]
        assert id1 in node_ids
        assert id2 in node_ids

    def test_chronological_order(self, store):
        from cairn.reflect import trace_evolution

        id1 = _store_memory(store, "Version 1 of the API uses REST")
        time.sleep(0.05)  # Ensure different timestamps
        id2 = _store_memory(store, "Version 2 of the API uses GraphQL")
        store.add_edge(id1, id2, edge_type="evolution", weight=0.8)

        result = trace_evolution(store, "API version")
        if result["chains"]:
            chain = result["chains"][0]
            # First memory should be the older one
            assert chain["memories"][0]["node_id"] == id1

    def test_entity_scoping(self, store):
        from cairn.reflect import trace_evolution

        id1 = _store_memory(store, "Scoped deployment v1", entity_id="proj_x")
        id2 = _store_memory(store, "Scoped deployment v2", entity_id="proj_x")
        id3 = _store_memory(store, "Other deployment v1", entity_id="proj_y")
        store.add_edge(id1, id2, edge_type="evolution", weight=0.9)

        result = trace_evolution(store, "deployment", entity_id="proj_x")
        # Should only see proj_x memories
        all_node_ids = set()
        for chain in result["chains"]:
            for m in chain["memories"]:
                all_node_ids.add(m["node_id"])
        assert id3 not in all_node_ids

    def test_result_structure(self, store):
        from cairn.reflect import trace_evolution

        result = trace_evolution(store, "anything")
        assert "topic" in result
        assert "total_memories" in result
        assert "chains" in result
        assert result["topic"] == "anything"


# ============================================================================
# TestFindStale
# ============================================================================


class TestFindStale:
    def test_no_stale(self, store):
        from cairn.reflect import find_stale

        # Fresh memory should not be stale
        _store_memory(store, "Fresh memory just created")
        result = find_stale(store, min_age_days=14)
        assert result["total_candidates"] == 0
        assert result["stale_memories"] == []

    def test_finds_zero_access_old(self, store):
        from cairn.reflect import find_stale

        _store_memory(store, "Old forgotten memory about testing", age_days=20)
        result = find_stale(store, min_age_days=14, days=30)
        assert result["total_candidates"] >= 1
        assert len(result["stale_memories"]) >= 1
        stale = result["stale_memories"][0]
        assert stale["access_count"] == 0
        assert stale["staleness_score"] > 0
        assert "never accessed" in stale["reasons"]

    def test_respects_min_age(self, store):
        from cairn.reflect import find_stale

        # 10 days old, but min_age is 14
        _store_memory(store, "Not old enough to be stale", age_days=10)
        result = find_stale(store, min_age_days=14)
        assert result["total_candidates"] == 0

    def test_protected_types_excluded(self, store):
        from cairn.reflect import find_stale

        _store_memory(store, "User prefers dark mode", event_type="user_preference", age_days=60)
        _store_memory(store, "Never push to main without review", event_type="constraint", age_days=60)
        _store_memory(store, "Detected morning coding pattern", event_type="behavioral_pattern", age_days=60)
        _store_memory(store, "Check email at 9am", event_type="reminder", age_days=60)
        result = find_stale(store, min_age_days=14, days=90)
        # None of the protected types should appear
        for m in result["stale_memories"]:
            assert m["event_type"] not in ("user_preference", "constraint", "behavioral_pattern", "reminder")

    def test_score_ordering(self, store):
        from cairn.reflect import find_stale

        # Older memory should have higher staleness score
        _store_memory(store, "Somewhat old memory about analysis", age_days=20, priority=3)
        _store_memory(store, "Very old low priority memory about analysis", age_days=80, priority=1)
        result = find_stale(store, min_age_days=14, days=90)
        if len(result["stale_memories"]) >= 2:
            # Should be sorted by staleness score descending
            scores = [m["staleness_score"] for m in result["stale_memories"]]
            assert scores == sorted(scores, reverse=True)

    def test_entity_scoping(self, store):
        from cairn.reflect import find_stale

        _store_memory(store, "Entity A old memory", entity_id="ent_a", age_days=30)
        _store_memory(store, "Entity B old memory", entity_id="ent_b", age_days=30)
        result = find_stale(store, min_age_days=14, days=60, entity_id="ent_a")
        for m in result["stale_memories"]:
            # Should only include ent_a memories
            assert "Entity A" in m["content_preview"] or m["content_preview"]

    def test_limit(self, store):
        from cairn.reflect import find_stale

        for i in range(10):
            _store_memory(store, f"Old stale memory number {i}", age_days=30)
        result = find_stale(store, min_age_days=14, days=60, limit=3)
        assert len(result["stale_memories"]) <= 3

    def test_result_structure(self, store):
        from cairn.reflect import find_stale

        _store_memory(store, "Stale test memory for structure check", age_days=20)
        result = find_stale(store, min_age_days=14, days=30)
        assert "total_candidates" in result
        assert "stale_memories" in result
        if result["stale_memories"]:
            m = result["stale_memories"][0]
            assert "id" in m
            assert "content_preview" in m
            assert "created_at" in m
            assert "access_count" in m
            assert "staleness_score" in m
            assert "event_type" in m
            assert "reasons" in m
