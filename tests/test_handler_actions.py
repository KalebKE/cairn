"""Tests for handler actions added/updated in v0.11.0.

Covers:
  - handle_cairn_reflect: pro-only module graceful fallback
  - cairn_memory action=link: manual edge creation
  - cairn_memory action=flagged: flagged memory listing
  - cairn_memory action=supersede: manual supersession
  - cairn_stats action=forgetting_log: pro-only graceful fallback
  - cairn_stats action=dedup: dedup stats
  - cairn_stats action=milestones: milestone progress
  - handle_cairn_browse: browse by type/session/recent
"""
from unittest.mock import patch

import pytest

from cairn.server.handlers import (
    HANDLERS,
    handle_cairn_reflect,
    handle_cairn_memory,
    handle_cairn_stats,
    handle_cairn_browse,
)
from cairn.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path):
    """Real SQLiteStore in a temp directory."""
    db_path = str(tmp_path / "test.db")
    s = SQLiteStore(db_path)
    return s


@pytest.fixture
def mock_get_store(store):
    """Patch _get_store to return our real SQLiteStore."""
    with patch("cairn.bridge._get_store", return_value=store):
        yield store


# ---------------------------------------------------------------------------
# cairn_reflect
# ---------------------------------------------------------------------------


class TestCairnReflect:
    """Tests for handle_cairn_reflect — core module."""

    @pytest.mark.asyncio
    async def test_handler_in_registry(self):
        assert "cairn_reflect" in HANDLERS

    @pytest.mark.asyncio
    async def test_stale_action_succeeds(self):
        """cairn.reflect is core — stale action should work."""
        result = await handle_cairn_reflect({"action": "stale"})
        assert not result.get("isError")

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self):
        """Unknown action should return an error."""
        result = await handle_cairn_reflect({"action": "bogus"})
        assert result.get("isError")


# ---------------------------------------------------------------------------
# cairn_memory action=link
# ---------------------------------------------------------------------------


class TestCairnMemoryLink:
    @pytest.mark.asyncio
    async def test_link_success(self, mock_get_store):
        store = mock_get_store
        # Store two memories
        id1 = store.store("Memory A", metadata={"event_type": "decision"})
        id2 = store.store("Memory B", metadata={"event_type": "decision"})

        result = await handle_cairn_memory({
            "action": "link", "memory_id": id1, "target_id": id2, "edge_type": "related"
        })
        assert not result.get("isError")
        assert "Linked" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_link_missing_memory_id(self, mock_get_store):
        result = await handle_cairn_memory({
            "action": "link", "memory_id": "", "target_id": "some-id"
        })
        assert result.get("isError")
        assert "memory_id" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_link_missing_target_id(self, mock_get_store):
        result = await handle_cairn_memory({
            "action": "link", "memory_id": "some-id"
        })
        assert result.get("isError")
        assert "target_id" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# cairn_memory action=flagged
# ---------------------------------------------------------------------------


class TestCairnMemoryFlagged:
    @pytest.mark.asyncio
    async def test_no_flagged(self, mock_get_store):
        result = await handle_cairn_memory({"action": "flagged"})
        assert not result.get("isError")
        assert "No memories flagged" in result["content"][0]["text"] or "No flagged" in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_with_flagged_memories(self, mock_get_store):
        store = mock_get_store
        node_id = store.store("Bad memory", metadata={"event_type": "decision", "feedback_score": -5})
        result = await handle_cairn_memory({"action": "flagged"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "Flagged" in text or "score=" in text


# ---------------------------------------------------------------------------
# cairn_memory action=supersede
# ---------------------------------------------------------------------------


class TestCairnMemorySupersede:
    @pytest.mark.asyncio
    async def test_supersede_success(self, mock_get_store):
        store = mock_get_store
        id1 = store.store("Old decision", metadata={"event_type": "decision"})
        id2 = store.store("New decision", metadata={"event_type": "decision"})

        result = await handle_cairn_memory({
            "action": "supersede", "memory_id": id2, "target_id": id1
        })
        assert not result.get("isError")
        assert "superseded" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_supersede_missing_memory_id(self, mock_get_store):
        result = await handle_cairn_memory({
            "action": "supersede", "memory_id": "", "target_id": "some-id"
        })
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_supersede_missing_target_id(self, mock_get_store):
        result = await handle_cairn_memory({
            "action": "supersede", "memory_id": "some-id"
        })
        assert result.get("isError")


# ---------------------------------------------------------------------------
# cairn_stats action=forgetting_log
# ---------------------------------------------------------------------------


class TestCairnStatsForgettingLog:
    @pytest.mark.asyncio
    async def test_forgetting_log_works(self, mock_get_store):
        """forgetting_log action should work in core build."""
        result = await handle_cairn_stats({"action": "forgetting_log"})
        assert not result.get("isError")


# ---------------------------------------------------------------------------
# cairn_stats action=dedup
# ---------------------------------------------------------------------------


class TestCairnStatsDedup:
    @pytest.mark.asyncio
    async def test_dedup_stats(self, mock_get_store):
        store = mock_get_store
        store.store("Memory 1", metadata={"event_type": "decision"})
        store.store("Memory 2", metadata={"event_type": "decision"})

        result = await handle_cairn_stats({"action": "dedup"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "dedup" in text.lower()
        assert "0" in text  # dedup counters present


# ---------------------------------------------------------------------------
# cairn_stats action=milestones
# ---------------------------------------------------------------------------


class TestCairnStatsMilestones:
    @pytest.mark.asyncio
    async def test_milestones_empty(self, mock_get_store):
        result = await handle_cairn_stats({"action": "milestones"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert "milestone" in text.lower() or "streak" in text.lower()

    @pytest.mark.asyncio
    async def test_milestones_with_data(self, mock_get_store):
        store = mock_get_store
        # Use very distinct content to avoid dedup
        topics = [
            "We chose PostgreSQL for the orders database",
            "Authentication uses JWT tokens not sessions",
            "Frontend built with React and TypeScript strict mode",
            "Deployed on AWS ECS with Fargate launch type",
            "Monitoring uses Datadog with custom dashboards",
        ]
        for topic in topics:
            store.store(topic, metadata={"event_type": "decision"})

        result = await handle_cairn_stats({"action": "milestones"})
        text = result["content"][0]["text"]
        assert "milestone" in text.lower() or "streak" in text.lower()


# ---------------------------------------------------------------------------
# cairn_browse
# ---------------------------------------------------------------------------


class TestCairnBrowse:
    @pytest.mark.asyncio
    async def test_browse_recent_empty(self, mock_get_store):
        result = await handle_cairn_browse({"browse_by": "recent"})
        assert not result.get("isError")
        assert "memor" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_browse_recent_with_data(self, mock_get_store):
        store = mock_get_store
        store.store("First memory", metadata={"event_type": "decision"})
        store.store("Second memory", metadata={"event_type": "lesson_learned"})

        result = await handle_cairn_browse({"browse_by": "recent"})
        text = result["content"][0]["text"]
        assert "memor" in text.lower()

    @pytest.mark.asyncio
    async def test_browse_by_type(self, mock_get_store):
        store = mock_get_store
        store.store("Decision 1", metadata={"event_type": "decision"})
        store.store("Decision 2", metadata={"event_type": "decision"})
        store.store("Lesson 1", metadata={"event_type": "lesson_learned"})

        result = await handle_cairn_browse({"browse_by": "type"})
        text = result["content"][0]["text"]
        assert "type" in text.lower() or "decision" in text

    @pytest.mark.asyncio
    async def test_browse_respects_limit(self, mock_get_store):
        store = mock_get_store
        for i in range(10):
            store.store(f"Memory {i}", metadata={"event_type": "decision"})

        result = await handle_cairn_browse({"browse_by": "recent", "limit": 3})
        text = result["content"][0]["text"]
        lines = [l for l in text.split("\n") if l.strip().startswith("[")]
        assert len(lines) <= 3

    @pytest.mark.asyncio
    async def test_browse_by_session(self, mock_get_store):
        store = mock_get_store
        store.store("Sess memory", metadata={"event_type": "decision", "session_id": "sess-abc123"})

        result = await handle_cairn_browse({"browse_by": "session", "session_id": "sess-abc123"})
        text = result["content"][0]["text"]
        assert "session" in text.lower()


# ---------------------------------------------------------------------------
# cairn_stats unknown action
# ---------------------------------------------------------------------------


class TestCairnStatsUnknown:
    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_get_store):
        result = await handle_cairn_stats({"action": "bogus"})
        assert result.get("isError")
        assert "Unknown" in result["content"][0]["text"]


class TestCairnMemoryUnknown:
    @pytest.mark.asyncio
    async def test_unknown_action(self, mock_get_store):
        result = await handle_cairn_memory({"action": "bogus"})
        assert result.get("isError")
        assert "Unknown" in result["content"][0]["text"]


# ---------------------------------------------------------------------------
# cairn_memory action=get — fetch-by-id hydration (full or prefix id).
# Completes the pointers-not-payloads loop: surfacing blocks and context
# packets print 8/12-char id prefixes; get resolves them to full content.
# ---------------------------------------------------------------------------


class TestGetNodeByPrefix:
    """Store-level prefix resolver."""

    def test_exact_id(self, store):
        nid = store.store("Prefix resolver memory", metadata={"event_type": "decision"})
        node, candidates = store.get_node_by_prefix(nid)
        assert node is not None and node.id == nid
        assert candidates == []

    def test_unique_prefix(self, store):
        nid = store.store("Unique prefix memory", metadata={"event_type": "decision"})
        node, candidates = store.get_node_by_prefix(nid[:8])
        assert node is not None and node.id == nid
        assert candidates == []

    def test_ambiguous_prefix_lists_candidates(self, store):
        id1 = store.store("Ambiguous memory one about rollup markers",
                          metadata={"event_type": "decision"})
        id2 = store.store("Ambiguous memory two about socket watchdogs",
                          metadata={"event_type": "decision"})
        with store._lock:
            store._conn.execute("UPDATE memories SET node_id = ? WHERE node_id = ?",
                                ("collidee-aaaa-1111", id1))
            store._conn.execute("UPDATE memories SET node_id = ? WHERE node_id = ?",
                                ("collidee-aaaa-2222", id2))
            store._commit()
        node, candidates = store.get_node_by_prefix("collidee-")
        assert node is None
        assert set(candidates) == {"collidee-aaaa-1111", "collidee-aaaa-2222"}

    def test_missing_id(self, store):
        assert store.get_node_by_prefix("deadbeefcafe") == (None, [])

    def test_short_prefix_rejected(self, store):
        nid = store.store("Short prefix memory", metadata={"event_type": "decision"})
        assert store.get_node_by_prefix(nid[:3]) == (None, [])

    def test_access_bumped_only_on_unique_hit(self, store):
        nid = store.store("Access bump memory", metadata={"event_type": "decision"})
        before = store.get_node(nid, track_access=False).access_count
        store.get_node_by_prefix(nid[:10])
        after = store.get_node(nid, track_access=False).access_count
        assert after == before + 1
        store.get_node_by_prefix("deadbeefcafe")  # miss: no bump anywhere
        assert store.get_node(nid, track_access=False).access_count == after


class TestCairnMemoryGet:
    @pytest.mark.asyncio
    async def test_get_exact_id(self, mock_get_store):
        store = mock_get_store
        long_content = "Decision: keep the flock marker semantics. " * 20
        nid = store.store(long_content, metadata={"event_type": "decision",
                                                  "tags": ["architecture"]})
        result = await handle_cairn_memory({"action": "get", "memory_id": nid})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert long_content.strip() in text, "content must be returned untruncated"
        assert "decision" in text

    @pytest.mark.asyncio
    async def test_get_unique_prefix(self, mock_get_store):
        store = mock_get_store
        nid = store.store("Prefix-resolved get", metadata={"event_type": "lesson_learned"})
        result = await handle_cairn_memory({"action": "get", "memory_id": nid[:8]})
        assert not result.get("isError")
        assert nid in result["content"][0]["text"]

    @pytest.mark.asyncio
    async def test_get_ambiguous_prefix_errors_with_candidates(self, mock_get_store):
        store = mock_get_store
        id1 = store.store("First collider content", metadata={"event_type": "decision"})
        id2 = store.store("Second collider content", metadata={"event_type": "decision"})
        with store._lock:
            store._conn.execute("UPDATE memories SET node_id = ? WHERE node_id = ?",
                                ("clash000-aaaa", id1))
            store._conn.execute("UPDATE memories SET node_id = ? WHERE node_id = ?",
                                ("clash000-bbbb", id2))
            store._commit()
        result = await handle_cairn_memory({"action": "get", "memory_id": "clash000"})
        assert result.get("isError")
        text = result["content"][0]["text"]
        assert "clash000-aaaa" in text and "clash000-bbbb" in text

    @pytest.mark.asyncio
    async def test_get_missing_id(self, mock_get_store):
        result = await handle_cairn_memory({"action": "get", "memory_id": "deadbeefcafe"})
        assert result.get("isError")
        assert "not found" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_get_requires_memory_id(self, mock_get_store):
        result = await handle_cairn_memory({"action": "get"})
        assert result.get("isError")

    @pytest.mark.asyncio
    async def test_get_includes_edges(self, mock_get_store):
        store = mock_get_store
        id1 = store.store("Edge source memory", metadata={"event_type": "decision"})
        id2 = store.store("Edge target memory", metadata={"event_type": "decision"})
        store.add_edge(id1, id2, edge_type="related", weight=0.9)

        import json as _json

        result = await handle_cairn_memory({"action": "get", "memory_id": id1})
        assert not result.get("isError")
        payload = _json.loads(result["content"][0]["text"])
        assert any(e["other_id"] == id2 for e in payload["edges"])

        bare = await handle_cairn_memory({
            "action": "get", "memory_id": id1, "include_related": False,
        })
        assert "edges" not in _json.loads(bare["content"][0]["text"])
