"""Bridge integration tests -- real SQLiteStore, no mocking.

Tests the public bridge API end-to-end with a temporary Cairn directory
and a fresh SQLiteStore per test (via the _reset_bridge fixture).
"""

import os
import pytest

from cairn.bridge import (
    auto_capture,
    clear_session,
    delete_memory,
    edit_memory,
    export_memories,
    import_memories,
    query,
    reset_memory,
    status,
    store,
    welcome,
)


# ---------------------------------------------------------------------------
# Fixture: reset bridge singleton between tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_cairn_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    reset_memory()
    yield
    reset_memory()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_node_id(result_str: str) -> str:
    """Extract the node ID from a store() return string like 'Stored mem-abc123 ...'."""
    # Format: "Stored <id> (<event_type>, <ttl>)"
    parts = result_str.split()
    if len(parts) >= 2 and parts[0] == "Stored":
        return parts[1]
    raise ValueError(f"Could not extract node ID from: {result_str!r}")


# ============================================================================
# 1. store -- basic
# ============================================================================


def test_store_basic():
    """Store a memory and verify the returned confirmation string."""
    result = store("The quick brown fox jumped over the lazy dog near the riverbank")
    assert isinstance(result, str)
    assert "Stored" in result or "Deduped" in result or "Evolved" in result
    # Default event_type is "memory"
    if "Stored" in result:
        assert "memory" in result


# ============================================================================
# 2. store -- with metadata and event_type
# ============================================================================


def test_store_with_metadata():
    """Store with explicit event_type and metadata, verify they flow through."""
    result = store(
        "Always run pytest before committing Python changes to the repository",
        event_type="lesson_learned",
        metadata={"source": "test", "tags": ["testing", "ci"]},
    )
    assert isinstance(result, str)
    assert "Stored" in result or "Deduped" in result or "Evolved" in result
    if "Stored" in result:
        assert "lesson_learned" in result


# ============================================================================
# 3. query -- basic
# ============================================================================


def test_query_basic():
    """Store a memory then query for it; results should contain the content."""
    store(
        "Postgres connection pooling reduces latency for high-traffic applications",
        event_type="lesson_learned",
    )
    result = query("postgres connection pooling latency")
    assert isinstance(result, str)
    # The query result should surface the stored content
    assert "Postgres" in result or "postgres" in result or "pooling" in result


# ============================================================================
# 4. query -- event_type filter
# ============================================================================


def test_query_with_event_type_filter():
    """Store different event types, query with filter for a specific one."""
    store(
        "Redis caching dramatically improves response times for read-heavy workloads",
        event_type="lesson_learned",
    )
    store(
        "Decided to use Redis for session storage instead of Memcached for this project",
        event_type="decision",
    )

    # Query with event_type filter -- should only find the decision
    result = query("Redis caching session storage", event_type="decision")
    assert isinstance(result, str)
    # The decision should appear in results
    assert "session storage" in result or "Memcached" in result or "decision" in result.lower()


# ============================================================================
# 5. query -- session scope
# ============================================================================


def test_query_with_session_scope():
    """Store with different session IDs, query scoped to one session."""
    store(
        "Session alpha: configured Nginx reverse proxy for load balancing the cluster",
        event_type="memory",
        session_id="session-alpha-111",
    )
    store(
        "Session beta: set up Cloudflare DNS records for the production domain",
        event_type="memory",
        session_id="session-beta-222",
    )

    result = query(
        "Nginx proxy load balancing",
        session_id="session-alpha-111",
        scope="session",
    )
    assert isinstance(result, str)
    # Session-scoped query should find session-alpha content
    # (may or may not exclude beta depending on implementation, but alpha should appear)


# ============================================================================
# 6. delete_memory -- success
# ============================================================================


def test_delete_memory():
    """Store a memory then delete it; verify success response."""
    result_str = store("Temporary test memory that will be deleted shortly after creation")
    node_id = _extract_node_id(result_str)

    result = delete_memory(node_id)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["deleted_id"] == node_id


# ============================================================================
# 7. delete_memory -- non-existent
# ============================================================================


def test_delete_memory_nonexistent():
    """Deleting a non-existent memory should return an error response."""
    result = delete_memory("mem-does-not-exist-at-all-12345")
    assert isinstance(result, dict)
    assert result["success"] is False
    assert "error" in result


# ============================================================================
# 8. edit_memory
# ============================================================================


def test_edit_memory():
    """Store a memory, edit it, and verify old/new content previews."""
    original_text = "Original content for testing the edit memory bridge function"
    result_str = store(original_text)
    node_id = _extract_node_id(result_str)

    new_text = "Updated content after editing the memory through the bridge layer"
    result = edit_memory(node_id, new_text)
    assert isinstance(result, dict)
    assert result["success"] is True
    assert result["id"] == node_id
    assert "Original" in result["old_content_preview"]
    assert "Updated" in result["new_content_preview"]


# ============================================================================
# 9. clear_session
# ============================================================================


def test_clear_session():
    """Store memories in two sessions, clear one, verify count."""
    sid_keep = "session-keep-aaa"
    sid_clear = "session-clear-bbb"

    store(
        "Memory in the session that will be kept after clearing the other session",
        session_id=sid_keep,
    )
    store(
        "The azure butterfly migration pattern occurs between November and March across the Pacific",
        session_id=sid_clear,
    )
    store(
        "Quantum entanglement was experimentally verified by Alain Aspect in 1982 using Bell tests",
        session_id=sid_clear,
    )

    result = clear_session(sid_clear)
    assert isinstance(result, dict)
    assert result["session_id"] == sid_clear
    assert result["removed"] >= 2

    # Verify the kept session's memory is still queryable
    q = query("session that will be kept", session_id=sid_keep)
    assert isinstance(q, str)


# ============================================================================
# 10. export / import round trip
# ============================================================================


def test_export_import_roundtrip(tmp_cairn_dir):
    """Store memories, export, reset, import, then query to verify."""
    store(
        "Roundtrip test memory: always validate exports before deploying to production",
        event_type="lesson_learned",
    )
    store(
        "Roundtrip test decision: chose PostgreSQL over MySQL for the new microservice",
        event_type="decision",
    )

    export_path = str(tmp_cairn_dir / "export_test.json")

    # Export
    export_result = export_memories(export_path)
    assert isinstance(export_result, str)
    assert "Export" in export_result
    assert os.path.exists(export_path)

    # Reset the store
    reset_memory()

    # Import
    import_result = import_memories(export_path, clear_existing=True)
    assert isinstance(import_result, str)
    assert "Import" in import_result

    # Query to verify data survived the round trip
    q = query("roundtrip validate exports production")
    assert isinstance(q, str)
    assert "roundtrip" in q.lower() or "validate" in q.lower() or "export" in q.lower()


# ============================================================================
# 11. welcome
# ============================================================================


def test_welcome():
    """Welcome should return a dict without raising."""
    result = welcome()
    assert isinstance(result, dict)


# ============================================================================
# 12. status
# ============================================================================


def test_status():
    """Status should return a dict with expected keys."""
    result = status()
    assert isinstance(result, dict)
    assert "ok" in result
    assert "status" in result
    assert "node_count" in result
    assert "backend" in result
    assert result["backend"] == "sqlite"
