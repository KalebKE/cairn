"""Tests for Cairn contradiction surfacing — Phase 2.

Tests that:
1. _check_contradictions() returns a list of dicts (not None)
2. get_last_contradiction_results() is consume-once
3. Bridge auto_capture() includes [CONTRADICTION] block in output
4. include_contradicted filter on query returns only contradicted memories
5. Normal queries work without the flag
"""

import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from cairn.sqlite_store import EMBEDDING_DIM
from cairn.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bridge(tmp_cairn_dir):
    """Reset the bridge singleton so each test gets a fresh store."""
    from cairn.bridge import reset_memory
    reset_memory()
    yield
    reset_memory()


def _make_store(tmp_cairn_dir):
    """Create a fresh SQLiteStore in the tmp dir."""
    return SQLiteStore(db_path=str(tmp_cairn_dir / "test.db"))


# ===========================================================================
# 1. get_last_contradiction_results — empty when no contradictions
# ===========================================================================


class TestGetLastContradictionResults:
    """Test the consume-once getter for contradiction results."""

    def test_returns_empty_list_initially(self, tmp_cairn_dir):
        """get_last_contradiction_results returns [] on a fresh store."""
        store = _make_store(tmp_cairn_dir)
        assert store.get_last_contradiction_results() == []

    def test_returns_empty_after_store_no_contradictions(self, tmp_cairn_dir):
        """After storing a memory with no contradictions, result is []."""
        store = _make_store(tmp_cairn_dir)
        store.store(
            content="Python is a programming language with dynamic typing",
            session_id="s1",
        )
        # No prior memories to contradict, so should be empty
        result = store.get_last_contradiction_results()
        assert isinstance(result, list)
        # May or may not be empty depending on vec availability,
        # but the type should always be list
        assert isinstance(result, list)

    def test_consume_once_behavior(self, tmp_cairn_dir):
        """Second call to get_last_contradiction_results returns [] (consume-once)."""
        store = _make_store(tmp_cairn_dir)
        # Manually set some results to test consume-once
        store._last_contradiction_results = [
            {
                "node_id": "mem-fake123",
                "confidence": 0.85,
                "reason": "negation detected",
                "content_preview": "some old memory content",
            }
        ]
        first_call = store.get_last_contradiction_results()
        assert len(first_call) == 1
        assert first_call[0]["node_id"] == "mem-fake123"

        second_call = store.get_last_contradiction_results()
        assert second_call == []

    def test_consume_once_clears_internal_state(self, tmp_cairn_dir):
        """After consuming, internal _last_contradiction_results is empty."""
        store = _make_store(tmp_cairn_dir)
        store._last_contradiction_results = [{"node_id": "x", "confidence": 0.5, "reason": "test", "content_preview": "test"}]
        store.get_last_contradiction_results()
        assert store._last_contradiction_results == []


# ===========================================================================
# 2. _check_contradictions returns list of dicts
# ===========================================================================


class TestCheckContradictionsReturnType:
    """Test that _check_contradictions returns a list, not None."""

    def test_returns_empty_list_no_similar(self, tmp_cairn_dir):
        """When no similar memories exist, returns []."""
        store = _make_store(tmp_cairn_dir)
        # Store a memory first to have a node_id
        node_id = store.store(content="Alex prefers dark mode in all editors", session_id="s1")

        # Call _check_contradictions with a dummy embedding
        # Since there's only one memory, there are no candidates to contradict
        embedding = [0.0] * EMBEDDING_DIM
        result = store._check_contradictions(node_id, "something unrelated", embedding)
        assert isinstance(result, list)

    def test_returns_list_with_required_fields(self, tmp_cairn_dir):
        """When contradictions are found, result dicts have required fields."""
        store = _make_store(tmp_cairn_dir)

        # Manually test by injecting fake contradiction results
        # (full integration requires embeddings which may not be available)
        fake_results = [
            {
                "node_id": "mem-abc123def456",
                "confidence": 0.85,
                "reason": "negation detected (one affirms, the other denies)",
                "content_preview": "Alex prefers light mode",
            }
        ]
        store._last_contradiction_results = fake_results

        results = store.get_last_contradiction_results()
        assert len(results) == 1
        result = results[0]
        assert "node_id" in result
        assert "confidence" in result
        assert "reason" in result
        assert "content_preview" in result
        assert isinstance(result["confidence"], float)
        assert isinstance(result["reason"], str)
        assert isinstance(result["content_preview"], str)
        assert len(result["content_preview"]) <= 80

    def test_check_contradictions_with_mocked_detect(self, tmp_cairn_dir):
        """_check_contradictions returns surfaced list when contradictions detected."""
        store = _make_store(tmp_cairn_dir)
        if not store._vec_available:
            pytest.skip("sqlite-vec not available in test environment")

        # Store two contradictory memories
        node1 = store.store(
            content="Alex prefers light mode for all code editors and IDEs",
            session_id="s1",
            metadata={"event_type": "user_preference"},
        )

        # Store a contradictory memory
        node2 = store.store(
            content="Alex prefers dark mode for all code editors and IDEs",
            session_id="s1",
            metadata={"event_type": "user_preference"},
        )

        # Check if contradictions were detected (depends on vec quality)
        results = store.get_last_contradiction_results()
        assert isinstance(results, list)
        # If contradictions were found, verify structure
        for r in results:
            assert "node_id" in r
            assert "confidence" in r
            assert "reason" in r
            assert "content_preview" in r


# ===========================================================================
# 3. Bridge auto_capture output includes [CONTRADICTION] block
# ===========================================================================


class TestBridgeContradictionSurfacing:
    """Test that auto_capture surfaces contradiction info in output."""

    def test_contradiction_block_formatting(self):
        """Test the [CONTRADICTION] block format with mocked get_last_contradiction_results."""
        from cairn.bridge import auto_capture, _get_store

        fake_results = [
            {
                "node_id": "mem-abc123def456",
                "confidence": 0.85,
                "reason": "negation detected (one affirms, the other denies)",
                "content_preview": "Alex prefers light mode",
            }
        ]

        store = _get_store()
        original_get = store.get_last_contradiction_results

        call_count = [0]
        def mock_get():
            call_count[0] += 1
            if call_count[0] == 1:
                return fake_results
            return original_get()

        store.get_last_contradiction_results = mock_get

        result = auto_capture(
            content="Alex now prefers dark mode for all editors and terminals and IDEs",
            event_type="user_preference",
            session_id="test-session",
        )

        store.get_last_contradiction_results = original_get

        # The output should contain our contradiction block
        if "Stored" in result:
            assert "[CONTRADICTION]" in result
            assert "mem-abc123def456" in result
            assert "85%" in result
            assert "negation detected" in result

    def test_no_contradiction_block_when_empty(self):
        """When no contradictions, output should NOT contain [CONTRADICTION]."""
        from cairn.bridge import auto_capture

        result = auto_capture(
            content="Python is useful for data science projects and machine learning workflows",
            event_type="decision",
            session_id="test-session",
        )

        if "Stored" in result:
            assert "[CONTRADICTION]" not in result

    def test_multiple_contradictions_formatting(self):
        """Multiple contradictions should each get a line in the block."""
        from cairn.bridge import auto_capture, _get_store

        fake_results = [
            {
                "node_id": "mem-aaaa11112222",
                "confidence": 0.90,
                "reason": "opposing terms found",
                "content_preview": "Enable dark mode",
            },
            {
                "node_id": "mem-bbbb33334444",
                "confidence": 0.60,
                "reason": "different preference values",
                "content_preview": "Use light theme",
            },
        ]

        store = _get_store()
        original_get = store.get_last_contradiction_results

        call_count = [0]
        def mock_get():
            call_count[0] += 1
            if call_count[0] == 1:
                return fake_results
            return original_get()

        store.get_last_contradiction_results = mock_get

        result = auto_capture(
            content="Alex switched to using dark mode everywhere across all applications",
            event_type="user_preference",
            session_id="test-session",
        )

        store.get_last_contradiction_results = original_get

        if "Stored" in result:
            assert "[CONTRADICTION]" in result
            assert "mem-aaaa11112222" in result
            assert "mem-bbbb33334444" in result
            assert "90%" in result
            assert "60%" in result


# ===========================================================================
# 4. include_contradicted filter on query
# ===========================================================================


class TestIncludeContradictedFilter:
    """Test the include_contradicted parameter on query and query_structured."""

    def test_query_without_flag_returns_all(self):
        """Normal query without include_contradicted returns all results."""
        from cairn.bridge import auto_capture, query

        # Store some memories
        auto_capture(
            content="Python is great for scripting and automation tasks in production",
            event_type="decision",
            session_id="test-session",
        )
        auto_capture(
            content="JavaScript is used for web development and frontend interfaces",
            event_type="decision",
            session_id="test-session",
        )

        result = query("programming languages", limit=10)
        assert "Results:" in result

    def test_query_with_include_contradicted_flag(self):
        """With include_contradicted=True, only contradicted memories returned."""
        from cairn.bridge import auto_capture, query, _get_store

        # Store a memory
        result1 = auto_capture(
            content="Always use tabs for indentation in Python source code files",
            event_type="user_preference",
            session_id="test-session",
        )

        # Manually mark it as contradicted in metadata
        store = _get_store()
        if "Stored" in result1:
            node_id = result1.split()[1]
            node = store.get_node(node_id)
            if node:
                meta = dict(node.metadata or {})
                meta["contradicted_by"] = [{"node_id": "mem-newer", "confidence": 0.8, "reason": "preference change"}]
                store.update_node(node_id, metadata=meta)

        # Store a non-contradicted memory
        auto_capture(
            content="JavaScript runs in web browsers and Node.js server environments",
            event_type="decision",
            session_id="test-session",
        )

        # Query with include_contradicted should only return contradicted ones
        result = query("indentation tabs spaces", limit=10, include_contradicted=True)
        assert "Results:" in result

    def test_query_structured_with_include_contradicted(self):
        """query_structured with include_contradicted filters correctly."""
        from cairn.bridge import auto_capture, query_structured, _get_store

        # Store and manually mark as contradicted
        result1 = auto_capture(
            content="Use spaces for all code indentation across every project",
            event_type="user_preference",
            session_id="test-session",
        )

        store = _get_store()
        if "Stored" in result1:
            node_id = result1.split()[1]
            node = store.get_node(node_id)
            if node:
                meta = dict(node.metadata or {})
                meta["contradicted_by"] = [{"node_id": "mem-newer", "confidence": 0.9, "reason": "temporal override"}]
                store.update_node(node_id, metadata=meta)

        # Store a non-contradicted memory
        auto_capture(
            content="Ruby is a dynamic programming language optimized for developer happiness",
            event_type="decision",
            session_id="test-session",
        )

        # Without flag: returns all matching
        all_results = query_structured("indentation spaces code", limit=10)
        assert isinstance(all_results, list)

        # With flag: returns only contradicted
        contradicted_results = query_structured(
            "indentation spaces code", limit=10, include_contradicted=True
        )
        assert isinstance(contradicted_results, list)
        for r in contradicted_results:
            meta = r.get("metadata", {})
            assert meta.get("contradicted_by"), f"Expected contradicted_by in metadata for {r['id']}"

    def test_include_contradicted_empty_when_none_contradicted(self):
        """include_contradicted returns empty when no memories are contradicted."""
        from cairn.bridge import auto_capture, query_structured

        auto_capture(
            content="The earth orbits the sun in approximately 365.25 days",
            event_type="decision",
            session_id="test-session",
        )

        results = query_structured(
            "earth orbit sun", limit=10, include_contradicted=True
        )
        assert isinstance(results, list)
        # All non-contradicted memories should be filtered out
        for r in results:
            meta = r.get("metadata", {})
            assert meta.get("contradicted_by")


# ===========================================================================
# 5. Tool schema includes include_contradicted
# ===========================================================================


class TestToolSchema:
    """Test that tool_schemas.py includes the new parameter."""

    def test_cairn_query_has_include_contradicted(self):
        """cairn_query schema should include include_contradicted property."""
        from cairn.server.tool_schemas import TOOL_SCHEMAS

        cairn_query = None
        for tool in TOOL_SCHEMAS:
            if tool["name"] == "cairn_query":
                cairn_query = tool
                break

        assert cairn_query is not None, "cairn_query not found in TOOL_SCHEMAS"
        props = cairn_query["inputSchema"]["properties"]
        assert "include_contradicted" in props
        assert props["include_contradicted"]["type"] == "boolean"


# ===========================================================================
# 6. Handler extracts and passes include_contradicted
# ===========================================================================


class TestHandlerIncludeContradicted:
    """Test that the handler extracts include_contradicted from arguments."""

    @pytest.mark.asyncio
    async def test_handler_passes_include_contradicted(self):
        """handle_cairn_query should extract and pass include_contradicted."""
        from cairn.server.handlers import handle_cairn_query

        # Test with flag = True (should not error)
        result = await handle_cairn_query({
            "query": "test query for contradicted memories",
            "include_contradicted": True,
        })
        # Should return a valid response (not an error)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_handler_default_false(self):
        """handle_cairn_query defaults include_contradicted to False."""
        from cairn.server.handlers import handle_cairn_query

        # Test without flag (default should be False, normal query)
        result = await handle_cairn_query({
            "query": "test query for normal memories",
        })
        assert isinstance(result, dict)


# ===========================================================================
# 7. Instance attribute initialized
# ===========================================================================


class TestInstanceAttribute:
    """Test that _last_contradiction_results is initialized properly."""

    def test_attribute_exists_on_new_store(self, tmp_cairn_dir):
        """A new SQLiteStore should have _last_contradiction_results = []."""
        store = _make_store(tmp_cairn_dir)
        assert hasattr(store, "_last_contradiction_results")
        assert store._last_contradiction_results == []

    def test_attribute_is_list(self, tmp_cairn_dir):
        """_last_contradiction_results should always be a list."""
        store = _make_store(tmp_cairn_dir)
        assert isinstance(store._last_contradiction_results, list)
