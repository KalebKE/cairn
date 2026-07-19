"""Backbone Resilience Tests (arxiv 2602.19320 §5.2).

Tests that Cairn gracefully handles malformed inputs that smaller
backbone models produce: malformed JSON metadata, empty content,
invalid event types, oversized payloads, and special characters.

Also verifies the new observability instrumentation:
- Agency tax timing (get_agency_tax)
- Format error rate tracking (get_format_error_rate)
- Maintenance backlog monitoring (get_maintenance_backlog)
"""

from __future__ import annotations

import threading

import pytest

from cairn.exceptions import StorageError


@pytest.fixture
def store(tmp_path):
    """Create a fresh SQLiteStore for testing."""
    from cairn.sqlite_store import SQLiteStore

    return SQLiteStore(db_path=tmp_path / "test.db")


# ------------------------------------------------------------------
# 1. Malformed JSON metadata
# ------------------------------------------------------------------


class TestMalformedJsonMetadata:
    """Verify store handles metadata edge cases without corruption."""

    def test_none_metadata(self, store):
        nid = store.store("test content", metadata=None)
        node = store.get_node(nid)
        assert node is not None
        assert node.content == "test content"

    def test_empty_dict_metadata(self, store):
        nid = store.store("test content", metadata={})
        node = store.get_node(nid)
        assert node is not None

    def test_nested_metadata(self, store):
        meta = {"outer": {"inner": {"deep": [1, 2, 3]}}}
        nid = store.store("nested metadata test", metadata=meta)
        node = store.get_node(nid)
        assert node is not None
        assert node.metadata["outer"]["inner"]["deep"] == [1, 2, 3]

    def test_string_metadata_instead_of_dict(self, store):
        """Smaller models sometimes emit metadata as a raw JSON string."""
        with pytest.raises((TypeError, AttributeError, ValueError)):
            store.store("test", metadata='{"event_type": "decision"}')

    def test_metadata_with_none_values(self, store):
        meta = {"key1": None, "key2": "value", "event_type": "observation"}
        nid = store.store("null value test", metadata=meta)
        node = store.get_node(nid)
        assert node is not None


# ------------------------------------------------------------------
# 2. Empty / whitespace content
# ------------------------------------------------------------------


class TestEmptyContentHandling:
    """Content validation — empty/whitespace should be rejected."""

    def test_empty_string_rejected(self, store):
        with pytest.raises(StorageError, match="non-empty"):
            store.store("")

    def test_whitespace_only_stored(self, store):
        """Whitespace-only content is technically non-empty."""
        nid = store.store("   ")
        node = store.get_node(nid)
        assert node is not None


# ------------------------------------------------------------------
# 3. Invalid event types
# ------------------------------------------------------------------


class TestInvalidEventType:
    """Event type edge cases."""

    def test_unknown_event_type(self, store):
        nid = store.store(
            "unknown type test",
            metadata={"event_type": "completely_unknown_type_xyz"},
        )
        node = store.get_node(nid)
        assert node is not None
        assert node.metadata.get("event_type") == "completely_unknown_type_xyz"

    def test_empty_event_type(self, store):
        nid = store.store("empty event type", metadata={"event_type": ""})
        node = store.get_node(nid)
        assert node is not None


# ------------------------------------------------------------------
# 4. Malformed entity IDs
# ------------------------------------------------------------------


class TestMalformedEntityId:
    """Entity ID validation edge cases."""

    def test_very_long_entity_id(self, store):
        long_id = "x" * 1000
        nid = store.store("long entity test", entity_id=long_id)
        node = store.get_node(nid)
        assert node is not None

    def test_special_chars_entity_id(self, store):
        nid = store.store(
            "special entity test",
            entity_id="user@domain.com/project#1",
        )
        node = store.get_node(nid)
        assert node is not None


# ------------------------------------------------------------------
# 5. Oversized content
# ------------------------------------------------------------------


class TestOversizedContent:
    """Content size limit enforcement."""

    def test_at_size_limit(self, store):
        """Content at max size should be accepted."""
        content = "x" * (store._MAX_CONTENT_SIZE)
        nid = store.store(content)
        assert nid is not None

    def test_over_size_limit(self, store):
        """Content over max size should be rejected."""
        content = "x" * (store._MAX_CONTENT_SIZE + 1)
        with pytest.raises(StorageError, match="exceeds limit"):
            store.store(content)


# ------------------------------------------------------------------
# 6. Special characters
# ------------------------------------------------------------------


class TestSpecialCharsInFields:
    """Unicode, emoji, and SQL injection attempts."""

    def test_unicode_content(self, store):
        content = "日本語テスト 🧠 مرحبا мир"
        nid = store.store(content)
        node = store.get_node(nid)
        assert node is not None
        assert "🧠" in node.content

    def test_sql_injection_in_content(self, store):
        content = "'; DROP TABLE memories; --"
        nid = store.store(content)
        node = store.get_node(nid)
        assert node is not None
        assert node.content == content
        # Table should still exist
        assert store.node_count() >= 1

    def test_null_bytes_in_content(self, store):
        content = "test\x00content\x00with\x00nulls"
        nid = store.store(content)
        node = store.get_node(nid)
        assert node is not None


# ------------------------------------------------------------------
# 7. Concurrent mixed writes
# ------------------------------------------------------------------


class TestConcurrentMixedWrites:
    """Concurrent stores with mixed valid/invalid data."""

    def test_concurrent_stores(self, store):
        """Multiple threads storing simultaneously should not corrupt state."""
        errors = []
        node_ids = []

        def store_one(i):
            try:
                nid = store.store(f"concurrent content {i}")
                node_ids.append(nid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_one, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Errors during concurrent writes: {errors}"
        assert len(node_ids) == 10


# ------------------------------------------------------------------
# 8. Observability instrumentation (Items 3-5)
# ------------------------------------------------------------------


class TestStoreIntegrityAfterErrors:
    """Verify new observability methods work correctly."""

    def test_format_error_rate_tracking(self, store):
        """Format error rate should track errors vs total writes."""
        store.store("write 1")
        store.store("write 2")
        store.record_format_error("store", "malformed JSON")
        store.store("write 3")

        rate = store.get_format_error_rate()
        assert rate == pytest.approx(1 / 3, abs=0.01)

    def test_agency_tax_tracking(self, store):
        """Agency tax should record timing for store operations."""
        store.store("timing test 1")
        store.store("timing test 2")

        tax = store.get_agency_tax()
        assert "write" in tax
        assert tax["write"]["count"] >= 2
        assert tax["write"]["median_ms"] >= 0
        assert tax["write"]["mean_ms"] >= 0

    def test_maintenance_backlog_tracking(self, store):
        """Maintenance backlog should count writes since last consolidation."""
        # Use very distinct content to avoid embedding dedup
        distinct_contents = [
            "The capital of France is Paris and the Eiffel Tower stands tall",
            "Python programming uses indentation for block structure syntax",
            "Quantum mechanics describes subatomic particle wave functions",
            "Renaissance art flourished in Florence during the 15th century",
            "The Pacific Ocean is the largest body of water on Earth today",
        ]
        for content in distinct_contents:
            store.store(content)

        backlog = store.get_maintenance_backlog()
        assert backlog["writes_since_consolidation"] >= 3  # some may dedup
        assert backlog["backlog_critical"] is False

        # Consolidate should reset the counter
        store.consolidate()
        backlog = store.get_maintenance_backlog()
        assert backlog["writes_since_consolidation"] == 0
        assert len(backlog["recent_consolidations"]) >= 1

    def test_health_check_includes_new_metrics(self, store):
        """check_memory_health should include format_error_rate and agency_tax."""
        store.store("health check test")
        health = store.check_memory_health()

        assert "format_error_rate" in health
        assert "agency_tax" in health
        assert "maintenance_backlog" in health
        assert isinstance(health["format_error_rate"], float)
        assert isinstance(health["agency_tax"], dict)
        assert isinstance(health["maintenance_backlog"], dict)
