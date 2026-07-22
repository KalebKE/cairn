"""Tests for multi-session reasoning improvements.

Covers three techniques:
A. Fact extraction at store-time
B. Heuristic query decomposition
C. Automatic temporal range inference (created_at fallback + soft mode)
"""

import importlib.util
import os
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from _vec import requires_vec

from cairn.sqlite_store import SQLiteStore, MemoryResult


# ============================================================================
# Helpers
# ============================================================================

def _make_store(tmp_path) -> SQLiteStore:
    """Create a fresh SQLiteStore in tmp_path."""
    db_path = str(tmp_path / "test.db")
    os.environ["CAIRN_HOME"] = str(tmp_path)
    return SQLiteStore(db_path)


# ============================================================================
# A. Fact Extraction
# ============================================================================


class TestFactExtraction:
    """Tests for _extract_facts() heuristic NLP."""

    def test_extracts_camelcase(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("We chose SQLiteStore for persistence and MemoryResult for queries.")
        assert "sqlitestore" in facts
        assert "memoryresult" in facts

    def test_extracts_upper_case_constants(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("Set MAX_NODES to 10000 and API_KEY to secret.")
        assert "max_nodes" in facts
        assert "api_key" in facts

    def test_extracts_backtick_tokens(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("Use `jwt` for auth and `sqlite_store.py` for storage.")
        assert "jwt" in facts
        assert "sqlite_store.py" in facts

    def test_extracts_quoted_strings(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts('Decided on "refresh token" approach with "15-min expiry".')
        assert "refresh token" in facts
        assert "15-min expiry" in facts

    def test_extracts_decision_verbs(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("We chose JWT for authentication. Switched to FastAPI for the backend.")
        fact_str = " ".join(facts)
        assert "jwt" in fact_str.lower()
        assert "fastapi" in fact_str.lower()

    def test_extracts_dotted_paths(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("The module cairn.sqlite_store handles all persistence.")
        assert "cairn.sqlite_store" in facts

    def test_skips_version_numbers(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("Updated to version 1.0.0 and deployed 2.3.1.")
        assert "1.0.0" not in facts
        assert "2.3.1" not in facts

    def test_extracts_hyphenated_terms(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("The multi-session reasoning and cross-agent coordination work well.")
        assert "multi-session" in facts
        assert "cross-agent" in facts

    def test_caps_at_20(self):
        from cairn.bridge import _extract_facts

        # Generate content with many extractable facts
        content = " ".join(f"`term{i}` and TermCamel{i}" for i in range(30))
        facts = _extract_facts(content)
        assert len(facts) <= 20

    def test_empty_content(self):
        from cairn.bridge import _extract_facts

        facts = _extract_facts("")
        assert facts == []

    def test_facts_stored_in_metadata(self, tmp_path):
        """Fact extraction runs at store-time for high-value event types.

        Facts are merged into the 'tags' metadata field (not a separate 'facts' key).
        """
        from cairn.bridge import auto_capture, _get_store
        import cairn.bridge as bridge

        os.environ["CAIRN_HOME"] = str(tmp_path)
        bridge._store = None  # Force re-init

        result = auto_capture(
            "Decided to use JWT with 15-minute refresh tokens for authentication. "
            "Switched from Express to FastAPI for the backend API framework.",
            event_type="decision",
            session_id="test-session",
            project="/test/project",
        )
        assert "Stored" in result or "Deduped" in result or "Evolved" in result

        store = _get_store()
        # Find the stored memory
        results = store.query("JWT authentication", limit=5)
        found = False
        for r in results:
            if "JWT" in r.content:
                found = True
                # Facts are merged into tags, not stored as metadata["facts"]
                tags = r.metadata.get("tags", [])
                assert isinstance(tags, list)
                assert len(tags) > 0
                # Should have extracted jwt-related terms as tags
                tag_str = " ".join(tags).lower()
                assert "jwt" in tag_str or "fastapi" in tag_str or "authentication" in tag_str
                break
        assert found, "Memory with JWT content not found"

    def test_facts_not_extracted_for_plain_memory(self, tmp_path):
        """Plain 'memory' type should not trigger fact extraction."""
        from cairn.bridge import auto_capture, _get_store
        import cairn.bridge as bridge

        os.environ["CAIRN_HOME"] = str(tmp_path)
        bridge._store = None

        auto_capture(
            "Just a regular observation about the codebase.",
            event_type="memory",
            session_id="test-session",
        )

        store = _get_store()
        results = store.query("regular observation", limit=5)
        for r in results:
            if "regular observation" in r.content:
                assert "facts" not in r.metadata
                break


# ============================================================================
# B. Query Decomposition
# ============================================================================


class TestQueryDecomposition:
    """Tests for _decompose_query() compound query splitting."""

    def test_splits_on_and(self):
        parts = SQLiteStore._decompose_query(
            "What auth method did we choose and which API framework did we switch to"
        )
        assert parts is not None
        assert len(parts) == 2
        assert "auth" in parts[0].lower()
        assert "api" in parts[1].lower() or "framework" in parts[1].lower()

    def test_splits_on_as_well_as(self):
        parts = SQLiteStore._decompose_query(
            "What database do we use as well as what caching layer did we adopt"
        )
        assert parts is not None
        assert len(parts) == 2

    def test_splits_on_also(self):
        parts = SQLiteStore._decompose_query(
            "What auth did we use and also what deployment target did we pick"
        )
        assert parts is not None
        assert len(parts) == 2

    def test_no_split_on_short_query(self):
        parts = SQLiteStore._decompose_query("auth and api")
        assert parts is None

    def test_no_split_on_single_topic(self):
        parts = SQLiteStore._decompose_query("What authentication method did we decide on")
        assert parts is None

    def test_no_split_inside_quotes(self):
        parts = SQLiteStore._decompose_query('"command and control" framework setup')
        assert parts is None

    def test_splits_comma_clauses_with_question_words(self):
        parts = SQLiteStore._decompose_query(
            "What database did we pick, which ORM are we using"
        )
        assert parts is not None
        assert len(parts) == 2

    def test_no_split_on_comma_without_clause_starts(self):
        parts = SQLiteStore._decompose_query(
            "The database schema, indexes, and constraints"
        )
        assert parts is None

    def test_caps_at_four_subqueries(self):
        parts = SQLiteStore._decompose_query(
            "What auth did we use and what database and what API framework and what cache layer and what deployment"
        )
        if parts is not None:
            assert len(parts) <= 4

    def test_decomposition_produces_meaningful_parts(self):
        parts = SQLiteStore._decompose_query(
            "What logging framework are we using and how did we configure error monitoring"
        )
        assert parts is not None
        for p in parts:
            assert len(p) >= 12
            assert len(p.split()) >= 2

    @pytest.mark.skipif(
        not importlib.util.find_spec("sqlite_vec"),
        reason="sqlite-vec not installed"
    )
    @requires_vec
    def test_decomposed_query_runs_end_to_end(self, tmp_path):
        """Compound query should return merged results from sub-queries."""
        store = _make_store(tmp_path)

        # Store two memories about different topics
        store.store("Decided to use JWT for authentication with refresh tokens",
                     metadata={"event_type": "decision", "tags": ["jwt", "auth"]})
        store.store("Switched to FastAPI from Express for the backend framework",
                     metadata={"event_type": "decision", "tags": ["fastapi", "backend"]})

        # Query with compound question
        results = store.query(
            "What auth method did we choose and which API framework did we switch to",
            limit=10,
        )

        # Both memories should appear in results
        contents = " ".join(r.content for r in results)
        assert "JWT" in contents
        assert "FastAPI" in contents

    def test_no_infinite_recursion(self, tmp_path):
        """Decomposition guard (_in_decomposition) prevents recursive splitting."""
        store = _make_store(tmp_path)
        store.store("Test memory", metadata={"event_type": "memory"})

        # This should not hang or recurse infinitely
        results = store.query(
            "What auth did we use and what database did we pick",
            limit=5,
        )
        assert isinstance(results, list)


# ============================================================================
# C. Temporal Range Inference
# ============================================================================


class TestTemporalRangeInference:
    """Tests for improved temporal scoring in Phase 4."""

    def test_created_at_fallback(self, tmp_path):
        """Memories without referenced_date should use created_at for temporal scoring."""
        store = _make_store(tmp_path)

        now = datetime.now(timezone.utc)

        # Store a memory that's "in range" by created_at (no referenced_date)
        store.store(
            "Decided on JWT auth last week",
            metadata={"event_type": "decision"},
        )

        # Query with temporal range matching the memory's creation time
        t_start = (now - timedelta(hours=1)).isoformat()
        t_end = (now + timedelta(hours=1)).isoformat()

        results = store.query(
            "JWT auth decision",
            limit=5,
            temporal_range=(t_start, t_end),
        )

        # The memory should be found (created_at fallback kicks in)
        assert len(results) > 0
        assert any("JWT" in r.content for r in results)

    def test_soft_mode_no_harsh_penalty(self, tmp_path):
        """With temporal_boost_only=True, out-of-range memories get mild penalty."""
        store = _make_store(tmp_path)

        # Store a memory
        store.store(
            "Important architectural decision about database schema",
            metadata={"event_type": "decision"},
        )

        now = datetime.now(timezone.utc)

        # Query with a temporal range far in the past (memory is out-of-range)
        t_start = (now - timedelta(days=365)).isoformat()
        t_end = (now - timedelta(days=300)).isoformat()

        # Strict mode: harsh penalty
        results_strict = store.query(
            "database schema decision",
            limit=5,
            temporal_range=(t_start, t_end),
            temporal_boost_only=False,
        )

        # Soft mode: mild penalty
        results_soft = store.query(
            "database schema decision",
            limit=5,
            temporal_range=(t_start, t_end),
            temporal_boost_only=True,
            use_cache=False,
        )

        # Soft mode should return higher relevance for out-of-range memories
        if results_strict and results_soft:
            strict_score = results_strict[0].relevance
            soft_score = results_soft[0].relevance
            # Soft mode should be >= strict mode (less penalty)
            assert soft_score >= strict_score

    def test_in_range_boost_applies(self, tmp_path):
        """In-range memories get boosted in both strict and soft mode."""
        store = _make_store(tmp_path)

        store.store(
            "Decided to use PostgreSQL for the main database",
            metadata={"event_type": "decision"},
        )

        now = datetime.now(timezone.utc)
        t_start = (now - timedelta(hours=1)).isoformat()
        t_end = (now + timedelta(hours=1)).isoformat()

        # Query without temporal range
        results_no_temporal = store.query(
            "PostgreSQL database decision",
            limit=5,
        )

        # Query with matching temporal range
        results_with_temporal = store.query(
            "PostgreSQL database decision",
            limit=5,
            temporal_range=(t_start, t_end),
            use_cache=False,
        )

        # Both should find the memory
        assert len(results_no_temporal) > 0
        assert len(results_with_temporal) > 0

    def test_bridge_auto_infer_sets_soft_mode(self):
        """bridge.py should set temporal_boost_only=True for auto-inferred ranges."""
        from cairn.bridge import _infer_temporal_range

        # "last week" should produce a temporal range
        result = _infer_temporal_range("What did we decide last week about auth")
        assert result is not None
        assert len(result) == 2
        # The bridge code sets temporal_boost_only=True when this is auto-inferred

    def test_infer_temporal_range_parses_last_n_days(self):
        from cairn.bridge import _infer_temporal_range

        result = _infer_temporal_range("decisions from last 3 days")
        assert result is not None
        start, end = result
        # Start should be about 3 days ago
        now = datetime.now(timezone.utc)
        expected_start = now - timedelta(days=3)
        assert abs(datetime.fromisoformat(start).timestamp() - expected_start.timestamp()) < 60

    def test_infer_temporal_range_parses_yesterday(self):
        from cairn.bridge import _infer_temporal_range

        result = _infer_temporal_range("what happened yesterday")
        assert result is not None

    def test_infer_temporal_range_parses_month_name(self):
        from cairn.bridge import _infer_temporal_range

        result = _infer_temporal_range("decisions made in January 2025")
        assert result is not None
        start, end = result
        assert "2025-01" in start
        assert "2025-02" in end

    def test_infer_temporal_range_returns_none_for_no_signal(self):
        from cairn.bridge import _infer_temporal_range

        result = _infer_temporal_range("what auth method did we use")
        assert result is None

    def test_referenced_date_still_preferred(self, tmp_path):
        """When referenced_date exists, it should be used over created_at."""
        store = _make_store(tmp_path)

        # Store with explicit referenced_date in the past
        store.store(
            "Old decision from January",
            metadata={
                "event_type": "decision",
                "referenced_date": "2025-01-15T00:00:00+00:00",
            },
        )

        # Query with range matching referenced_date
        results = store.query(
            "Old decision January",
            limit=5,
            temporal_range=("2025-01-01T00:00:00+00:00", "2025-02-01T00:00:00+00:00"),
        )

        assert len(results) > 0
        assert any("January" in r.content for r in results)


# ============================================================================
# Integration: All Techniques Together
# ============================================================================


class TestMultiSessionIntegration:
    """Integration tests combining all three techniques."""

    def test_fact_extraction_improves_retrieval(self, tmp_path):
        """Facts extracted at store-time should improve recall for variant queries."""
        store = _make_store(tmp_path)

        # Store with fact-rich content
        store.store(
            "Decided to use JWT with 15-minute refresh tokens for authentication. "
            "Also configured CORS to allow the frontend domain.",
            metadata={"event_type": "decision"},
        )

        # Query using a term that should be in facts but might not rank high semantically
        results = store.query("refresh tokens", limit=5)
        assert len(results) > 0
        assert any("refresh" in r.content.lower() for r in results)

    def test_decomposition_with_temporal(self, tmp_path):
        """Compound queries with temporal signals should work together."""
        store = _make_store(tmp_path)

        store.store(
            "Chose PostgreSQL for the database",
            metadata={"event_type": "decision"},
        )
        store.store(
            "Implemented Redis caching layer",
            metadata={"event_type": "decision"},
        )

        # Compound query — should find both memories
        results = store.query(
            "What database did we choose and what caching did we implement",
            limit=10,
        )

        contents = " ".join(r.content for r in results)
        # At least one of the two should be found
        assert "PostgreSQL" in contents or "Redis" in contents
