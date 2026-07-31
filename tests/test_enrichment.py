"""Tests for doc2query write-time enrichment (cairn.enrichment).

All LLM calls are mocked — these tests must pass keyless and hermetic.
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from cairn.enrichment import (
    ENRICH_ELIGIBLE_TYPES,
    MIN_CONTENT_CHARS,
    enrich_pending,
    generate_anticipated_queries,
)

GARDEN_MEMORY = (
    "User maintains a backyard vegetable garden: tomatoes, basil, zucchini "
    "and bell peppers, harvested through late summer. Started composting "
    "kitchen scraps this year to feed the beds."
)
GARDEN_QUERIES = [
    "what should I cook for dinner with my homegrown ingredients",
    "meal ideas using produce I grew myself",
    "recipes with fresh summer vegetables from the yard",
]


def _age_all(store, hours: float) -> None:
    """Backdate every memory so the enrichment age gate passes."""
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    store._conn.execute("UPDATE memories SET created_at = ?", (ts,))
    store._commit()


class TestGenerateAnticipatedQueries:
    def test_empty_llm_returns_empty(self):
        with patch("cairn.enrichment.llm_complete", return_value=""):
            assert generate_anticipated_queries(GARDEN_MEMORY) == []

    def test_parses_json_array(self):
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            assert generate_anticipated_queries(GARDEN_MEMORY) == GARDEN_QUERIES

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(GARDEN_QUERIES) + "\n```"
        with patch("cairn.enrichment.llm_complete", return_value=raw):
            assert generate_anticipated_queries(GARDEN_MEMORY) == GARDEN_QUERIES

    def test_garbage_returns_empty(self):
        with patch("cairn.enrichment.llm_complete", return_value="Sure! Here are some queries:"):
            assert generate_anticipated_queries(GARDEN_MEMORY) == []

    def test_caps_at_five_and_drops_junk(self):
        raw = json.dumps(["q1 valid query", "", "x" * 300, "q2", "q3", "q4", "q5", "q6"])
        with patch("cairn.enrichment.llm_complete", return_value=raw):
            result = generate_anticipated_queries(GARDEN_MEMORY)
            assert len(result) == 5
            assert "" not in result
            assert all(len(q) <= 200 for q in result)


class TestEnrichPending:
    def test_no_llm_bails_early_and_writes_nothing(self, store):
        for i in range(5):
            store.store(
                content=f"Preference memory number {i}: " + GARDEN_MEMORY,
                metadata={"event_type": "user_preference"},
            )
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value="") as mock_llm:
            result = enrich_pending(store, limit=10)
        assert result["status"] == "no_llm"
        assert result["updated"] == 0
        assert mock_llm.call_count == 3  # early bail, not the whole batch
        row = store._conn.execute(
            "SELECT metadata FROM memories LIMIT 1"
        ).fetchone()
        assert "doc2query" not in (row[0] or "")

    def test_enriches_eligible_memory(self, store):
        nid = store.store(content=GARDEN_MEMORY, metadata={"event_type": "user_preference"})
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            result = enrich_pending(store, limit=10)
        assert result["status"] == "ok"
        assert result["updated"] == 1

        meta_raw, kw = store._conn.execute(
            "SELECT metadata, extracted_keywords FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        meta = json.loads(meta_raw)
        assert meta["doc2query"]["queries"] == GARDEN_QUERIES
        assert "homegrown" in kw  # anticipated-query terms landed in keywords

    def test_idempotent(self, store):
        store.store(content=GARDEN_MEMORY, metadata={"event_type": "user_preference"})
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            first = enrich_pending(store, limit=10)
            second = enrich_pending(store, limit=10)
        assert first["updated"] == 1
        assert second["updated"] == 0
        assert second["candidates"] == 0  # filtered out in SQL, no LLM spend

    def test_ineligible_types_skipped(self, store):
        store.store(
            content="Traceback (most recent call last): " + GARDEN_MEMORY,
            metadata={"event_type": "error_pattern"},
        )
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            result = enrich_pending(store, limit=10)
        assert result["updated"] == 0
        assert result["candidates"] == 0

    def test_short_content_skipped(self, store):
        store.store(
            content="Prefers ruff over flake8"[: MIN_CONTENT_CHARS - 1],
            metadata={"event_type": "user_preference"},
        )
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            result = enrich_pending(store, limit=10)
        assert result["candidates"] == 0

    def test_young_rows_deferred(self, store):
        store.store(content=GARDEN_MEMORY, metadata={"event_type": "user_preference"})
        # No backdating — the row is seconds old
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            result = enrich_pending(store, limit=10)
        assert result["updated"] == 0
        assert result["skipped_young"] == 1

    def test_fts_matches_anticipated_vocabulary(self, store):
        """The point of the feature: BM25 must match the enriched terms."""
        if not getattr(store, "_fts_available", False):
            pytest.skip("FTS5 unavailable")
        nid = store.store(content=GARDEN_MEMORY, metadata={"event_type": "user_preference"})
        _age_all(store, 2)

        def fts_hits(term):
            return {
                r[0] for r in store._conn.execute(
                    "SELECT m.node_id FROM memories_fts f JOIN memories m ON f.rowid = m.id "
                    "WHERE memories_fts MATCH ?", (term,)
                ).fetchall()
            }

        assert nid not in fts_hits('"homegrown"')  # not in the original wording
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            enrich_pending(store, limit=10)
        assert nid in fts_hits('"homegrown"')      # bridged after enrichment
        assert nid in fts_hits('"tomatoes"')       # original content still indexed

    def test_end_to_end_retrieval_bridges_vocabulary(self, store):
        """store.query() with indirect phrasing finds the enriched memory."""
        nid = store.store(content=GARDEN_MEMORY, metadata={"event_type": "user_preference"})
        # Distractors so the win isn't "only row in the store"
        for i in range(8):
            store.store(
                content=f"Deployment pipeline note {i}: CI stage ordering, artifact "
                f"caching and rollback procedure for service {i}.",
                metadata={"event_type": "memory"},
            )
        _age_all(store, 2)
        with patch("cairn.enrichment.llm_complete", return_value=json.dumps(GARDEN_QUERIES)):
            enrich_pending(store, limit=20)

        results = store.query(
            "what should I serve for dinner with my homegrown ingredients?",
            limit=5, use_cache=False,
        )
        from cairn.embedding import is_embedding_degraded

        assert not is_embedding_degraded(), (
            "embedding backend degraded to hash fallback — a runner memory or "
            "environment problem, not a retrieval regression"
        )
        assert any(r.id == nid for r in results), (
            "enriched memory not retrieved for indirect preference query"
        )

    def test_eligible_types_constant(self):
        """Episodic exhaust must never be eligible (that's rollup's job)."""
        assert "task_completion" not in ENRICH_ELIGIBLE_TYPES
        assert "session_summary" not in ENRICH_ELIGIBLE_TYPES
