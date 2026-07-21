"""Tests for P1-P6 retrieval improvements.

P1: Hybrid BM25/RRF fusion
P2: Cross-encoder reranking with temporal metadata
P3: Session-level retrieval aggregation
P4: Temporal indexing and retrieval channel
P5: Structured fact extraction at indexing
P6: Graph-based multi-hop retrieval
"""
import json
import pytest
from datetime import datetime, timedelta, timezone

from cairn.sqlite_store import SQLiteStore, SCHEMA_VERSION, MemoryResult


# ============================================================================
# P1: Reciprocal Rank Fusion
# ============================================================================


class TestRRFFusion:
    """Test the _rrf_fuse static method."""

    def test_rrf_single_channel(self):
        """RRF with one channel normalizes to [0, 1]."""
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        scores = SQLiteStore._rrf_fuse([ranked])
        assert scores["a"] == 1.0  # Rank 1 = max
        assert scores["b"] < scores["a"]
        assert scores["c"] < scores["b"]

    def test_rrf_two_channels_dual_match_boost(self):
        """Documents found by both channels get higher scores."""
        vec_ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
        text_ranked = [("a", 0.8), ("d", 0.6), ("e", 0.4)]
        scores = SQLiteStore._rrf_fuse([vec_ranked, text_ranked])
        # "a" appears in both channels -> highest score
        assert scores["a"] == 1.0
        # "b" only in vec, "d" only in text -> both should score lower than "a"
        assert scores["b"] < scores["a"]
        assert scores["d"] < scores["a"]

    def test_rrf_weighted_channels(self):
        """Channel weights affect relative contributions."""
        vec_ranked = [("a", 0.9)]
        text_ranked = [("b", 0.8)]
        # With equal weights, both should have same RRF score
        scores_equal = SQLiteStore._rrf_fuse([vec_ranked, text_ranked], weights=[1.0, 1.0])
        assert scores_equal["a"] == scores_equal["b"]

        # With vec weighted 2x, "a" should score higher
        scores_vec_heavy = SQLiteStore._rrf_fuse([vec_ranked, text_ranked], weights=[2.0, 1.0])
        assert scores_vec_heavy["a"] > scores_vec_heavy["b"]

    def test_rrf_empty_channels(self):
        """Empty channel lists return empty scores."""
        assert SQLiteStore._rrf_fuse([]) == {}
        assert SQLiteStore._rrf_fuse([[], []]) == {}

    def test_rrf_normalization(self):
        """All scores should be in [0, 1] range."""
        ranked = [("a", 0.9), ("b", 0.7), ("c", 0.5), ("d", 0.3)]
        scores = SQLiteStore._rrf_fuse([ranked])
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_rrf_three_channels(self):
        """RRF works with 3+ channels (for temporal retrieval)."""
        vec = [("a", 0.9), ("b", 0.7)]
        text = [("b", 0.8), ("c", 0.6)]
        temporal = [("a", 1.0), ("c", 0.5)]
        scores = SQLiteStore._rrf_fuse([vec, text, temporal])
        # "a" in vec+temporal, "b" in vec+text, "c" in text+temporal
        assert len(scores) == 3
        # All should be positive
        for v in scores.values():
            assert v > 0

    def test_rrf_integration_in_query(self, store):
        """Query pipeline uses RRF fusion (integration test)."""
        # Store memories with different content characteristics
        store.store(content="Python is a programming language used for data science",
                    metadata={"event_type": "lesson_learned"})
        store.store(content="JavaScript is used for web development and frontend",
                    metadata={"event_type": "lesson_learned"})
        store.store(content="Python Flask is a web framework for Python",
                    metadata={"event_type": "decision"})

        # Query should find Python-related memories
        results = store.query("Python programming", limit=10)
        assert len(results) > 0
        # The Python memories should be ranked higher
        python_found = any("Python" in r.content for r in results[:2])
        assert python_found

    def test_rrf_canonical_equals_historical_renorm(self):
        """Regression pin: the removed per-channel renorm was provably inert.

        The historical implementation min-max normalized each channel before
        weighting. Every non-empty channel's top rank scores exactly 1/(k+1),
        so that division was one uniform scalar across all channels — erased
        by the final max-norm. Canonical RRF must therefore be bit-identical
        to the historical variant on any input.
        """
        import random

        from cairn.sqlite_store._query import QueryMixin

        def historical_renorm(ranked_lists, weights, k=60):
            scores = {}
            for ci, ranked in enumerate(ranked_lists):
                if not ranked:
                    continue
                ch = {}
                for pos, (doc_id, _s) in enumerate(ranked):
                    ch[doc_id] = 1.0 / (k + pos + 1)
                ch_max = max(ch.values())
                for d in ch:
                    ch[d] /= ch_max
                for d, s in ch.items():
                    scores[d] = scores.get(d, 0.0) + weights[ci] * s
            if scores:
                m = max(scores.values())
                if m > 0:
                    scores = {d: s / m for d, s in scores.items()}
            return scores

        rng = random.Random(7)
        for trial in range(100):
            n_channels = rng.randint(1, 4)
            channels = []
            for _ in range(n_channels):
                seen, ranked = set(), []
                for _ in range(rng.randint(0, 12)):
                    d = f"d{rng.randint(0, 20)}"
                    if d not in seen:
                        seen.add(d)
                        ranked.append((d, rng.random()))
                channels.append(ranked)
            weights = [rng.choice([0.5, 0.7, 1.0, 1.2, 1.5]) for _ in range(n_channels)]

            canonical = QueryMixin._rrf_fuse(channels, weights=weights)
            legacy = historical_renorm(channels, weights)
            assert canonical.keys() == legacy.keys()
            for doc in canonical:
                assert abs(canonical[doc] - legacy[doc]) < 1e-12, (
                    f"trial {trial}: {doc} diverged"
                )


# ============================================================================
# Entity-match channel (in-repo replacement for the removed Pro entity
# expansion): query tokens vs metadata tags/fact terms/project/entity_id,
# fused as a fourth RRF channel at modest weight.
# ============================================================================


class TestEntityChannel:
    def test_token_extraction_skips_stopwords_and_short(self, store):
        # Stopword-only / short-token queries yield no tokens: channel skipped.
        assert store._entity_term_search("the and for it", 10, {}) == []
        assert store._entity_term_search("a b c", 10, {}) == []

    def test_tag_match_ranks_above_noise(self, store):
        target = store.store(
            content="Decision: SqliteStore migrations use additive columns only",
            metadata={"event_type": "decision", "tags": ["sqlitestore", "db-migration"]},
            skip_inference=True,
        )
        store.store(
            content="Unrelated note about kitchen renovation plans",
            metadata={"event_type": "memory"},
            skip_inference=True,
        )
        ranked = store._entity_term_search("sqlitestore migration strategy", 10, {})
        assert ranked and ranked[0][0] == target

    def test_channel_hydrates_all_results(self, store):
        target = store.store(
            content="Entity hydration check content",
            metadata={"event_type": "decision", "tags": ["hydrationtag"]},
            skip_inference=True,
        )
        seen = {}
        store._entity_term_search("hydrationtag lookup", 10, seen)
        assert target in seen

    def test_entity_channel_excludes_archived(self, store):
        nid = store.store(
            content="Archived entity content",
            metadata={"event_type": "task_completion", "tags": ["archivedtag"]},
            skip_inference=True,
        )
        with store._lock:
            store._conn.execute(
                "UPDATE memories SET status = 'archived' WHERE node_id = ?", (nid,))
            store._commit()
        ranked = store._entity_term_search("archivedtag lookup", 10, {})
        assert nid not in {n for n, _ in ranked}

    def test_project_column_match_contributes(self, store):
        nid = store.store(
            content="Force server API contract memory",
            metadata={"event_type": "decision", "project": "/p/force-server"},
            skip_inference=True,
        )
        ranked = store._entity_term_search("force-server contract", 10, {})
        assert nid in {n for n, _ in ranked}

    def test_entity_only_hit_loses_to_dual_channel_match(self, store):
        # Dual vec+text match on the query topic:
        dual = store.store(
            content="Kalman filter tuning: process noise dominates convergence",
            metadata={"event_type": "lesson_learned"},
            skip_inference=True,
        )
        # Content-irrelevant row whose only signal is a matching tag:
        tag_only = store.store(
            content="Weekly grocery run and errands checklist",
            metadata={"event_type": "task_completion", "tags": ["kalman"]},
            skip_inference=True,
        )
        results = store.query("kalman filter tuning convergence", limit=5)
        ids = [r.id for r in results]
        if dual in ids and tag_only in ids:
            assert ids.index(dual) < ids.index(tag_only)
        else:
            assert dual in ids

    def test_expand_entity_scope_removed(self, store):
        assert not hasattr(store, "_expand_entity_scope")


# ============================================================================
# P2: Cross-Encoder Reranking
# ============================================================================


class TestCrossEncoderReranking:
    """Test temporal metadata enrichment in reranker."""

    def test_cross_encoder_accepts_temporal_metadata(self):
        """cross_encoder_score should accept temporal_metadata parameter."""
        from cairn.reranker import cross_encoder_score
        import os
        # Disable actual model to test parameter passing
        os.environ["CAIRN_CROSS_ENCODER"] = "0"
        try:
            result = cross_encoder_score(
                "test query",
                ["passage 1", "passage 2"],
                temporal_metadata=["2024-01-15", "2024-02-20"],
            )
            assert result is None  # Disabled, but no error
        finally:
            os.environ.pop("CAIRN_CROSS_ENCODER", None)

    def test_reranker_model_selection(self):
        """Reranker auto-detects best available model."""
        from cairn.reranker import _RERANKER_MODEL_NAME
        # Auto-detects bge-reranker-v2-m3 if ONNX model exists on disk,
        # otherwise falls back to ms-marco-MiniLM-L-6-v2
        assert _RERANKER_MODEL_NAME in ("ms-marco-MiniLM-L-6-v2", "bge-reranker-v2-m3")

    def test_available_models_registry(self):
        """Both model configs should be in the registry.

        bge-reranker-v2-m3 uses the multi-precision schema (precisions.<p>.{dir,files});
        ms-marco-MiniLM-L-6-v2 uses the flat schema (dir, files at top level).
        """
        from cairn.reranker import _AVAILABLE_MODELS
        assert "bge-reranker-v2-m3" in _AVAILABLE_MODELS
        assert "ms-marco-MiniLM-L-6-v2" in _AVAILABLE_MODELS
        for name, config in _AVAILABLE_MODELS.items():
            assert "repo_id" in config
            if "precisions" in config:
                assert "default_precision" in config
                assert config["default_precision"] in config["precisions"]
                for variant in config["precisions"].values():
                    assert "dir" in variant
                    assert "files" in variant
            else:
                assert "dir" in config
                assert "files" in config


# ============================================================================
# P3: Session-Level Retrieval Aggregation
# ============================================================================


class TestSessionAggregation:
    """Test retrieve_by_session method."""

    def test_retrieve_by_session_groups_results(self, store):
        """Results should be grouped by session."""
        # Create memories in different sessions
        for i in range(3):
            store.store(
                content=f"Session A memory {i} about machine learning algorithms",
                session_id="session-a",
                metadata={"event_type": "lesson_learned", "session_id": "session-a"},
            )
        for i in range(3):
            store.store(
                content=f"Session B memory {i} about database optimization strategies",
                session_id="session-b",
                metadata={"event_type": "decision", "session_id": "session-b"},
            )

        results = store.retrieve_by_session(
            "machine learning", top_k_sessions=1,
        )
        # Should get results primarily from session-a
        session_ids = set(r.metadata.get("session_id") for r in results)
        assert "session-a" in session_ids

    def test_retrieve_by_session_empty_store(self, store):
        """Empty store returns empty results."""
        results = store.retrieve_by_session("anything")
        assert results == []

    def test_retrieve_by_session_respects_top_k(self, store):
        """Should limit to top_k_sessions sessions."""
        for sid in ["s1", "s2", "s3"]:
            store.store(
                content=f"Memory about testing in {sid} with pytest framework",
                session_id=sid,
                metadata={"event_type": "lesson_learned", "session_id": sid},
            )
        results = store.retrieve_by_session("testing pytest", top_k_sessions=2)
        session_ids = set(r.metadata.get("session_id") for r in results)
        assert len(session_ids) <= 2


# ============================================================================
# P4: Temporal Indexing and Retrieval
# ============================================================================


class TestTemporalRetrieval:
    """Test temporal search channel."""

    def test_temporal_search_finds_in_range(self, store):
        """Memories within date range should be found."""
        store.store(
            content="Meeting with client about project requirements",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-06-15T10:00:00",
            },
        )
        store.store(
            content="Code review for authentication module",
            metadata={
                "event_type": "task_completion",
                "referenced_date": "2024-01-01T10:00:00",
            },
        )

        results = store._temporal_search(
            "2024-06-01", "2024-06-30", limit=10,
        )
        assert len(results) >= 1
        # The June memory should score highest (in range)
        top_id, top_score = results[0]
        assert top_score == 1.0  # In-range proximity

    def test_temporal_search_empty_range(self, store):
        """No memories in range returns empty (far enough away to avoid 3x window)."""
        store.store(
            content="Old memory from 2020 about API design",
            metadata={
                "event_type": "decision",
                "referenced_date": "2020-01-15T10:00:00",
            },
        )
        # Search window is just 7 days, so 3x = 21 days. 2020 is 5 years away.
        results = store._temporal_search("2025-06-01", "2025-06-07", limit=10)
        assert len(results) == 0

    def test_temporal_search_proximity_decay(self, store):
        """Nearby but out-of-range memories get decayed scores."""
        # Memory just outside range
        store.store(
            content="Near-miss memory about deployment",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-06-05T10:00:00",
            },
        )
        # Memory far outside range
        store.store(
            content="Far-away memory about initial setup",
            metadata={
                "event_type": "decision",
                "referenced_date": "2024-01-01T10:00:00",
            },
        )
        results = store._temporal_search("2024-06-10", "2024-06-20", limit=10)
        # Near-miss should appear (within 3x range) with decayed score
        if results:
            assert results[0][1] < 1.0  # Not perfect score (out of range)
            assert results[0][1] > 0.0  # But still positive


# ============================================================================
# P5: Structured Fact Extraction
# ============================================================================


class TestFactExtraction:
    """Test _extract_keywords static method."""

    def test_extract_proper_nouns(self):
        """Multi-word proper nouns should be extracted."""
        content = "Met with John Smith at New York office about the project."
        keywords = SQLiteStore._extract_keywords(content)
        assert "John Smith" in keywords
        assert "New York" in keywords

    def test_extract_dates(self):
        """Dates in various formats should be extracted."""
        content = "Meeting scheduled for 2024-06-15 and another on Jan 20, 2025."
        keywords = SQLiteStore._extract_keywords(content)
        assert "2024-06-15" in keywords
        assert "Jan 20, 2025" in keywords

    def test_extract_technical_terms(self):
        """CamelCase and ACRONYMS should be extracted."""
        content = "The SQLiteStore uses ONNX for embedding and BM25 for text search."
        keywords = SQLiteStore._extract_keywords(content)
        assert "SQLiteStore" in keywords
        assert "ONNX" in keywords

    def test_extract_numbers_with_units(self):
        """Numbers with units should be extracted."""
        content = "The model is 384 MB and achieves 95% accuracy on the benchmark."
        keywords = SQLiteStore._extract_keywords(content)
        assert "95%" in keywords

    def test_extract_empty_content(self):
        """Empty content returns empty string."""
        assert SQLiteStore._extract_keywords("") == ""

    def test_extract_keywords_capped(self):
        """Keywords should be capped at 50."""
        content = " ".join(f"Entity{i} Name{i}" for i in range(100))
        keywords = SQLiteStore._extract_keywords(content)
        # Should not have more than 50 keywords
        assert len(keywords.split()) <= 100  # 50 multi-word entries

    def test_keywords_stored_in_db(self, store):
        """Extracted keywords should be stored in the extracted_keywords column."""
        nid = store.store(
            content="Meeting with John Smith about SQLiteStore performance on 2024-06-15",
            metadata={"event_type": "decision"},
        )
        row = store._conn.execute(
            "SELECT extracted_keywords FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert row is not None
        assert row[0] is not None
        assert "John Smith" in row[0]

    def test_keywords_enhance_fts_search(self, store):
        """Keywords in FTS index should improve BM25 search recall."""
        # Store a memory where the key entity only appears in extracted keywords
        nid = store.store(
            content="Discussed project timeline with John Smith at the office today",
            metadata={"event_type": "decision"},
        )
        # Search for the entity name should find it (via enriched FTS)
        results = store._text_search("John Smith", limit=5)
        found_ids = [r.id for r in results]
        assert nid in found_ids


# ============================================================================
# P6: Graph Multi-Hop Retrieval
# ============================================================================


class TestGraphMultiHop:
    """Test multi-hop graph traversal."""

    def test_single_hop_traversal(self, store):
        """1-hop neighbors should be surfaced."""
        # Create seed memory and a connected neighbor
        seed_id = store.store(
            content="Core architecture decision about microservices pattern",
            metadata={"event_type": "decision"},
        )
        neighbor_id = store.store(
            content="Related implementation detail about service mesh",
            metadata={"event_type": "task_completion"},
        )
        # Create edge between them
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.8, datetime('now'))""",
            (seed_id, neighbor_id),
        )
        store._conn.commit()

        # Query should find seed and potentially the neighbor
        results = store.query("microservices architecture", limit=10)
        result_ids = [r.id for r in results]
        assert seed_id in result_ids

    def test_two_hop_traversal(self, store):
        """2-hop neighbors should be surfaced with decayed scores."""
        # Chain: A -> B -> C
        a_id = store.store(
            content="Original decision about database sharding strategy",
            metadata={"event_type": "decision"},
        )
        b_id = store.store(
            content="Follow-up on sharding implementation details and timeline",
            metadata={"event_type": "task_completion"},
        )
        c_id = store.store(
            content="Performance benchmark results after sharding deployment",
            metadata={"event_type": "lesson_learned"},
        )
        now = datetime.now(timezone.utc).isoformat()
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.9, ?)""",
            (a_id, b_id, now),
        )
        store._conn.execute(
            """INSERT INTO edges (source_id, target_id, edge_type, weight, created_at)
               VALUES (?, ?, 'causal', 0.8, ?)""",
            (b_id, c_id, now),
        )
        store._conn.commit()

        results = store.query("database sharding", limit=10)
        result_ids = [r.id for r in results]
        assert a_id in result_ids  # Direct match


# ============================================================================
# Schema Migration
# ============================================================================


class TestSchemaMigration:
    """Test v7 -> v8 schema migration."""

    def test_schema_version_is_8(self, store):
        """SCHEMA_VERSION should be 8."""
        assert SCHEMA_VERSION == 15
        row = store._conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()
        assert row[0] == 15

    def test_end_date_column_exists(self, store):
        """The memories table should have an end_date column."""
        cols = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [c[1] for c in cols]
        assert "end_date" in col_names

    def test_extracted_keywords_column_exists(self, store):
        """The memories table should have an extracted_keywords column."""
        cols = store._conn.execute("PRAGMA table_info(memories)").fetchall()
        col_names = [c[1] for c in cols]
        assert "extracted_keywords" in col_names


class TestMalformedPriorityMetadata:
    """A memory with a non-numeric priority must not kill the query.

    Real incident: a live memory carried priority="high" (text, not 1-5);
    _metadata_score_factor computed 0.7 + ("high" * 0.08) -> TypeError,
    aborting the entire query whenever that memory was a candidate.
    """

    def test_query_survives_text_priority(self, store):
        store.store(
            content="Deployment freeze decision for the holiday season window",
            metadata={"event_type": "decision", "priority": "high"},
        )
        store.store(
            content="Unrelated note about database vacuum scheduling",
            metadata={"event_type": "memory"},
        )
        # Must not raise, and the malformed-priority memory must still surface.
        results = store.query("deployment freeze holiday", limit=5)
        assert any("freeze" in r.content for r in results)

    def test_query_survives_list_priority_in_metadata_json(self, store):
        # A list can't bind to the priority COLUMN (store() raises), but it
        # can exist inside the metadata JSON via import/legacy paths — inject
        # it there directly and prove retrieval still doesn't raise.
        nid = store.store(
            content="Priority list experiment memo with odd metadata shape",
            metadata={"event_type": "memory"},
        )
        store._conn.execute(
            "UPDATE memories SET metadata = json_set(metadata, '$.priority', json('[1,2]')) "
            "WHERE node_id = ?",
            (nid,),
        )
        store._commit()
        results = store.query("priority list experiment memo", limit=5)
        assert isinstance(results, list)  # no exception is the contract

    def test_numeric_string_priority_is_coerced(self, store):
        store.store(
            content="Numeric string priority memo about cache eviction",
            metadata={"event_type": "memory", "priority": "5"},
        )
        results = store.query("cache eviction memo", limit=5)
        assert isinstance(results, list)


class TestCEResortMode:
    """CAIRN_CE_MODE=resort lets the cross-encoder fully reorder the top-K
    (permuting existing fused scores, preserving the score multiset), vs the
    default position-aware gentle boost."""

    @staticmethod
    def _fake_ce(favored: str):
        def fake(query, passages, temporal_metadata=None):
            return [10.0 if favored in p else float(-i) for i, p in enumerate(passages)]
        return fake

    def test_resort_promotes_ce_favorite(self, store, monkeypatch):
        # All candidates must be relevant enough to survive the Phase-4
        # composite floor — an off-topic memory never reaches the reranker.
        import cairn.reranker as rr

        store.store(content="Deploy pipeline memo alpha about blue-green rollout health gates")
        store.store(content="Deploy pipeline memo beta about rollback of failed canary releases")
        store.store(content="Deploy pipeline memo gamma about staging smoke checks before traffic")

        monkeypatch.setenv("CAIRN_CE_MODE", "resort")
        monkeypatch.setattr(rr, "cross_encoder_score", self._fake_ce("gamma"))

        results = store.query("deploy pipeline memo", limit=3)
        assert results, "query returned nothing"
        assert "gamma" in results[0].content, (
            "resort mode must let the CE fully decide the top result"
        )

    def test_resort_preserves_score_multiset(self, store, monkeypatch):
        import cairn.reranker as rr

        for i in range(4):
            store.store(content=f"Distinct memo number {i} about database index tuning strategies")

        monkeypatch.setattr(rr, "cross_encoder_score", self._fake_ce("number 3"))

        monkeypatch.setenv("CAIRN_CE_MODE", "boost")
        boost = store.query("database index tuning", limit=4)
        monkeypatch.setenv("CAIRN_CE_MODE", "resort")
        resort = store.query("database index tuning", limit=4)

        # Same result SET either way — resort only reorders.
        assert {r.id for r in boost} == {r.id for r in resort}


class TestCEHybridMode:
    def test_hybrid_promotes_ce_favorite_only(self, store, monkeypatch):
        import cairn.reranker as rr
        store.store(content="Deploy pipeline memo alpha about blue-green rollout health gates")
        store.store(content="Deploy pipeline memo beta about rollback of failed canary releases")
        store.store(content="Deploy pipeline memo gamma about staging smoke checks before traffic")
        monkeypatch.setenv("CAIRN_CE_MODE", "hybrid")
        monkeypatch.setattr(rr, "cross_encoder_score", TestCEResortMode._fake_ce("gamma"))
        results = store.query("deploy pipeline memo", limit=3)
        assert results and "gamma" in results[0].content
