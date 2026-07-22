"""Knowledge-first retention regression tests (fork Phase 1).

The store's value is durable knowledge (decisions, lessons, plans, docs).
These tests pin the retention semantics that protect it:

1. consolidate() must never age-prune decisions/lessons/insights (upstream
   Phase 0 deleted zero-access decisions with priority<5 at 14d — and the
   DEFAULT decision priority is 4).
2. Embedding dedup must keep the NEWER memory for knowledge types (upstream
   kept the older and silently discarded the refinement).
3. mark_superseded must write a traversable `supersedes` edge (upstream kept
   lineage only in a scalar metadata field).
4. Hash-fallback embeddings must be tagged for recovery at store time and be
   re-embeddable once the real model is back (the silent-degradation
   incident: model missing for months, hash vectors stored permanently).
5. Stray status values must normalize to the enum.
"""
import json

import pytest
from _vec import requires_vec


def _backdate(store, node_id: str, days: int) -> None:
    store._conn.execute(
        "UPDATE memories SET created_at = datetime('now', ?) WHERE node_id = ?",
        (f"-{days} days", node_id),
    )
    store._conn.commit()


def _row(store, node_id: str):
    return store._conn.execute(
        "SELECT node_id, event_type, status, access_count, metadata "
        "FROM memories WHERE node_id = ?",
        (node_id,),
    ).fetchone()


class TestConsolidatePreservesKnowledge:
    def test_zero_access_default_priority_decision_survives(self, store):
        nid = store.store(
            content="Decision: route all TracQi weather lookups through the Force-Server "
                    "/weather endpoint instead of per-client API keys",
            metadata={"event_type": "decision"},  # default priority (4)
        )
        _backdate(store, nid, 30)
        store.consolidate(prune_days=14)
        assert _row(store, nid) is not None, "zero-access decision must never be age-pruned"

    def test_zero_access_low_priority_decision_survives(self, store):
        nid = store.store(
            content="Decision: prefer Flyway repeatable migrations for reference data loads",
            metadata={"event_type": "decision", "priority": 2},
        )
        _backdate(store, nid, 60)
        store.consolidate(prune_days=14)
        assert _row(store, nid) is not None, "even low-priority decisions are knowledge — keep"

    def test_zero_access_lesson_survives(self, store):
        nid = store.store(
            content="Lesson: SwiftData migrations silently drop rows when a unique "
                    "constraint is added to a populated store — always stage-copy first",
            metadata={"event_type": "lesson_learned"},
        )
        _backdate(store, nid, 45)
        store.consolidate(prune_days=14)
        assert _row(store, nid) is not None, "zero-access lessons must never be age-pruned"

    def test_zero_access_advisor_insight_survives(self, store):
        nid = store.store(
            content="Insight: the OBD2 ingestion pipeline and billFold importer share the "
                    "same batching flaw — both need idempotent upserts",
            metadata={"event_type": "advisor_insight"},
        )
        _backdate(store, nid, 45)
        store.consolidate(prune_days=14)
        assert _row(store, nid) is not None

    def test_zero_access_episodic_is_still_pruned(self, store):
        """Guard against over-protecting: episodic exhaust must still age out."""
        nid = store.store(
            content="Completed: bumped Gradle wrapper and re-ran CI for obd2-android",
            metadata={"event_type": "task_completion"},
        )
        _backdate(store, nid, 45)
        store.consolidate(prune_days=14)
        assert _row(store, nid) is None, "zero-access task_completion should be pruned"


class TestKnowledgeDedupKeepsNewer:
    def _force_dedup_hit(self, store, old_node_id, monkeypatch):
        """Make the embedding-dedup precheck report a >=0.88 match on old_node_id."""
        old_rowid = store._conn.execute(
            "SELECT id FROM memories WHERE node_id = ?", (old_node_id,)
        ).fetchone()[0]
        monkeypatch.setattr(
            store, "_vec_query", lambda emb, limit=1, **kw: [(old_rowid, 0.05)]
        )

    @requires_vec

    def test_decision_dedup_supersedes_old_and_keeps_new(self, store, monkeypatch):
        old = store.store(
            content="Decision v1: CoDriver enrichment cards use the A3 prompt variant",
            metadata={"event_type": "decision"},
        )
        self._force_dedup_hit(store, old, monkeypatch)
        new = store.store(
            content="Decision v2: CoDriver enrichment cards use the A5 explainer prompt "
                    "variant with forced-inline citations",
            metadata={"event_type": "decision"},
        )
        assert new != old, "knowledge dedup must create the new memory, not swallow it"
        old_row = _row(store, old)
        assert old_row[2] == "superseded", "older duplicate decision must be superseded"
        meta = json.loads(old_row[4])
        assert meta.get("superseded_by") == new
        edge = store._conn.execute(
            "SELECT 1 FROM edges WHERE source_id=? AND target_id=? AND edge_type='supersedes'",
            (new, old),
        ).fetchone()
        assert edge, "supersession must be graph-traversable (supersedes edge new->old)"

    @requires_vec

    def test_episodic_dedup_still_keeps_old(self, store, monkeypatch):
        old = store.store(
            content="Completed: nightly Force-Server backup verified restore path",
            metadata={"event_type": "task_completion"},
        )
        self._force_dedup_hit(store, old, monkeypatch)
        new = store.store(
            content="Completed again: nightly Force-Server backup verified restore path ok",
            metadata={"event_type": "task_completion"},
        )
        assert new == old, "episodic dedup keeps the existing memory (unchanged behavior)"


class TestSupersedesEdge:
    def test_mark_superseded_writes_edge(self, store):
        v1 = store.store(
            content="Plan v1: ship tracqi-web dashboards behind a beta flag in July",
            metadata={"event_type": "decision"},
        )
        v2 = store.store(
            content="Plan v2: ship tracqi-web dashboards GA in August with usage caps",
            metadata={"event_type": "decision"},
        )
        assert store.mark_superseded(v1, superseded_by=v2)
        edge = store._conn.execute(
            "SELECT weight FROM edges WHERE source_id=? AND target_id=? AND edge_type='supersedes'",
            (v2, v1),
        ).fetchone()
        assert edge, "mark_superseded must write a supersedes edge (new->old)"


class TestHashEmbeddingRecovery:
    def _degrade(self, monkeypatch):
        import cairn.embedding as E
        monkeypatch.setattr(E, "generate_embedding", lambda text, dimension=384: [0.1] * 384)
        monkeypatch.setattr(E, "is_embedding_degraded", lambda: True)
        monkeypatch.setattr(E, "get_active_backend", lambda: None)

    def test_store_tags_hash_embeddings(self, store, monkeypatch):
        """When the model never loaded (backend None), hash vectors are accepted
        for continuity — but MUST be tagged so a recovery job can find them."""
        self._degrade(monkeypatch)
        nid = store.store(
            content="Business plan note: TracQi subscription tiers land in Q4",
            metadata={"event_type": "decision"},
        )
        meta = json.loads(_row(store, nid)[4])
        assert meta.get("_embedding_backend") == "hash", (
            "hash-fallback embeddings must be tagged for later re-embedding"
        )

    def test_reembed_hash_tainted_recovers_and_clears_tag(self, store, monkeypatch):
        self._degrade(monkeypatch)
        nid = store.store(
            content="Technical plan: move MQTT sync cursors to server-issued tokens",
            metadata={"event_type": "decision"},
        )
        # model comes back healthy
        monkeypatch.undo()
        result = store.reembed_hash_tainted(batch_size=50)
        assert result["reembedded"] >= 1
        meta = json.loads(_row(store, nid)[4])
        assert meta.get("_embedding_backend") != "hash", "tag must clear after re-embed"

    def test_reembed_skips_while_still_degraded(self, store, monkeypatch):
        self._degrade(monkeypatch)
        store.store(
            content="History: April outage traced to WAL checkpoint starvation",
            metadata={"event_type": "lesson_learned"},
        )
        result = store.reembed_hash_tainted(batch_size=50)
        assert result["reembedded"] == 0
        assert result.get("skipped_degraded"), "must refuse to re-embed with hash vectors"


class TestStrengthDecayProtectsKnowledge:
    def test_old_zero_access_lesson_not_strength_decayed(self, store):
        nid = store.store(
            content="Lesson: Hetzner snapshot restores need the volume detached first",
            metadata={"event_type": "lesson_learned"},
        )
        store._conn.execute(
            "UPDATE memories SET created_at = datetime('now', '-400 days'), access_count = 0 "
            "WHERE node_id = ?", (nid,),
        )
        store._conn.commit()
        store.apply_strength_decay(min_age_days=90)
        row = store._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert not json.loads(row[0]).get("superseded"), (
            "durable knowledge must not be soft-deleted by strength decay"
        )


class TestStatusNormalization:
    @pytest.mark.parametrize("stray", ["complete", "completed", "verified", "partial_complete"])
    def test_store_normalizes_stray_status(self, store, stray):
        nid = store.store(
            content=f"Documented deploy runbook step with status {stray}",
            metadata={"event_type": "memory"},
            status=stray,
        )
        assert _row(store, nid)[2] == "active", "stray status values must normalize to enum"

    def test_valid_statuses_pass_through(self, store):
        nid = store.store(
            content="Speculative: kalman-filter drift model might explain OBD2 jitter",
            metadata={"event_type": "memory"},
            status="speculative",
        )
        assert _row(store, nid)[2] == "speculative"
