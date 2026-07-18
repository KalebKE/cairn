"""Tests for the LLM-assisted episodic rollup (fork Phase 4).

Rollup is the knowledge-first replacement for silently TTL-expiring episodic
exhaust: aging task_completions/session_summaries are synthesized (real LLM
abstraction, not sentence-picking) into one durable `project_history` memory
per project-month, linked via derived_from edges; the raw rows then expire on
a short grace TTL.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest

from omega import rollup as R


def _seed(store, project="/p/force-server", days_ago=40, n_tasks=4, n_summaries=2):
    """Seed old episodic rows for one project/calendar window."""
    created = (datetime.now(timezone.utc) - timedelta(days=days_ago))
    window = created.strftime("%Y-%m")
    ids = []
    for i in range(n_tasks):
        nid = store.store(
            content=f"Completed: enrichment pipeline step {i} shipped and verified",
            metadata={"event_type": "task_completion", "project": project},
            skip_inference=True,  # deterministic seeding: no embedding dedup
        )
        ids.append(nid)
    for i in range(n_summaries):
        nid = store.store(
            content=f"Session summary {i}: worked on catalog enrichment recovery",
            metadata={"event_type": "session_summary", "project": project},
            skip_inference=True,
        )
        ids.append(nid)
    ph = ",".join("?" * len(ids))
    store._conn.execute(
        f"UPDATE memories SET created_at = ? WHERE node_id IN ({ph})",
        [created.isoformat(), *ids],
    )
    store._conn.commit()
    return window, ids


@pytest.fixture
def fake_llm(monkeypatch):
    calls = {"n": 0, "prompts": []}

    def fake(prompt, system, **kw):
        calls["n"] += 1
        calls["prompts"].append(prompt)
        return ("Force-Server June: catalog enrichment pipeline overhauled and "
                "recovered; sync-cursor race fixed; verified against dev.")

    monkeypatch.setattr(R, "llm_complete", fake)
    return calls


class TestRollup:
    def test_rollup_creates_project_history_and_expires_raw(self, store, fake_llm):
        window, ids = _seed(store)
        result = R.rollup_pending(store, min_age_days=30)
        assert result["windows_rolled"] == 1
        assert fake_llm["n"] == 1
        # all source contents made it into the synthesis prompt
        assert "enrichment pipeline step 0" in fake_llm["prompts"][0]

        row = store._conn.execute(
            "SELECT node_id, content, ttl_seconds, metadata FROM memories "
            "WHERE event_type = 'project_history'"
        ).fetchone()
        assert row is not None, "one durable project_history memory must be created"
        assert "catalog enrichment" in row[1]
        assert row[2] is None, "project_history must be PERMANENT (no TTL)"
        meta = json.loads(row[3])
        assert meta.get("rollup_window") == window
        assert meta.get("source_count") == len(ids)

        # derived_from edges to every source row
        n_edges = store._conn.execute(
            "SELECT count(*) FROM edges WHERE source_id = ? AND edge_type = 'derived_from'",
            (row[0],),
        ).fetchone()[0]
        assert n_edges == len(ids)

        # raw rows: marked rolled_up + short grace TTL (expire within ~15 days)
        for nid in ids:
            r = store._conn.execute(
                "SELECT ttl_seconds, created_at, metadata FROM memories WHERE node_id = ?",
                (nid,),
            ).fetchone()
            assert json.loads(r[2]).get("rolled_up"), "source rows must be marked rolled_up"
            assert r[0] is not None
            expiry = store._parse_dt(r[1]) + timedelta(seconds=r[0])
            days_left = (expiry - datetime.now(timezone.utc)).total_seconds() / 86400
            assert 0 < days_left <= 15, f"grace TTL wrong: {days_left:.1f}d left"

    def test_rollup_is_idempotent(self, store, fake_llm):
        _seed(store)
        R.rollup_pending(store, min_age_days=30)
        second = R.rollup_pending(store, min_age_days=30)
        assert second["windows_rolled"] == 0, "already-rolled windows must not re-roll"
        n_hist = store._conn.execute(
            "SELECT count(*) FROM memories WHERE event_type = 'project_history'"
        ).fetchone()[0]
        assert n_hist == 1

    def test_recent_rows_not_rolled(self, store, fake_llm):
        _seed(store, days_ago=5)  # too recent
        result = R.rollup_pending(store, min_age_days=30)
        assert result["windows_rolled"] == 0
        assert fake_llm["n"] == 0

    def test_knowledge_types_never_rolled(self, store, fake_llm):
        nid = store.store(
            content="Decision: keep weather lookups server-side",
            metadata={"event_type": "decision", "project": "/p/x"},
        )
        store._conn.execute(
            "UPDATE memories SET created_at = datetime('now','-40 days') WHERE node_id = ?",
            (nid,),
        )
        store._conn.commit()
        R.rollup_pending(store, min_age_days=30)
        r = store._conn.execute(
            "SELECT ttl_seconds, metadata FROM memories WHERE node_id = ?", (nid,)
        ).fetchone()
        assert r[0] is None and not json.loads(r[1]).get("rolled_up"), (
            "decisions are never rollup sources"
        )

    def test_dry_run_writes_nothing(self, store, fake_llm):
        _seed(store)
        result = R.rollup_pending(store, min_age_days=30, dry_run=True)
        assert result["windows_rolled"] == 1  # would have rolled
        assert store._conn.execute(
            "SELECT count(*) FROM memories WHERE event_type='project_history'"
        ).fetchone()[0] == 0
        assert store._conn.execute(
            "SELECT count(*) FROM memories WHERE json_extract(metadata,'$.rolled_up') = 1"
        ).fetchone()[0] == 0

    def test_llm_failure_skips_window_untouched(self, store, monkeypatch):
        _seed(store)
        monkeypatch.setattr(R, "llm_complete", lambda *a, **k: "")
        result = R.rollup_pending(store, min_age_days=30)
        assert result["windows_rolled"] == 0
        assert result.get("skipped_llm") == 1
        assert store._conn.execute(
            "SELECT count(*) FROM memories WHERE json_extract(metadata,'$.rolled_up') = 1"
        ).fetchone()[0] == 0, "raw rows must be untouched when synthesis fails"

    def test_archive_raw_mode(self, store, fake_llm):
        """archive_raw=True keeps raw rows on disk (status=archived, no TTL)
        instead of grace-TTL deletion — MemPalace-style verbatim fidelity
        behind the synthesis, out of the retrieval path."""
        _, ids = _seed(store)
        R.rollup_pending(store, min_age_days=30, archive_raw=True)
        for nid in ids:
            r = store._conn.execute(
                "SELECT status, ttl_seconds, metadata FROM memories WHERE node_id = ?",
                (nid,),
            ).fetchone()
            assert r[0] == "archived"
            assert r[1] is None, "archived raws must not TTL-expire"
            meta = json.loads(r[2])
            assert meta.get("rolled_up") and meta.get("archived")

    def test_archived_rows_excluded_from_query(self, store, fake_llm):
        _, ids = _seed(store)
        R.rollup_pending(store, min_age_days=30, archive_raw=True)
        results = store.query("enrichment pipeline step shipped verified", limit=10)
        got = {r.id for r in results}
        assert not (got & set(ids)), "archived raw rows must never surface in retrieval"

    def test_min_rows_threshold(self, store, fake_llm):
        _seed(store, n_tasks=1, n_summaries=0)  # below default min of 3
        result = R.rollup_pending(store, min_age_days=30)
        assert result["windows_rolled"] == 0, "tiny windows are not worth an LLM call"
