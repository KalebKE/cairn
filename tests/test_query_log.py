"""Durable query_log: real-query corpus for retrieval eval (replay probes).

Every SQLiteStore.query() appends (ts, query_text, surfacing_context,
session_id, result_count, top_score) to the query_log table, gated by the
call-time CAIRN_QUERY_LOG env read (default on, "0" disables). Maintenance
prunes entries older than 90 days.
"""
from datetime import datetime, timedelta, timezone

import pytest


def _log_rows(store):
    return store._conn.execute(
        "SELECT ts, query_text, session_id, result_count, top_score FROM query_log ORDER BY id"
    ).fetchall()


def test_schema_has_query_log_table(store):
    row = store._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'"
    ).fetchone()
    assert row is not None, "query_log table missing from fresh schema"
    version = store._conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert version >= 15


def test_query_inserts_log_row(store, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    store.store(content="The deploy pipeline uses blue-green rollouts on fly.io")
    store.query("deploy pipeline rollout")

    rows = _log_rows(store)
    assert len(rows) == 1
    ts, qtext, session_id, result_count, top_score = rows[0]
    assert qtext == "deploy pipeline rollout"
    assert result_count >= 0
    # ts parses as ISO UTC
    datetime.fromisoformat(ts)


def test_zero_result_queries_are_logged(store, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    store.query("xyzzy quantum ferret nonesuch")
    rows = _log_rows(store)
    assert len(rows) == 1
    assert rows[0][3] == 0  # result_count


def test_env_kill_switch(store, monkeypatch):
    monkeypatch.setenv("CAIRN_QUERY_LOG", "0")
    store.store(content="Kill switch memory about caching layers")
    store.query("caching layers")
    assert _log_rows(store) == []


def test_long_query_truncated_to_500(store, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    long_q = "database " * 100  # 900 chars
    store.query(long_q)
    rows = _log_rows(store)
    assert len(rows) == 1
    assert len(rows[0][1]) <= 500


def test_session_id_recorded(store, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    store.query("anything at all", session_id="sess-ql-1")
    rows = _log_rows(store)
    assert rows[0][2] == "sess-ql-1"


def test_prune_query_log(store, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
    fresh_ts = datetime.now(timezone.utc).isoformat()
    store._conn.execute(
        "INSERT INTO query_log (ts, query_text, result_count, top_score) VALUES (?, 'old', 0, 0)",
        (old_ts,),
    )
    store._conn.execute(
        "INSERT INTO query_log (ts, query_text, result_count, top_score) VALUES (?, 'fresh', 0, 0)",
        (fresh_ts,),
    )
    store._commit()

    removed = store.prune_query_log(max_age_days=90)
    assert removed == 1
    remaining = [r[1] for r in _log_rows(store)]
    assert remaining == ["fresh"]
