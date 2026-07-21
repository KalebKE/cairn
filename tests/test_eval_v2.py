"""Eval v2: frozen non-self-referential probe sets, snapshot isolation,
history writer, and the paired sign test.

No LLM calls anywhere — probes are hand-built with explicit qrels.
"""
import json
from pathlib import Path

import pytest

from cairn.evaluation.probe_set import (
    PROBE_SET_VERSION,
    ProbeV2,
    lexical_overlap,
    load_probe_set,
)
from cairn.evaluation.retrieval_eval import (
    EvalReportV2,
    run_evaluation_v2,
    sign_test,
    snapshot_db,
    write_history_row,
)


def _freeze(path: Path, probes):
    from dataclasses import asdict
    payload = {
        "version": PROBE_SET_VERSION,
        "created_at": "2026-07-21T00:00:00+00:00",
        "seed": 1,
        "requested_size": len(probes),
        "probe_count": len(probes),
        "store_rows": 0,
        "skipped": [],
        "content_sha256": "deadbeef" * 8,
        "probes": [asdict(p) for p in probes],
    }
    path.write_text(json.dumps(payload))
    return str(path)


# --- probe set io -----------------------------------------------------------

def test_probe_set_round_trip(tmp_path):
    p = ProbeV2(query_text="how do we roll out deploys safely",
                method="llm-divergent",
                qrels={"mem-aaa": 3, "mem-bbb": 0},
                meta={"seed_memory_id": "mem-aaa"})
    path = _freeze(tmp_path / "probes.json", [p])
    probes, meta = load_probe_set(path)
    assert len(probes) == 1
    assert probes[0].qrels == {"mem-aaa": 3, "mem-bbb": 0}
    assert meta["version"] == PROBE_SET_VERSION


def test_probe_set_rejects_wrong_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 1, "probes": [{"query_text": "x", "qrels": {}}]}))
    with pytest.raises(ValueError, match="version"):
        load_probe_set(str(path))


def test_probe_set_rejects_empty(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"version": PROBE_SET_VERSION, "probes": []}))
    with pytest.raises(ValueError, match="no probes"):
        load_probe_set(str(path))


def test_lexical_overlap():
    content = "The deploy pipeline uses blue-green rollouts on flyio with health gates"
    assert lexical_overlap("deploy pipeline rollouts", content) > 0.6
    assert lexical_overlap("safely ship new versions without downtime", content) < 0.5


# --- sign test --------------------------------------------------------------

def test_sign_test_balanced_is_insignificant():
    pos, neg, p = sign_test([1.0, -1.0, 0.5, -0.5])
    assert (pos, neg) == (2, 2)
    assert p == 1.0


def test_sign_test_one_sided_wins():
    pos, neg, p = sign_test([0.1] * 10)
    assert (pos, neg) == (10, 0)
    assert p < 0.01


def test_sign_test_ignores_zero_deltas():
    pos, neg, p = sign_test([0.0, 0.0, 0.0])
    assert (pos, neg, p) == (0, 0, 1.0)


# --- runner against a real tiny store --------------------------------------

@pytest.fixture
def seeded_store(store):
    ids = {}
    ids["deploy"] = None
    r1 = store.store(content="Deploy pipeline decision: blue-green rollouts on fly.io with health gates before traffic shift")
    r2 = store.store(content="Lesson learned: never run database migrations during peak traffic hours")
    r3 = store.store(content="The kalman filter tuning uses process noise 0.01 for accelerometer fusion")
    return store, {"deploy": r1, "migrations": r2, "kalman": r3}


def test_run_evaluation_v2_end_to_end(seeded_store, tmp_path, monkeypatch):
    store, ids = seeded_store
    probe = ProbeV2(
        query_text="how should we ship new versions without downtime",
        method="llm-divergent",
        qrels={ids["deploy"]: 3, ids["migrations"]: 1, ids["kalman"]: 0},
        meta={},
    )
    path = _freeze(tmp_path / "probes.json", [probe, ProbeV2(
        query_text="sensor fusion noise configuration",
        method="llm-divergent",
        qrels={ids["kalman"]: 3, ids["deploy"]: 0},
        meta={},
    )])

    report = run_evaluation_v2(path, db_path=store.db_path, top_k=5)
    assert isinstance(report, EvalReportV2)
    assert report.probe_count == 2
    assert 0.0 <= report.mrr <= 1.0
    assert 0.0 <= report.ndcg_at_k <= 1.0
    assert len(report.per_probe) == 2
    # determinism controls recorded
    assert report.expansion_enabled is False


def test_runner_restores_env(seeded_store, tmp_path, monkeypatch):
    store, ids = seeded_store
    monkeypatch.setenv("CAIRN_QUERY_EXPANSION", "1")
    monkeypatch.setenv("CAIRN_CROSS_ENCODER", "1")
    path = _freeze(tmp_path / "p.json", [ProbeV2(
        query_text="anything", method="llm-divergent", qrels={ids["deploy"]: 2}, meta={})])
    run_evaluation_v2(path, db_path=store.db_path, top_k=3,
                      env={"CAIRN_CROSS_ENCODER": "0"}, variant="ce-off")
    import os
    assert os.environ["CAIRN_QUERY_EXPANSION"] == "1"
    assert os.environ["CAIRN_CROSS_ENCODER"] == "1"


# --- snapshot isolation -----------------------------------------------------

def test_snapshot_isolation(seeded_store, tmp_path):
    store, ids = seeded_store
    before = store._conn.execute(
        "SELECT node_id, access_count, COALESCE(retrieval_count,0) FROM memories ORDER BY node_id"
    ).fetchall()

    snap = snapshot_db(store.db_path, str(tmp_path))
    assert Path(snap).exists() and snap != store.db_path

    path = _freeze(tmp_path / "p.json", [ProbeV2(
        query_text="ship versions without downtime",
        method="llm-divergent", qrels={ids["deploy"]: 3}, meta={})])
    run_evaluation_v2(path, db_path=snap, top_k=5)

    after = store._conn.execute(
        "SELECT node_id, access_count, COALESCE(retrieval_count,0) FROM memories ORDER BY node_id"
    ).fetchall()
    assert before == after, "eval against snapshot mutated the source store"


# --- history writer ---------------------------------------------------------

def test_history_row_matches_doctor_parser(tmp_path):
    report = EvalReportV2(
        timestamp="2026-07-21T00:00:00+00:00",
        probe_set_path="p.json", probe_set_sha256="cafebabe1234",
        probe_count=5, top_k=5, variant="ce-on",
        expansion_enabled=False, cross_encoder_enabled=True,
        db_path="x.db", mrr=0.7321, precision_at_k=0.44, ndcg_at_k=0.61,
    )
    csv = tmp_path / "eval-history.csv"
    write_history_row(report, str(csv))
    write_history_row(report, str(csv))  # append path

    lines = csv.read_text().strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    # the doctor's trend parser: rows[1:], float(col[1]) == MRR
    for row in lines[1:]:
        cols = row.split(",")
        assert abs(float(cols[1]) - 0.7321) < 1e-9


# --- paired A/B determinism ------------------------------------------------

def test_compare_ab_identical_variants_zero_delta(seeded_store, tmp_path):
    """Same env on both arms => the harness must measure exactly zero delta.

    This is the determinism guarantee that makes any nonzero A/B delta
    attributable to the knob under test rather than harness noise.
    """
    from cairn.evaluation.retrieval_eval import compare_ab

    store, ids = seeded_store
    path = _freeze(tmp_path / "p.json", [
        ProbeV2(query_text="ship versions without downtime",
                method="llm-divergent", qrels={ids["deploy"]: 3}, meta={}),
        ProbeV2(query_text="sensor fusion noise configuration",
                method="llm-divergent", qrels={ids["kalman"]: 3}, meta={}),
    ])

    cmp = compare_ab(
        path,
        ("arm-a", {"CAIRN_CROSS_ENCODER": "0"}),
        ("arm-b", {"CAIRN_CROSS_ENCODER": "0"}),
        src_db=store.db_path,
        top_k=5,
    )
    for metric, d in cmp["delta"].items():
        assert abs(d) < 1e-9, f"nonzero delta on identical variants: {metric}={d}"
    assert cmp["sign_test"]["p_value"] == 1.0


# --- follow-up batch: replay probes, dedup-suppress, confidence signal ------

def test_query_log_replay_topics(store, monkeypatch):
    from cairn.evaluation import probe_set as ps
    for q in ("how do we tune the kalman filter noise", "deploy rollback steps for canary"):
        store._conn.execute(
            "INSERT INTO query_log (ts, query_text, result_count, top_score) VALUES ('2026-07-21T00:00:00', ?, 1, 0.5)", (q,))
    store._commit()
    topics = ps._query_log_topics(store, 10)
    assert len(topics) == 2 and topics[0]["event_type"] == "query_log"


def test_build_probes_from_query_log(store, monkeypatch):
    from cairn.evaluation import probe_set as ps
    nid = store.store(content="Kalman filter noise tuning uses process noise 0.01")
    store._conn.execute(
        "INSERT INTO query_log (ts, query_text, result_count, top_score) VALUES ('2026-07-21T00:00:00', 'kalman filter noise tuning', 1, 0.9)")
    store._commit()
    monkeypatch.setattr(ps, "judge_grade", lambda q, c: 3)
    payload = ps.build_probe_set(store, size=5, out_path=str(__import__('tempfile').mktemp()), from_query_log=True)
    assert payload["probe_count"] == 1
    assert payload["probes"][0]["method"] == "query-log-replay"
    assert nid in payload["probes"][0]["qrels"]


def test_auto_capture_does_not_pollute_query_log(tmp_cairn_dir, monkeypatch):
    monkeypatch.delenv("CAIRN_QUERY_LOG", raising=False)
    from cairn.bridge import reset_memory, auto_capture, _get_store
    reset_memory()
    auto_capture(content="A decision about database indexes worth deduping later",
                 event_type="decision")
    n = _get_store()._conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    reset_memory()
    assert n == 0, "internal dedup queries leaked into the replay corpus"


def test_confidence_signal_env_gated(store, monkeypatch):
    class _Node:
        metadata = {"event_type": "memory", "capture_confidence": "low"}
        last_accessed = None; created_at = None; access_count = 0
    base = store._metadata_score_factor(_Node())
    monkeypatch.setenv("CAIRN_CONFIDENCE_SIGNAL", "1")
    gated = store._metadata_score_factor(_Node())
    assert gated < base  # low confidence demoted only when enabled
