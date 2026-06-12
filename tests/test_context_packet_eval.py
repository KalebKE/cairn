"""Tests for context packet evaluation."""

import json
from dataclasses import asdict
from unittest.mock import patch

import pytest

from omega.evaluation.context_packet_eval import (
    PacketEvalReport,
    backfill_packet_miss_edges,
    backfill_packet_miss_report,
    backfill_source_edges,
    diagnose_context_packet_report,
    filter_packet_miss_probes,
    format_packet_diagnosis,
    format_packet_report,
    run_context_packet_maintenance_loop,
    run_context_packet_evaluation,
    _classify_packet_miss,
)
from omega.sqlite_store import SQLiteStore


def _seed_store(tmp_omega_dir):
    store = SQLiteStore(str(tmp_omega_dir / "omega.db"))
    memories = [
        ("Decision: auth handlers validate JWT tokens before database lookup.", "decision"),
        ("Lesson: OAuth callbacks must validate state nonce to prevent replay.", "lesson_learned"),
        ("Constraint: public launch copy must not mention confidential pilot pricing.", "constraint"),
        ("Error pattern: webhook delivery fails when endpoint returns 503.", "error_pattern"),
        ("Preference: use pytest for context packet eval tests.", "user_preference"),
    ]
    for content, event_type in memories:
        store.store(content=content, metadata={"event_type": event_type})
    return store


def test_run_context_packet_evaluation_basic(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        report = run_context_packet_evaluation(
            db_path=str(tmp_omega_dir / "omega.db"),
            sample_size=5,
            budget_tokens=500,
            seed=7,
        )
    finally:
        store.close()

    assert report.sample_size > 0
    assert 0.0 <= report.packet_hit_rate <= 1.0
    assert 0.0 <= report.packet_precision_proxy <= 1.0
    assert 0.0 <= report.budget_compliance <= 1.0
    assert report.budget_tokens == 500
    assert report.probe_results
    assert all("packet_tokens" in row for row in report.probe_results)
    assert all("source_edge_count" in row for row in report.probe_results)
    assert all("rendered_warnings_count" in row for row in report.probe_results)
    assert all("source_candidate_hit" in row for row in report.probe_results)
    assert all("miss_reason" in row for row in report.probe_results)
    assert 0.0 <= report.source_edge_coverage <= 1.0
    assert 0.0 <= report.missed_source_edge_coverage <= 1.0
    assert 0.0 <= report.rendered_warning_rate <= 1.0
    assert report.avg_warnings >= 0.0
    assert report.avg_rendered_warnings >= 0.0


def test_context_packet_evaluation_writes_output_and_probe_cache(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    output = tmp_omega_dir / "packet-report.json"
    cache = tmp_omega_dir / "packet-probes.json"
    try:
        report = run_context_packet_evaluation(
            db_path=str(tmp_omega_dir / "omega.db"),
            sample_size=4,
            budget_tokens=500,
            output_path=str(output),
            probe_cache_path=str(cache),
        )
    finally:
        store.close()

    assert output.exists()
    assert cache.exists()
    loaded = json.loads(output.read_text())
    assert loaded["sample_size"] == report.sample_size
    assert "packet_hit_rate" in loaded


def test_context_packet_evaluation_restores_access_state(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        before = store._conn.execute(
            "SELECT node_id, access_count, last_accessed FROM memories ORDER BY node_id"
        ).fetchall()
        run_context_packet_evaluation(
            db_path=str(tmp_omega_dir / "omega.db"),
            sample_size=4,
            budget_tokens=500,
            seed=11,
        )
        after = store._conn.execute(
            "SELECT node_id, access_count, last_accessed FROM memories ORDER BY node_id"
        ).fetchall()
    finally:
        store.close()

    assert after == before


def test_format_packet_report_includes_core_metrics():
    report = PacketEvalReport(
        timestamp="2026-06-12T00:00:00+00:00",
        sample_size=2,
        budget_tokens=800,
        mode="before_edit",
        total_memories=10,
        packet_hit_rate=0.5,
        packet_precision_proxy=0.25,
        chain_hit_rate=0.5,
        source_edge_coverage=0.75,
        missed_source_edge_coverage=0.25,
        budget_compliance=1.0,
        warning_rate=0.0,
        rendered_warning_rate=0.0,
        abstention_rate=0.5,
        avg_packet_tokens=300.0,
        avg_memories_used=2.0,
        avg_warnings=1.5,
        avg_rendered_warnings=0.5,
        by_miss_reason={
            "no_source_edges": {
                "count": 1,
                "share_of_misses": 1.0,
                "avg_source_edges": 0.0,
            }
        },
    )

    text = format_packet_report(report)
    assert "Context Packet Evaluation Report" in text
    assert "Packet Hit Rate" in text
    assert "Budget Compliance" in text
    assert "Rendered Warning Rate" in text
    assert "Avg Rendered Warnings" in text
    assert "Source Edge Coverage" in text
    assert "By Miss Reason" in text
    assert "no_source_edges" in text
    assert "50.0%" in text


def test_classify_packet_miss_reasons():
    assert _classify_packet_miss(
        {"warnings": []},
        source_id="mem-hit",
        source_hit=True,
        source_candidate_hit=True,
        chain_hit=False,
        source_edge_count=0,
        budget_ok=True,
    ) == ""
    assert _classify_packet_miss(
        {"warnings": [{"reason": "stale", "item": {"id": "mem-stale"}}]},
        source_id="mem-stale",
        source_hit=False,
        source_candidate_hit=True,
        chain_hit=False,
        source_edge_count=1,
        budget_ok=True,
    ) == "warning_filtered:stale"
    assert _classify_packet_miss(
        {"warnings": []},
        source_id="mem-orphan",
        source_hit=False,
        source_candidate_hit=False,
        chain_hit=False,
        source_edge_count=0,
        budget_ok=True,
    ) == "no_source_edges"
    assert _classify_packet_miss(
        {"warnings": [], "metrics": {"memories_used": 4}},
        source_id="mem-missing-candidate",
        source_hit=False,
        source_candidate_hit=False,
        chain_hit=False,
        source_edge_count=2,
        budget_ok=True,
    ) == "candidate_not_retrieved"
    assert _classify_packet_miss(
        {"warnings": [], "metrics": {"memories_used": 8}},
        source_id="mem-evicted",
        source_hit=False,
        source_candidate_hit=True,
        chain_hit=False,
        source_edge_count=2,
        budget_ok=True,
    ) == "rank_or_render_cap_eviction"
    assert _classify_packet_miss(
        {"warnings": [], "metrics": {"memories_used": 8}},
        source_id="mem-chain",
        source_hit=False,
        source_candidate_hit=True,
        chain_hit=True,
        source_edge_count=2,
        budget_ok=True,
    ) == "chain_not_rendered"
    assert _classify_packet_miss(
        {"warnings": [], "metrics": {"memories_used": 4}},
        source_id="mem-low-rank",
        source_hit=False,
        source_candidate_hit=True,
        chain_hit=False,
        source_edge_count=2,
        budget_ok=True,
    ) == "edge_not_retrieved_or_ranked"


def test_cli_eval_context_packet_json(capsys):
    from argparse import Namespace
    from omega.cli import cmd_eval_context_packet

    report = PacketEvalReport(
        timestamp="2026-06-12T00:00:00+00:00",
        sample_size=1,
        budget_tokens=500,
        mode="before_edit",
        total_memories=3,
        packet_hit_rate=1.0,
    )

    with patch(
        "omega.evaluation.context_packet_eval.run_context_packet_evaluation",
        return_value=report,
    ):
        cmd_eval_context_packet(Namespace(
            sample_size=1,
            budget_tokens=500,
            mode="before_edit",
            seed=42,
            output=None,
            probe_cache=None,
            json=True,
        ))

    out = capsys.readouterr().out
    parsed = json.loads(out[out.index("{"):])
    assert parsed == asdict(report)


def test_cli_backfill_context_packet_json(capsys):
    from argparse import Namespace
    from omega.cli import cmd_backfill_context_packet

    manifest = {
        "report_path": "packet-report.json",
        "total_probes": 2,
        "eligible_misses": 1,
        "candidate_edges": 1,
        "created": 1,
        "skipped": 0,
        "capped_edges": 0,
        "missing": 0,
        "max_edges": 10,
        "dry_run": True,
        "similarity_threshold": 0.72,
        "event_types": ["decision"],
        "edges": [],
    }

    with patch("omega.bridge._get_store", return_value=object()), patch(
        "omega.evaluation.context_packet_eval.backfill_packet_miss_report",
        return_value=manifest,
    ) as mocked:
        cmd_backfill_context_packet(Namespace(
            report="packet-report.json",
            threshold=0.72,
            max_connections_per_source=1,
            max_edges=10,
            event_types="decision",
            apply=False,
            output=None,
            json=True,
        ))

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert parsed == manifest
    assert mocked.call_args.kwargs["dry_run"] is True
    assert mocked.call_args.kwargs["max_edges"] == 10
    assert mocked.call_args.kwargs["event_types"] == ["decision"]


def test_cli_backfill_context_packet_apply_requires_pro(capsys):
    from argparse import Namespace
    from omega.cli import cmd_backfill_context_packet

    with patch("omega.cli._is_pro_licensed", return_value=False), pytest.raises(SystemExit) as exc:
        cmd_backfill_context_packet(Namespace(
            report="packet-report.json",
            threshold=0.72,
            max_connections_per_source=1,
            max_edges=10,
            event_types="decision",
            apply=True,
            output=None,
            json=False,
        ))

    assert exc.value.code == 1
    assert "requires OMEGA Pro" in capsys.readouterr().err


def test_cli_backfill_context_packet_apply_allows_pro(capsys):
    from argparse import Namespace
    from omega.cli import cmd_backfill_context_packet

    manifest = {
        "report_path": "packet-report.json",
        "total_probes": 2,
        "eligible_misses": 1,
        "candidate_edges": 1,
        "created": 1,
        "skipped": 0,
        "capped_edges": 0,
        "missing": 0,
        "max_edges": 10,
        "dry_run": False,
        "similarity_threshold": 0.72,
        "event_types": ["decision"],
        "edges": [],
    }

    with patch("omega.cli._is_pro_licensed", return_value=True), patch(
        "omega.bridge._get_store", return_value=object()
    ), patch(
        "omega.evaluation.context_packet_eval.backfill_packet_miss_report",
        return_value=manifest,
    ) as mocked:
        cmd_backfill_context_packet(Namespace(
            report="packet-report.json",
            threshold=0.72,
            max_connections_per_source=1,
            max_edges=10,
            event_types="decision",
            apply=True,
            output=None,
            json=True,
        ))

    assert json.loads(capsys.readouterr().out) == manifest
    assert mocked.call_args.kwargs["dry_run"] is False


def test_diagnose_context_packet_report_explains_miss(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    output = tmp_omega_dir / "diagnosis.json"
    try:
        ids = store.get_recent(limit=2)
        source_id = ids[0].id
        rendered_id = ids[1].id
        store.add_edge(source_id, rendered_id, edge_type="related", weight=0.8)
        report_path = tmp_omega_dir / "packet-report.json"
        report_path.write_text(json.dumps({
            "sample_size": 1,
            "packet_hit_rate": 0.0,
            "by_miss_reason": {
                "chain_not_rendered": {
                    "count": 1,
                    "share_of_misses": 1.0,
                    "avg_source_edges": 1.0,
                }
            },
            "probe_results": [
                {
                    "query": "auth state nonce",
                    "source_id": source_id,
                    "source_type": "decision",
                    "source_hit": False,
                    "source_candidate_hit": True,
                    "chain_hit": True,
                    "source_edge_count": 1,
                    "memory_ids_used": [rendered_id],
                    "packet_tokens": 300,
                    "miss_reason": "chain_not_rendered",
                }
            ],
        }))

        diagnosis = diagnose_context_packet_report(
            store,
            str(report_path),
            output_path=str(output),
        )
    finally:
        store.close()

    assert diagnosis["diagnosed"] == 1
    miss = diagnosis["diagnoses"][0]
    assert miss["source"]["id"] == source_id
    assert miss["rendered"][0]["id"] == rendered_id
    assert miss["source_edges"][0]["neighbor_id"] == rendered_id
    assert miss["source_edges"][0]["neighbor_rendered"] is True
    assert "rank_trace" in miss
    assert miss["rank_trace"]["candidate_count"] >= 1
    assert "lost final rendering" in miss["recommendation"]
    assert output.exists()
    formatted = format_packet_diagnosis(diagnosis)
    assert "Context Packet Diagnosis" in formatted
    assert "Source rank:" in formatted


def test_cli_diagnose_context_packet_json(capsys):
    from argparse import Namespace
    from omega.cli import cmd_diagnose_context_packet

    result = {
        "report_path": "packet-report.json",
        "diagnosed": 1,
        "diagnoses": [],
    }

    with patch("omega.bridge._get_store", return_value=object()), patch(
        "omega.evaluation.context_packet_eval.diagnose_context_packet_report",
        return_value=result,
    ) as mocked:
        cmd_diagnose_context_packet(Namespace(
            report="packet-report.json",
            limit=3,
            include_hits=False,
            output=None,
            json=True,
        ))

    assert json.loads(capsys.readouterr().out) == result
    assert mocked.call_args.args[1] == "packet-report.json"
    assert mocked.call_args.kwargs["limit"] == 3
    assert mocked.call_args.kwargs["include_hits"] is False


def test_cli_maintain_context_packet_json(capsys):
    from argparse import Namespace
    from omega.cli import cmd_maintain_context_packet

    result = {
        "artifact_prefix": "packet-loop",
        "before_report": "packet-loop-before.json",
        "backfill_manifest": "packet-loop-backfill.json",
        "after_report": None,
        "applied": False,
        "re_evaluated": False,
        "before": {"packet_hit_rate": 0.5},
        "backfill": {"created": 1},
    }

    with patch("omega.cli._is_pro_licensed", return_value=True), patch(
        "omega.bridge._get_store", return_value=object()
    ), patch(
        "omega.evaluation.context_packet_eval.run_context_packet_maintenance_loop",
        return_value=result,
    ) as mocked:
        cmd_maintain_context_packet(Namespace(
            artifact_prefix="packet-loop",
            sample_size=5,
            budget_tokens=500,
            mode="before_edit",
            seed=42,
            probe_cache="packet-probes.json",
            threshold=0.72,
            max_connections_per_source=1,
            max_edges=10,
            event_types="decision",
            apply=False,
            re_eval=False,
            json=True,
        ))

    parsed = json.loads(capsys.readouterr().out)
    assert parsed == result
    assert mocked.call_args.kwargs["artifact_prefix"] == "packet-loop"
    assert mocked.call_args.kwargs["apply"] is False
    assert mocked.call_args.kwargs["re_eval"] is False
    assert mocked.call_args.kwargs["event_types"] == ["decision"]


def test_cli_maintain_context_packet_requires_pro(capsys):
    from argparse import Namespace
    from omega.cli import cmd_maintain_context_packet

    with patch("omega.cli._is_pro_licensed", return_value=False), pytest.raises(SystemExit) as exc:
        cmd_maintain_context_packet(Namespace(
            artifact_prefix="packet-loop",
            sample_size=5,
            budget_tokens=500,
            mode="before_edit",
            seed=42,
            probe_cache=None,
            threshold=0.72,
            max_connections_per_source=1,
            max_edges=10,
            event_types=None,
            apply=False,
            re_eval=False,
            json=True,
        ))

    assert exc.value.code == 1
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["requires_pro"] is True
    assert "requires OMEGA Pro" in parsed["error"]


def test_backfill_source_edges_dry_run_does_not_mutate(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        source_id = store.get_by_type("decision", limit=1)[0].id
        result = backfill_source_edges(
            store,
            [source_id],
            similarity_threshold=0.0,
            max_connections_per_source=1,
            dry_run=True,
        )
        edge_count = store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
            (source_id, source_id),
        ).fetchone()[0]
    finally:
        store.close()

    assert result["dry_run"] is True
    assert result["created"] >= 0
    assert edge_count == 0


def test_backfill_source_edges_creates_typed_edge(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        source_id = store.get_by_type("decision", limit=1)[0].id
        result = backfill_source_edges(
            store,
            [source_id],
            similarity_threshold=0.0,
            max_connections_per_source=1,
            dry_run=False,
        )
        rows = store._conn.execute(
            "SELECT edge_type, metadata FROM edges WHERE source_id = ? OR target_id = ?",
            (source_id, source_id),
        ).fetchall()
    finally:
        store.close()

    assert result["dry_run"] is False
    assert result["created"] >= 1
    assert rows
    assert rows[0][0] in {"evolution", "related", "temporal_cluster"}
    assert "context_packet_eval_backfill" in rows[0][1]


def test_backfill_packet_miss_edges_links_source_to_used_memory(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        ids = store.get_recent(limit=2)
        source_id = ids[0].id
        target_id = ids[1].id
        probe_results = [{
            "source_hit": False,
            "source_id": source_id,
            "memory_ids_used": [target_id],
        }]
        result = backfill_packet_miss_edges(
            store,
            probe_results,
            similarity_threshold=0.0,
            max_connections_per_source=1,
            dry_run=False,
        )
        rows = store._conn.execute(
            """SELECT edge_type, metadata FROM edges
               WHERE (source_id = ? AND target_id = ?)
                  OR (source_id = ? AND target_id = ?)""",
            (source_id, target_id, target_id, source_id),
        ).fetchall()
    finally:
        store.close()

    assert result["created"] == 1
    assert rows
    assert rows[0][0] in {"evolution", "related", "temporal_cluster"}
    assert "context_packet_miss_backfill" in rows[0][1]


def test_backfill_packet_miss_edges_respects_global_cap(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    try:
        ids = store.get_recent(limit=3)
        probe_results = [
            {
                "source_hit": False,
                "source_id": ids[0].id,
                "memory_ids_used": [ids[1].id],
            },
            {
                "source_hit": False,
                "source_id": ids[1].id,
                "memory_ids_used": [ids[2].id],
            },
        ]
        result = backfill_packet_miss_edges(
            store,
            probe_results,
            similarity_threshold=0.0,
            max_connections_per_source=1,
            max_edges=1,
            dry_run=False,
        )
        edge_count = store._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    finally:
        store.close()

    assert result["candidate_edges"] == 2
    assert result["created"] == 1
    assert result["capped_edges"] == 1
    assert edge_count == 1


def test_filter_packet_miss_probes_respects_event_type_allowlist():
    probes = [
        {
            "source_hit": False,
            "source_id": "mem-decision",
            "source_type": "decision",
            "memory_ids_used": ["mem-used"],
        },
        {
            "source_hit": False,
            "source_id": "mem-fact",
            "source_type": "user_fact",
            "memory_ids_used": ["mem-used"],
        },
        {
            "source_hit": True,
            "source_id": "mem-hit",
            "source_type": "decision",
            "memory_ids_used": ["mem-hit"],
        },
    ]

    eligible = filter_packet_miss_probes(probes, event_types=["decision"])

    assert [probe["source_id"] for probe in eligible] == ["mem-decision"]


def test_backfill_packet_miss_report_dry_run_writes_manifest(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    output = tmp_omega_dir / "backfill-manifest.json"
    try:
        ids = store.get_recent(limit=2)
        source_id = ids[0].id
        target_id = ids[1].id
        report = {
            "probe_results": [
                {
                    "source_hit": False,
                    "source_id": source_id,
                    "source_type": "decision",
                    "memory_ids_used": [target_id],
                }
            ]
        }
        report_path = tmp_omega_dir / "packet-report.json"
        report_path.write_text(json.dumps(report))

        result = backfill_packet_miss_report(
            store,
            str(report_path),
            similarity_threshold=0.0,
            max_connections_per_source=1,
            max_edges=10,
            dry_run=True,
            output_path=str(output),
        )
        edge_count = store._conn.execute(
            "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
            (source_id, source_id),
        ).fetchone()[0]
    finally:
        store.close()

    assert result["dry_run"] is True
    assert result["eligible_misses"] == 1
    assert result["candidate_edges"] == 1
    assert result["created"] == 1
    assert result["capped_edges"] == 0
    assert edge_count == 0
    assert json.loads(output.read_text())["created"] == 1


def test_context_packet_maintenance_loop_dry_run_skips_after_eval(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    before = PacketEvalReport(
        timestamp="2026-06-12T00:00:00+00:00",
        sample_size=1,
        budget_tokens=500,
        mode="before_edit",
        total_memories=3,
        packet_hit_rate=0.5,
        packet_precision_proxy=0.25,
        chain_hit_rate=0.1,
        warning_rate=0.2,
        rendered_warning_rate=0.1,
        avg_packet_tokens=300.0,
        avg_memories_used=2.0,
        avg_warnings=1.0,
        avg_rendered_warnings=0.5,
    )
    manifest = {
        "report_path": str(tmp_omega_dir / "loop-before.json"),
        "total_probes": 1,
        "eligible_misses": 1,
        "candidate_edges": 1,
        "created": 1,
        "skipped": 0,
        "capped_edges": 0,
        "missing": 0,
        "max_edges": 10,
        "dry_run": True,
        "edges": [],
    }
    try:
        with patch(
            "omega.evaluation.context_packet_eval.run_context_packet_evaluation",
            return_value=before,
        ) as eval_mock, patch(
            "omega.evaluation.context_packet_eval.backfill_packet_miss_report",
            return_value=manifest,
        ) as backfill_mock:
            result = run_context_packet_maintenance_loop(
                store,
                artifact_prefix=str(tmp_omega_dir / "loop"),
                sample_size=1,
                budget_tokens=500,
                apply=False,
                re_eval=True,
            )
    finally:
        store.close()

    assert eval_mock.call_count == 1
    assert backfill_mock.call_args.kwargs["dry_run"] is True
    assert result["before_report"].endswith("loop-before.json")
    assert result["backfill_manifest"].endswith("loop-backfill.json")
    assert result["after_report"] is None
    assert result["applied"] is False
    assert result["re_evaluated"] is False
    assert result["before"]["rendered_warning_rate"] == 0.1
    assert result["before"]["avg_rendered_warnings"] == 0.5


def test_context_packet_maintenance_loop_apply_re_eval_runs_after(tmp_omega_dir):
    store = _seed_store(tmp_omega_dir)
    before = PacketEvalReport(
        timestamp="2026-06-12T00:00:00+00:00",
        sample_size=1,
        budget_tokens=500,
        mode="before_edit",
        total_memories=3,
        packet_hit_rate=0.5,
    )
    after = PacketEvalReport(
        timestamp="2026-06-12T00:01:00+00:00",
        sample_size=1,
        budget_tokens=500,
        mode="before_edit",
        total_memories=3,
        packet_hit_rate=1.0,
        rendered_warning_rate=0.0,
        avg_rendered_warnings=0.0,
    )
    try:
        with patch(
            "omega.evaluation.context_packet_eval.run_context_packet_evaluation",
            side_effect=[before, after],
        ) as eval_mock, patch(
            "omega.evaluation.context_packet_eval.backfill_packet_miss_report",
            return_value={"created": 1, "eligible_misses": 1, "total_probes": 1},
        ) as backfill_mock:
            result = run_context_packet_maintenance_loop(
                store,
                artifact_prefix=str(tmp_omega_dir / "loop-apply"),
                sample_size=1,
                budget_tokens=500,
                apply=True,
                re_eval=True,
            )
    finally:
        store.close()

    assert eval_mock.call_count == 2
    assert backfill_mock.call_args.kwargs["dry_run"] is False
    assert result["after_report"].endswith("loop-apply-after.json")
    assert result["re_evaluated"] is True
    assert result["after"]["packet_hit_rate"] == 1.0
    assert result["after"]["rendered_warning_rate"] == 0.0
    assert result["after"]["avg_rendered_warnings"] == 0.0
