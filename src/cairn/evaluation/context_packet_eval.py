# Copyright 2025-2026 Kokyo Keisho Zaidan Stichting
# SPDX-License-Identifier: Apache-2.0
"""Context packet quality evaluation for Cairn.

This evaluates the user-facing context packet surface, not raw retrieval.
It generates probes from known memories, builds packets for those probes, and
measures whether the source memory appears in the packet within budget.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cairn.evaluation.retrieval_eval import Probe, generate_keyword_probe, sample_memories
from cairn.server.context_handlers import build_context_packet, _estimate_tokens

DEFAULT_PACKET_MISS_EVENT_TYPES = (
    "decision",
    "lesson_learned",
    "error_pattern",
    "constraint",
    "user_preference",
    "task_completion",
)


@dataclass
class PacketProbeResult:
    """Results for one context packet probe."""

    query: str
    source_id: str
    source_type: str
    packet_tokens: int
    budget_tokens: int
    memories_used: int
    warnings_count: int
    rendered_warnings_count: int
    chain_count: int
    source_edge_count: int
    memory_ids_used: List[str]
    source_hit: bool
    source_candidate_hit: bool
    chain_hit: bool
    budget_ok: bool
    abstained: bool
    miss_reason: str = ""


@dataclass
class PacketEvalReport:
    """Complete packet evaluation report."""

    timestamp: str
    sample_size: int
    budget_tokens: int
    mode: str
    total_memories: int
    seed: int = 42

    packet_hit_rate: float = 0.0
    packet_precision_proxy: float = 0.0
    chain_hit_rate: float = 0.0
    source_edge_coverage: float = 0.0
    missed_source_edge_coverage: float = 0.0
    budget_compliance: float = 0.0
    warning_rate: float = 0.0
    rendered_warning_rate: float = 0.0
    abstention_rate: float = 0.0
    avg_packet_tokens: float = 0.0
    avg_memories_used: float = 0.0
    avg_warnings: float = 0.0
    avg_rendered_warnings: float = 0.0
    avg_tokens_saved: float = 0.0
    duration_seconds: float = 0.0

    by_event_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_miss_reason: Dict[str, Dict[str, float]] = field(default_factory=dict)
    probe_results: List[Dict[str, Any]] = field(default_factory=list)


def _typed_edge_for_nodes(source: Any, target: Any, similarity: float) -> Optional[str]:
    source_meta = source.metadata or {}
    target_meta = target.metadata or {}
    source_entity = source_meta.get("entity_id") or ""
    target_entity = target_meta.get("entity_id") or ""
    source_event = source_meta.get("event_type") or ""
    target_event = target_meta.get("event_type") or ""

    if source_entity and target_entity and source_entity == target_entity:
        return "same_entity"
    if source_event and target_event and source_event == target_event and similarity >= 0.75:
        return "evolution"
    if getattr(source, "created_at", None) and getattr(target, "created_at", None):
        try:
            if abs((source.created_at - target.created_at).total_seconds()) <= 3600:
                return "temporal_cluster"
        except Exception:
            pass
    if similarity >= 0.80:
        return "related"
    return None


def backfill_source_edges(
    store,
    source_ids: List[str],
    *,
    similarity_threshold: float = 0.78,
    max_connections_per_source: int = 2,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Create typed edges for explicit source IDs, usually packet-eval misses."""
    created: List[Dict[str, Any]] = []
    skipped = 0
    missing = 0

    for source_id in source_ids:
        source = store.get_node(source_id, track_access=False)
        if source is None:
            missing += 1
            continue
        embedding = store.get_embedding(source_id)
        if not embedding:
            missing += 1
            continue
        similar = store.find_similar(embedding, limit=max_connections_per_source + 8)
        made = 0
        for target in similar:
            if made >= max_connections_per_source:
                break
            if target.id == source_id or target.relevance < similarity_threshold:
                continue
            edge_type = _typed_edge_for_nodes(source, target, target.relevance)
            if edge_type is None:
                continue

            try:
                with store._lock:
                    exists = store._conn.execute(
                        """SELECT 1 FROM edges
                           WHERE ((source_id = ? AND target_id = ?)
                              OR (source_id = ? AND target_id = ?))
                           AND edge_type = ?""",
                        (source_id, target.id, target.id, source_id, edge_type),
                    ).fetchone()
            except Exception:
                exists = None
            if exists:
                skipped += 1
                continue

            row = {
                "source_id": source_id,
                "target_id": target.id,
                "edge_type": edge_type,
                "weight": round(float(target.relevance), 3),
            }
            if not dry_run:
                store.add_edge(
                    source_id,
                    target.id,
                    edge_type=edge_type,
                    weight=target.relevance,
                    metadata={"source": "context_packet_eval_backfill", "auto": True, "typed": edge_type},
                )
            created.append(row)
            made += 1

    return {
        "sources": len(source_ids),
        "created": len(created),
        "skipped": skipped,
        "missing": missing,
        "dry_run": dry_run,
        "edges": created,
    }


def backfill_packet_miss_edges(
    store,
    probe_results: List[Dict[str, Any]],
    *,
    similarity_threshold: float = 0.72,
    max_connections_per_source: int = 2,
    max_edges: Optional[int] = None,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Link missed sources to memories already used in their packets."""
    created: List[Dict[str, Any]] = []
    skipped = 0
    missing = 0
    candidate_edges = 0
    capped_edges = 0
    global_cap = max_edges if max_edges and max_edges > 0 else None

    for probe in probe_results:
        if probe.get("source_hit"):
            continue
        source_id = probe.get("source_id")
        source = store.get_node(source_id, track_access=False) if source_id else None
        source_emb = store.get_embedding(source_id) if source_id else None
        if source is None or not source_emb:
            missing += 1
            continue

        candidates: List[tuple[Any, float]] = []
        for target_id in probe.get("memory_ids_used") or []:
            if target_id == source_id:
                continue
            target = store.get_node(target_id, track_access=False)
            target_emb = store.get_embedding(target_id)
            if target is None or not target_emb:
                continue
            try:
                from cairn.sqlite_store._types import _cosine_similarity
                similarity = _cosine_similarity(source_emb, target_emb)
            except Exception:
                similarity = 0.0
            if similarity >= similarity_threshold:
                candidates.append((target, similarity))

        candidates.sort(key=lambda row: row[1], reverse=True)
        made = 0
        for target, similarity in candidates:
            if made >= max_connections_per_source:
                break
            edge_type = _typed_edge_for_nodes(source, target, similarity)
            if edge_type is None:
                continue
            try:
                with store._lock:
                    exists = store._conn.execute(
                        """SELECT 1 FROM edges
                           WHERE ((source_id = ? AND target_id = ?)
                              OR (source_id = ? AND target_id = ?))
                           AND edge_type = ?""",
                        (source_id, target.id, target.id, source_id, edge_type),
                    ).fetchone()
            except Exception:
                exists = None
            if exists:
                skipped += 1
                continue
            row = {
                "source_id": source_id,
                "target_id": target.id,
                "edge_type": edge_type,
                "weight": round(float(similarity), 3),
            }
            candidate_edges += 1
            made += 1
            if global_cap is not None and len(created) >= global_cap:
                capped_edges += 1
                continue
            if not dry_run:
                store.add_edge(
                    source_id,
                    target.id,
                    edge_type=edge_type,
                    weight=similarity,
                    metadata={"source": "context_packet_miss_backfill", "auto": True, "typed": edge_type},
                )
            created.append(row)

    return {
        "probes": len(probe_results),
        "candidate_edges": candidate_edges,
        "created": len(created),
        "skipped": skipped,
        "capped_edges": capped_edges,
        "missing": missing,
        "max_edges": global_cap,
        "dry_run": dry_run,
        "edges": created,
    }


def filter_packet_miss_probes(
    probe_results: List[Dict[str, Any]],
    *,
    event_types: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Return packet misses eligible for guarded graph backfill."""
    allowed = set(event_types or DEFAULT_PACKET_MISS_EVENT_TYPES)
    eligible: List[Dict[str, Any]] = []
    for probe in probe_results:
        if probe.get("source_hit"):
            continue
        if not probe.get("source_id"):
            continue
        if not probe.get("memory_ids_used"):
            continue
        source_type = probe.get("source_type") or ""
        if allowed and source_type not in allowed:
            continue
        eligible.append(probe)
    return eligible


def load_packet_eval_report(path: str) -> Dict[str, Any]:
    """Load a serialized context packet eval report."""
    with open(path) as f:
        report = json.load(f)
    if not isinstance(report, dict):
        raise ValueError(f"Packet eval report at {path} is not a JSON object")
    probes = report.get("probe_results")
    if not isinstance(probes, list):
        raise ValueError(f"Packet eval report at {path} has no probe_results list")
    return report


def _snippet(text: str, max_chars: int = 180) -> str:
    snippet = " ".join((text or "").strip().split())
    if len(snippet) > max_chars:
        return snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _memory_summary(store, node_id: str) -> Dict[str, Any]:
    memory = store.get_node(node_id, track_access=False) if node_id else None
    if memory is None:
        return {"id": node_id, "exists": False}
    metadata = memory.metadata or {}
    return {
        "id": node_id,
        "exists": True,
        "event_type": metadata.get("event_type") or "memory",
        "status": getattr(memory, "status", None) or metadata.get("status") or "active",
        "sensitivity": metadata.get("sensitivity") or "internal",
        "created_at": memory.created_at.isoformat() if getattr(memory, "created_at", None) else "",
        "access_count": getattr(memory, "access_count", 0) or 0,
        "snippet": _snippet(memory.content),
    }


def _source_edges(store, source_id: str, rendered_ids: set[str], limit: int = 12) -> List[Dict[str, Any]]:
    try:
        with store._lock:
            rows = store._conn.execute(
                """SELECT source_id, target_id, edge_type, weight, metadata
                   FROM edges
                   WHERE source_id = ? OR target_id = ?
                   ORDER BY weight DESC
                   LIMIT ?""",
                (source_id, source_id, limit),
            ).fetchall()
    except Exception:
        return []

    edges = []
    for src, target, edge_type, weight, metadata_raw in rows:
        neighbor_id = target if src == source_id else src
        try:
            metadata = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            metadata = {}
        edges.append({
            "neighbor_id": neighbor_id,
            "edge_type": edge_type,
            "weight": round(float(weight or 0.0), 3),
            "edge_source": metadata.get("source") or "",
            "typed": bool(metadata.get("typed")),
            "neighbor_rendered": neighbor_id in rendered_ids,
            "neighbor": _memory_summary(store, neighbor_id),
        })
    return edges


def _rendered_similarity(store, source_id: str, rendered_ids: List[str], limit: int = 5) -> List[Dict[str, Any]]:
    source_emb = store.get_embedding(source_id) if source_id else None
    if not source_emb:
        return []
    rows = []
    try:
        from cairn.sqlite_store._types import _cosine_similarity
    except Exception:
        return []
    for target_id in rendered_ids:
        if target_id == source_id:
            continue
        target_emb = store.get_embedding(target_id)
        if not target_emb:
            continue
        try:
            similarity = _cosine_similarity(source_emb, target_emb)
        except Exception:
            similarity = 0.0
        rows.append({
            "id": target_id,
            "similarity": round(float(similarity), 4),
            "memory": _memory_summary(store, target_id),
        })
    rows.sort(key=lambda row: row["similarity"], reverse=True)
    return rows[:limit]


def _miss_recommendation(probe: Dict[str, Any], edges: List[Dict[str, Any]], nearest: List[Dict[str, Any]]) -> str:
    reason = probe.get("miss_reason") or "unknown"
    if reason == "candidate_not_retrieved":
        rendered_edge = any(edge.get("neighbor_rendered") for edge in edges)
        if rendered_edge:
            return "Candidate generation missed the source even though a graph neighbor rendered; inspect edge traversal/ranking from rendered neighbor."
        if edges:
            return "Source has edges, but none connect to rendered packet items; inspect edge quality before adding retrieval breadth."
        return "Source has no usable graph path; consider memory consolidation or targeted edge creation."
    if reason == "chain_not_rendered":
        return "Source reached candidate/chain space but lost final rendering; inspect rank score and section competition."
    if reason == "rank_or_render_cap_eviction":
        return "Source was a candidate but lost to render cap; inspect top rendered competitors and scoring weights."
    if reason == "no_source_edges":
        if nearest:
            return "No source edges exist; nearest rendered memory may be a safe backfill candidate if semantically justified."
        return "No source edges exist and no rendered similarity signal was found; likely needs memory consolidation or better source capture."
    if reason.startswith("warning_filtered:"):
        return "Source was intentionally filtered into warning path; inspect status/contradiction metadata rather than ranking."
    return "Inspect rendered competitors, edge path, and candidate status before changing retrieval."


def _rank_trace_for_probe(store, probe: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    query = probe.get("query") or ""
    source_id = probe.get("source_id") or ""
    if not query:
        return {}
    try:
        packet = build_context_packet(
            store,
            task=query,
            files=[],
            scope=None,
            mode=report.get("mode") or "before_edit",
            budget_tokens=int(report.get("budget_tokens") or 800),
            include_receipt=True,
        )
    except Exception:
        return {}

    receipts = packet.get("candidate_receipts") or []
    source_receipt = next((row for row in receipts if row.get("id") == source_id), None)
    rendered = [row for row in receipts if row.get("rendered")]
    rendered.sort(key=lambda row: row.get("render_rank") or 9999)
    if source_receipt:
        source_rank = int(source_receipt.get("rank") or 9999)
        competitors = [
            row for row in receipts
            if row.get("admitted") and int(row.get("rank") or 9999) < source_rank
        ][:8]
    else:
        competitors = rendered[:8]
    return {
        "source_candidate_receipt": source_receipt,
        "rendered_receipts": rendered[:8],
        "above_source_competitors": competitors,
        "candidate_count": len(receipts),
    }


def diagnose_context_packet_report(
    store,
    report_path: str,
    *,
    limit: int = 10,
    include_hits: bool = False,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Explain packet eval misses without mutating retrieval state."""
    report = load_packet_eval_report(report_path)
    probes = report.get("probe_results") or []
    selected = [
        probe for probe in probes
        if include_hits or not probe.get("source_hit")
    ][: max(int(limit or 10), 1)]

    access_snapshot = _snapshot_access_state(store)
    try:
        diagnoses = []
        for probe in selected:
            source_id = probe.get("source_id") or ""
            rendered_ids = list(probe.get("memory_ids_used") or [])
            rendered_set = set(rendered_ids)
            edges = _source_edges(store, source_id, rendered_set)
            nearest = _rendered_similarity(store, source_id, rendered_ids)
            diagnoses.append({
                "query": probe.get("query") or "",
                "source": _memory_summary(store, source_id),
                "source_type": probe.get("source_type") or "",
                "miss_reason": probe.get("miss_reason") or "",
                "source_hit": bool(probe.get("source_hit")),
                "source_candidate_hit": bool(probe.get("source_candidate_hit")),
                "chain_hit": bool(probe.get("chain_hit")),
                "source_edge_count": int(probe.get("source_edge_count") or 0),
                "packet_tokens": int(probe.get("packet_tokens") or 0),
                "rendered_count": len(rendered_ids),
                "rendered": [_memory_summary(store, node_id) for node_id in rendered_ids],
                "source_edges": edges,
                "nearest_rendered": nearest,
                "rank_trace": _rank_trace_for_probe(store, probe, report),
                "recommendation": _miss_recommendation(probe, edges, nearest),
            })
    finally:
        _restore_access_state(store, access_snapshot)

    result = {
        "report_path": report_path,
        "sample_size": report.get("sample_size"),
        "packet_hit_rate": report.get("packet_hit_rate"),
        "by_miss_reason": report.get("by_miss_reason") or {},
        "included_hits": include_hits,
        "diagnosed": len(diagnoses),
        "diagnoses": diagnoses,
    }
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result


def format_packet_diagnosis(diagnosis: Dict[str, Any]) -> str:
    """Format packet miss diagnosis as markdown."""
    lines = [
        "# Cairn Context Packet Diagnosis",
        "",
        f"Report: `{diagnosis.get('report_path')}`",
        f"Packet hit rate: {float(diagnosis.get('packet_hit_rate') or 0.0):.1%}",
        f"Diagnosed probes: {diagnosis.get('diagnosed', 0)}",
    ]
    if diagnosis.get("by_miss_reason"):
        lines.extend(["", "## Miss Reasons", ""])
        for reason, metrics in sorted(diagnosis["by_miss_reason"].items()):
            lines.append(f"- `{reason}`: {metrics.get('count', 0)}")

    for item in diagnosis.get("diagnoses") or []:
        source = item.get("source") or {}
        lines.extend([
            "",
            "## Probe",
            "",
            f"Query: `{item.get('query', '')[:100]}`",
            f"Source: `{source.get('id', '')}` ({item.get('source_type') or source.get('event_type') or 'memory'})",
            f"Reason: `{item.get('miss_reason') or 'hit'}`",
            f"Candidate hit: {item.get('source_candidate_hit')} · Chain hit: {item.get('chain_hit')} · Rendered: {item.get('source_hit')}",
            f"Recommendation: {item.get('recommendation')}",
            "",
            f"Source snippet: {source.get('snippet', '')}",
        ])
        rank_trace = item.get("rank_trace") or {}
        source_receipt = rank_trace.get("source_candidate_receipt")
        if source_receipt:
            lines.extend([
                "",
                (
                    "Source rank: "
                    f"#{source_receipt.get('rank')} score={source_receipt.get('score')} "
                    f"section={source_receipt.get('section')} "
                    f"admitted={source_receipt.get('admitted')} "
                    f"rendered={source_receipt.get('rendered')}"
                ),
            ])
        elif rank_trace:
            lines.extend(["", "Source rank: not present in rebuilt candidate set."])
        if rank_trace.get("above_source_competitors"):
            lines.extend(["", "Above-source competitors:"])
            for row in rank_trace["above_source_competitors"][:8]:
                lines.append(
                    f"- #{row.get('rank')} `{row.get('id')}` "
                    f"score={row.get('score')} {row.get('section')} "
                    f"rendered={row.get('rendered')}"
                )
        lines.extend(["", "Rendered packet items:"])
        for rendered in (item.get("rendered") or [])[:8]:
            lines.append(f"- `{rendered.get('id', '')}` {rendered.get('snippet', '')}")
        if item.get("source_edges"):
            lines.extend(["", "Source edges:"])
            for edge in (item.get("source_edges") or [])[:8]:
                marker = "rendered" if edge.get("neighbor_rendered") else "not rendered"
                lines.append(
                    f"- `{edge.get('neighbor_id')}` {edge.get('edge_type')} "
                    f"{edge.get('weight')} {marker}: "
                    f"{(edge.get('neighbor') or {}).get('snippet', '')}"
                )
        if item.get("nearest_rendered"):
            lines.extend(["", "Nearest rendered by embedding:"])
            for near in (item.get("nearest_rendered") or [])[:5]:
                lines.append(
                    f"- `{near.get('id')}` sim={near.get('similarity')}: "
                    f"{(near.get('memory') or {}).get('snippet', '')}"
                )

    return "\n".join(lines)


def backfill_packet_miss_report(
    store,
    report_path: str,
    *,
    similarity_threshold: float = 0.72,
    max_connections_per_source: int = 2,
    max_edges: Optional[int] = 10,
    event_types: Optional[List[str]] = None,
    dry_run: bool = True,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Backfill edges from a serialized packet eval report."""
    report = load_packet_eval_report(report_path)
    probes = report.get("probe_results") or []
    eligible = filter_packet_miss_probes(probes, event_types=event_types)
    result = backfill_packet_miss_edges(
        store,
        eligible,
        similarity_threshold=similarity_threshold,
        max_connections_per_source=max_connections_per_source,
        max_edges=max_edges,
        dry_run=dry_run,
    )
    result.update({
        "report_path": report_path,
        "total_probes": len(probes),
        "eligible_misses": len(eligible),
        "similarity_threshold": similarity_threshold,
        "max_connections_per_source": max_connections_per_source,
        "max_edges": max_edges if max_edges and max_edges > 0 else None,
        "event_types": list(event_types or DEFAULT_PACKET_MISS_EVENT_TYPES),
    })
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
    return result


def run_context_packet_maintenance_loop(
    store,
    *,
    artifact_prefix: str,
    sample_size: int = 20,
    budget_tokens: int = 800,
    mode: str = "before_edit",
    seed: int = 42,
    probe_cache_path: Optional[str] = None,
    similarity_threshold: float = 0.72,
    max_connections_per_source: int = 2,
    max_edges: Optional[int] = 10,
    event_types: Optional[List[str]] = None,
    apply: bool = False,
    re_eval: bool = False,
) -> Dict[str, Any]:
    """Run eval -> capped packet-miss backfill -> optional post-apply eval."""
    prefix = Path(artifact_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    before_path = f"{prefix}-before.json"
    manifest_path = f"{prefix}-backfill.json"
    after_path = f"{prefix}-after.json"

    db_path = str(getattr(store, "db_path", "") or getattr(store, "_db_path", "") or "")
    before_report = run_context_packet_evaluation(
        db_path=db_path or None,
        sample_size=sample_size,
        budget_tokens=budget_tokens,
        mode=mode,
        seed=seed,
        output_path=before_path,
        probe_cache_path=probe_cache_path,
    )
    manifest = backfill_packet_miss_report(
        store,
        before_path,
        similarity_threshold=similarity_threshold,
        max_connections_per_source=max_connections_per_source,
        max_edges=max_edges,
        event_types=event_types,
        dry_run=not apply,
        output_path=manifest_path,
    )
    after_report = None
    if apply and re_eval:
        after_report = run_context_packet_evaluation(
            db_path=db_path or None,
            sample_size=sample_size,
            budget_tokens=budget_tokens,
            mode=mode,
            seed=seed,
            output_path=after_path,
            probe_cache_path=probe_cache_path,
        )

    result: Dict[str, Any] = {
        "artifact_prefix": str(prefix),
        "before_report": before_path,
        "backfill_manifest": manifest_path,
        "after_report": after_path if after_report else None,
        "applied": apply,
        "re_evaluated": bool(after_report),
        "before": {
            "packet_hit_rate": before_report.packet_hit_rate,
            "precision_proxy": before_report.packet_precision_proxy,
            "chain_hit_rate": before_report.chain_hit_rate,
            "warning_rate": before_report.warning_rate,
            "rendered_warning_rate": before_report.rendered_warning_rate,
            "avg_packet_tokens": before_report.avg_packet_tokens,
            "avg_memories_used": before_report.avg_memories_used,
            "avg_warnings": before_report.avg_warnings,
            "avg_rendered_warnings": before_report.avg_rendered_warnings,
        },
        "backfill": manifest,
    }
    if after_report:
        result["after"] = {
            "packet_hit_rate": after_report.packet_hit_rate,
            "precision_proxy": after_report.packet_precision_proxy,
            "chain_hit_rate": after_report.chain_hit_rate,
            "warning_rate": after_report.warning_rate,
            "rendered_warning_rate": after_report.rendered_warning_rate,
            "avg_packet_tokens": after_report.avg_packet_tokens,
            "avg_memories_used": after_report.avg_memories_used,
            "avg_warnings": after_report.avg_warnings,
            "avg_rendered_warnings": after_report.avg_rendered_warnings,
        }
    return result


def _load_cached_probes(probe_cache_path: str) -> List[Probe]:
    with open(probe_cache_path) as f:
        cached = json.load(f)
    return [
        Probe(
            source_memory_id=row["source_memory_id"],
            source_content=row.get("source_content", ""),
            source_event_type=row["source_event_type"],
            query_text=row["query_text"],
            query_method=row.get("query_method", "keyword"),
        )
        for row in cached
    ]


def _save_probes(probes: List[Probe], probe_cache_path: str) -> None:
    Path(probe_cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(probe_cache_path, "w") as f:
        json.dump(
            [
                {
                    "source_memory_id": p.source_memory_id,
                    "source_content": p.source_content,
                    "source_event_type": p.source_event_type,
                    "query_text": p.query_text,
                    "query_method": p.query_method,
                }
                for p in probes
            ],
            f,
            indent=2,
        )


def _validate_probe_cache(store, probes: List[Probe], probe_cache_path: str) -> None:
    if not probes:
        return
    missing = sum(
        1
        for probe in probes
        if store.get_node(probe.source_memory_id, track_access=False) is None
    )
    missing_ratio = missing / len(probes)
    if missing_ratio > 0.10:
        raise RuntimeError(
            f"Probe cache at {probe_cache_path} references {missing}/{len(probes)} "
            f"memory IDs that no longer exist in this DB ({missing_ratio:.0%} missing). "
            "Delete the cache to regenerate, or evaluate against the original DB snapshot."
        )


def _source_chain_hit(packet: Dict[str, Any], source_id: str) -> bool:
    return any(
        node.get("id") == source_id
        for chain in packet.get("chains", [])
        for node in chain.get("nodes", [])
    )


def _source_edge_count(store, source_id: str) -> int:
    try:
        with store._lock:
            row = store._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE source_id = ? OR target_id = ?",
                (source_id, source_id),
            ).fetchone()
        return int(row[0] if row else 0)
    except Exception:
        return 0


def _classify_packet_miss(
    packet: Dict[str, Any],
    *,
    source_id: str,
    source_hit: bool,
    source_candidate_hit: bool,
    chain_hit: bool,
    source_edge_count: int,
    budget_ok: bool,
) -> str:
    """Classify why a source memory missed the rendered packet."""
    if source_hit:
        return ""
    for warning in packet.get("warnings") or []:
        item = warning.get("item") or {}
        if item.get("id") == source_id:
            return f"warning_filtered:{warning.get('reason', 'warning')}"
    if source_edge_count == 0:
        return "no_source_edges"
    if not source_candidate_hit:
        return "candidate_not_retrieved"
    if chain_hit:
        return "chain_not_rendered"
    if not budget_ok:
        return "over_budget"
    metrics = packet.get("metrics") or {}
    memories_used = int(metrics.get("memories_used") or len(packet.get("memories_used") or []))
    if memories_used >= 8:
        return "rank_or_render_cap_eviction"
    return "edge_not_retrieved_or_ranked"


def _snapshot_access_state(store) -> List[tuple[str, int, Optional[str]]]:
    """Capture advisory access counters so eval does not perturb retrieval."""
    try:
        with store._lock:
            return [
                (row[0], int(row[1] or 0), row[2])
                for row in store._conn.execute(
                    "SELECT node_id, access_count, last_accessed FROM memories"
                ).fetchall()
            ]
    except Exception:
        return []


def _restore_access_state(store, snapshot: List[tuple[str, int, Optional[str]]]) -> None:
    if not snapshot:
        return
    try:
        with store._lock:
            store._conn.executemany(
                "UPDATE memories SET access_count = ?, last_accessed = ? WHERE node_id = ?",
                [(access_count, last_accessed, node_id) for node_id, access_count, last_accessed in snapshot],
            )
            store._conn.commit()
    except Exception:
        pass


def run_context_packet_evaluation(
    db_path: Optional[str] = None,
    sample_size: int = 20,
    budget_tokens: int = 800,
    mode: str = "before_edit",
    seed: int = 42,
    output_path: Optional[str] = None,
    probe_cache_path: Optional[str] = None,
) -> PacketEvalReport:
    """Run context packet evaluation against stored memories."""
    from cairn.sqlite_store import SQLiteStore

    start_time = time.monotonic()
    if db_path:
        store = SQLiteStore(db_path)
    else:
        from cairn.bridge import _get_store
        store = _get_store()

    access_snapshot = _snapshot_access_state(store)
    try:
        total_memories = store.node_count()
        cache_hit = bool(probe_cache_path) and Path(probe_cache_path).exists()
        if cache_hit:
            probes = _load_cached_probes(probe_cache_path or "")
            _validate_probe_cache(store, probes, probe_cache_path or "")
        else:
            memories = sample_memories(store, sample_size=sample_size, seed=seed)
            probes = []
            for memory in memories:
                probe = generate_keyword_probe(memory["id"], memory["content"], memory["event_type"])
                if probe:
                    probes.append(probe)
            if probe_cache_path and len(probes) == len(memories) and probes:
                _save_probes(probes, probe_cache_path)

        report = PacketEvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_size=len(probes),
            budget_tokens=budget_tokens,
            mode=mode,
            total_memories=total_memories,
            seed=seed,
        )

        results: List[PacketProbeResult] = []
        for probe in probes:
            packet = build_context_packet(
                store,
                task=probe.query_text,
                files=[],
                scope=None,
                mode=mode,
                budget_tokens=budget_tokens,
                include_receipt=True,
            )
            used_ids = packet.get("memories_used") or []
            candidate_ids = packet.get("candidate_ids") or []
            metrics = packet.get("metrics") or {}
            packet_tokens = int(packet.get("estimated_tokens") or _estimate_tokens(packet.get("packet_markdown", "")))
            warnings_count = int(metrics.get("warnings_count") or len(packet.get("warnings") or []))
            rendered_warnings_count = int(metrics.get("rendered_warnings_count") or 0)
            chain_count = int(metrics.get("chain_count") or len(packet.get("chains") or []))
            source_edges = _source_edge_count(store, probe.source_memory_id)
            source_hit = probe.source_memory_id in used_ids
            source_candidate_hit = probe.source_memory_id in candidate_ids
            chain_hit = _source_chain_hit(packet, probe.source_memory_id)
            budget_ok = packet_tokens <= budget_tokens
            miss_reason = _classify_packet_miss(
                packet,
                source_id=probe.source_memory_id,
                source_hit=source_hit,
                source_candidate_hit=source_candidate_hit,
                chain_hit=chain_hit,
                source_edge_count=source_edges,
                budget_ok=budget_ok,
            )
            result = PacketProbeResult(
                query=probe.query_text,
                source_id=probe.source_memory_id,
                source_type=probe.source_event_type,
                packet_tokens=packet_tokens,
                budget_tokens=budget_tokens,
                memories_used=len(used_ids),
                warnings_count=warnings_count,
                rendered_warnings_count=rendered_warnings_count,
                chain_count=chain_count,
                source_edge_count=source_edges,
                memory_ids_used=list(used_ids),
                source_hit=source_hit,
                source_candidate_hit=source_candidate_hit,
                chain_hit=chain_hit,
                budget_ok=budget_ok,
                abstained=len(used_ids) == 0 and warnings_count == 0,
                miss_reason=miss_reason,
            )
            results.append(result)

        if results:
            report.packet_hit_rate = sum(1 for r in results if r.source_hit) / len(results)
            total_used = sum(r.memories_used for r in results)
            report.packet_precision_proxy = (
                sum(1 for r in results if r.source_hit) / total_used if total_used else 0.0
            )
            report.chain_hit_rate = sum(1 for r in results if r.chain_hit) / len(results)
            report.source_edge_coverage = sum(1 for r in results if r.source_edge_count > 0) / len(results)
            misses = [r for r in results if not r.source_hit]
            report.missed_source_edge_coverage = (
                sum(1 for r in misses if r.source_edge_count > 0) / len(misses) if misses else 0.0
            )
            report.budget_compliance = sum(1 for r in results if r.budget_ok) / len(results)
            report.warning_rate = sum(1 for r in results if r.warnings_count > 0) / len(results)
            report.rendered_warning_rate = sum(1 for r in results if r.rendered_warnings_count > 0) / len(results)
            report.abstention_rate = sum(1 for r in results if r.abstained) / len(results)
            report.avg_packet_tokens = sum(r.packet_tokens for r in results) / len(results)
            report.avg_memories_used = total_used / len(results)
            report.avg_warnings = sum(r.warnings_count for r in results) / len(results)
            report.avg_rendered_warnings = sum(r.rendered_warnings_count for r in results) / len(results)

        type_groups: Dict[str, List[PacketProbeResult]] = {}
        for result in results:
            type_groups.setdefault(result.source_type, []).append(result)
            report.probe_results.append(asdict(result))

        for event_type, group in type_groups.items():
            report.by_event_type[event_type] = {
                "count": len(group),
                "packet_hit_rate": round(sum(1 for r in group if r.source_hit) / len(group), 4),
                "budget_compliance": round(sum(1 for r in group if r.budget_ok) / len(group), 4),
                "abstention_rate": round(sum(1 for r in group if r.abstained) / len(group), 4),
            }

        miss_groups: Dict[str, List[PacketProbeResult]] = {}
        for result in results:
            if not result.source_hit:
                miss_groups.setdefault(result.miss_reason or "unknown", []).append(result)
        total_misses = sum(len(group) for group in miss_groups.values())
        for reason, group in sorted(miss_groups.items()):
            report.by_miss_reason[reason] = {
                "count": len(group),
                "share_of_misses": round(len(group) / total_misses, 4) if total_misses else 0.0,
                "avg_source_edges": round(sum(r.source_edge_count for r in group) / len(group), 2),
            }

        # Average from packet-level metric when available in serialized probe output
        # is intentionally omitted here; this report focuses on visible packet cost.
        report.packet_hit_rate = round(report.packet_hit_rate, 4)
        report.packet_precision_proxy = round(report.packet_precision_proxy, 4)
        report.chain_hit_rate = round(report.chain_hit_rate, 4)
        report.source_edge_coverage = round(report.source_edge_coverage, 4)
        report.missed_source_edge_coverage = round(report.missed_source_edge_coverage, 4)
        report.budget_compliance = round(report.budget_compliance, 4)
        report.warning_rate = round(report.warning_rate, 4)
        report.rendered_warning_rate = round(report.rendered_warning_rate, 4)
        report.abstention_rate = round(report.abstention_rate, 4)
        report.avg_packet_tokens = round(report.avg_packet_tokens, 1)
        report.avg_memories_used = round(report.avg_memories_used, 2)
        report.avg_warnings = round(report.avg_warnings, 2)
        report.avg_rendered_warnings = round(report.avg_rendered_warnings, 2)
        report.duration_seconds = round(time.monotonic() - start_time, 2)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(asdict(report), f, indent=2, default=str)

        return report
    finally:
        _restore_access_state(store, access_snapshot)


def format_packet_report(report: PacketEvalReport) -> str:
    """Format a packet eval report as markdown."""
    lines = [
        "# Cairn Context Packet Evaluation Report",
        "",
        f"**Date:** {report.timestamp[:19]}",
        f"**Mode:** {report.mode}",
        f"**Sample:** {report.sample_size} probes from {report.total_memories} memories",
        f"**Budget:** {report.budget_tokens} tokens",
        f"**Seed:** {report.seed}",
        f"**Duration:** {report.duration_seconds}s",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Packet Hit Rate | {report.packet_hit_rate:.1%} |",
        f"| Precision Proxy | {report.packet_precision_proxy:.1%} |",
        f"| Chain Hit Rate | {report.chain_hit_rate:.1%} |",
        f"| Source Edge Coverage | {report.source_edge_coverage:.1%} |",
        f"| Missed Source Edge Coverage | {report.missed_source_edge_coverage:.1%} |",
        f"| Budget Compliance | {report.budget_compliance:.1%} |",
        f"| Raw Warning Rate | {report.warning_rate:.1%} |",
        f"| Rendered Warning Rate | {report.rendered_warning_rate:.1%} |",
        f"| Abstention Rate | {report.abstention_rate:.1%} |",
        f"| Avg Packet Tokens | {report.avg_packet_tokens:.1f} |",
        f"| Avg Memories Used | {report.avg_memories_used:.2f} |",
        f"| Avg Raw Warnings | {report.avg_warnings:.2f} |",
        f"| Avg Rendered Warnings | {report.avg_rendered_warnings:.2f} |",
    ]

    if report.by_event_type:
        lines.extend([
            "",
            "## By Event Type",
            "",
            "| Type | Count | Hit Rate | Budget OK | Abstention |",
            "|------|-------|----------|-----------|------------|",
        ])
        for event_type, metrics in sorted(report.by_event_type.items()):
            lines.append(
                f"| {event_type} | {metrics['count']} | "
                f"{metrics['packet_hit_rate']:.1%} | "
                f"{metrics['budget_compliance']:.1%} | "
                f"{metrics['abstention_rate']:.1%} |"
            )

    if report.by_miss_reason:
        lines.extend([
            "",
            "## By Miss Reason",
            "",
            "| Reason | Count | Share of Misses | Avg Source Edges |",
            "|--------|------:|----------------:|-----------------:|",
        ])
        for reason, metrics in sorted(report.by_miss_reason.items()):
            lines.append(
                f"| {reason} | {metrics['count']} | "
                f"{metrics['share_of_misses']:.1%} | "
                f"{metrics['avg_source_edges']:.2f} |"
            )

    misses = [p for p in report.probe_results if not p["source_hit"]]
    if misses:
        lines.extend(["", f"## Misses ({len(misses)}/{report.sample_size} probes)", ""])
        for miss in misses[:10]:
            lines.append(f"- **Query:** `{miss['query'][:80]}`")
            lines.append(f"  Source: `{miss['source_id']}` ({miss['source_type']})")
            lines.append(f"  Reason: {miss.get('miss_reason') or 'unknown'}")
            lines.append(
                f"  Packet: {miss['memories_used']} memories, "
                f"{miss['warnings_count']} raw warnings, "
                f"{miss.get('rendered_warnings_count', 0)} rendered warnings, "
                f"{miss['source_edge_count']} source edges, "
                f"{miss['packet_tokens']} tokens"
            )

    return "\n".join(lines)
