"""LLM-assisted episodic rollup — the knowledge-first consolidation model.

Problem: episodic exhaust (task_completion, session_summary) dominates the
store (at one point 81% of all rows) and previously just TTL-expired at 90
days, taking its history with it. The upstream ``compact`` is heuristic
sentence-picking — no real abstraction.

Model: for each (project, calendar-month) window whose episodic rows have
aged past ``min_age_days``, synthesize ONE durable ``project_history`` memory
via the configured LLM (``omega.llm.llm_complete`` — routed through the local
litellm proxy in this install), link it to its sources with ``derived_from``
edges, then put the raw rows on a short grace TTL so normal TTL cleanup reaps
them. History is preserved in distilled form; noise ages out.

Safety properties:
- Idempotent: source rows are marked ``rolled_up``; a window with no unrolled
  rows is never re-processed.
- Fail-closed: if the LLM returns nothing, the window is left untouched and
  retried on a later run.
- Knowledge types (decisions, lessons, insights, …) are never rollup sources.
- ``dry_run`` synthesizes nothing-destructive: no writes, no TTL changes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from omega import json_compat as json
from omega.llm import llm_complete

logger = logging.getLogger("omega.rollup")

# Episodic types eligible for rollup. Never durable knowledge.
ROLLUP_SOURCE_TYPES = ("task_completion", "session_summary")

# Raw rows survive this long after a successful rollup (grace window for
# anything mid-flight that still references them), then TTL cleanup reaps them.
GRACE_TTL_DAYS = 14

# Windows smaller than this are not worth an LLM call; they'll either grow or
# TTL-expire naturally.
MIN_ROWS_PER_WINDOW = 3

# Cap prompt size: per-row excerpt and total budget (chars).
_ROW_EXCERPT_CHARS = 400
_PROMPT_BUDGET_CHARS = 14000

_SYSTEM_PROMPT = (
    "You are the memory consolidation layer of a personal engineering "
    "knowledge base. You receive a month of raw activity records (task "
    "completions and session summaries) for one project. Write a dense, "
    "factual project-history digest of what actually happened: major work "
    "streams, what shipped, key fixes with their root causes, and anything "
    "with lasting significance. Preserve concrete identifiers (PR numbers, "
    "branch names, component names, dates) — they are retrieval anchors. "
    "No preamble, no bullets-for-the-sake-of-bullets, no motivational tone. "
    "300 words maximum."
)


def find_pending_windows(store, min_age_days: int = 30) -> List[Dict[str, Any]]:
    """Return (project, YYYY-MM) windows with unrolled episodic rows old
    enough to consolidate."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    ph = ",".join("?" * len(ROLLUP_SOURCE_TYPES))
    rows = store._conn.execute(
        f"""SELECT COALESCE(project, ''), substr(created_at, 1, 7) AS w, COUNT(*)
            FROM memories
            WHERE event_type IN ({ph})
              AND created_at < ?
              AND json_extract(metadata, '$.rolled_up') IS NULL
            GROUP BY COALESCE(project, ''), w
            HAVING COUNT(*) >= ?
            ORDER BY w""",
        (*ROLLUP_SOURCE_TYPES, cutoff, MIN_ROWS_PER_WINDOW),
    ).fetchall()
    return [{"project": r[0], "window": r[1], "count": r[2]} for r in rows]


def _window_rows(store, project: str, window: str, cutoff: str) -> List[tuple]:
    ph = ",".join("?" * len(ROLLUP_SOURCE_TYPES))
    return store._conn.execute(
        f"""SELECT node_id, content, created_at, event_type
            FROM memories
            WHERE event_type IN ({ph})
              AND COALESCE(project, '') = ?
              AND substr(created_at, 1, 7) = ?
              AND created_at < ?
              AND json_extract(metadata, '$.rolled_up') IS NULL
            ORDER BY created_at""",
        (*ROLLUP_SOURCE_TYPES, project, window, cutoff),
    ).fetchall()


def _build_prompt(project: str, window: str, rows: List[tuple]) -> str:
    name = project.rstrip("/").rsplit("/", 1)[-1] or project or "unknown-project"
    parts = [f"Project: {name}", f"Window: {window}", f"Records: {len(rows)}", ""]
    used = sum(len(p) for p in parts)
    for _nid, content, created_at, etype in rows:
        line = f"[{created_at[:10]} {etype}] {(content or '')[:_ROW_EXCERPT_CHARS]}"
        if used + len(line) > _PROMPT_BUDGET_CHARS:
            parts.append(f"... ({len(rows)} records total; remainder truncated)")
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts)


def _archive_raw_config() -> bool:
    """Read rollup.archive_raw from ~/.omega/config.json (default False:
    grace-TTL deletion). True keeps raw rows on disk as status='archived' —
    verbatim fidelity behind the synthesis, excluded from retrieval."""
    import os
    from pathlib import Path
    try:
        cfg_path = Path(os.environ.get("OMEGA_HOME", os.path.expanduser("~/.omega"))) / "config.json"
        data = json.loads(cfg_path.read_text())
        return bool((data.get("rollup") or {}).get("archive_raw", False))
    except Exception:
        return False


def rollup_window(store, project: str, window: str,
                  min_age_days: int = 30, dry_run: bool = False,
                  archive_raw: "bool | None" = None) -> Dict[str, Any]:
    """Synthesize one (project, window) into a project_history memory."""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=min_age_days)).isoformat()
    rows = _window_rows(store, project, window, cutoff)
    if len(rows) < MIN_ROWS_PER_WINDOW:
        return {"status": "too_few_rows", "rows": len(rows)}

    prompt = _build_prompt(project, window, rows)
    # max_tokens=4000: reasoning models (gemini-2.5-*) spend budget on hidden
    # reasoning tokens BEFORE emitting text — a tight cap yields empty content
    # with finish_reason="length". The output length is governed by the
    # 300-word instruction in the system prompt, not this ceiling.
    synthesis = llm_complete(
        prompt, _SYSTEM_PROMPT,
        max_tokens=4000, temperature=0.2, timeout=90.0, model_tier="standard",
    )
    if not (synthesis or "").strip():
        logger.warning("rollup: LLM returned empty synthesis for %s %s — window left untouched",
                       project, window)
        return {"status": "llm_empty", "rows": len(rows)}

    if dry_run:
        return {"status": "dry_run", "rows": len(rows), "synthesis": synthesis}

    node_id = store.store(
        content=synthesis.strip(),
        metadata={
            "event_type": "project_history",
            "project": project,
            "rollup_window": window,
            "source_count": len(rows),
            "tags": ["history", "rollup"],
        },
        ttl_seconds=None,      # durable knowledge — never expires
        skip_inference=True,   # no dedup/contradiction pass against sources
    )

    if archive_raw is None:
        archive_raw = _archive_raw_config()

    now = datetime.now(timezone.utc)
    with store._lock:
        for src_id, _content, created_at, _etype in rows:
            store.add_edge(node_id, src_id, edge_type="derived_from")
            row = store._conn.execute(
                "SELECT metadata FROM memories WHERE node_id = ?", (src_id,)
            ).fetchone()
            meta = json.loads(row[0]) if row and row[0] else {}
            meta["rolled_up"] = 1
            meta["rolled_into"] = node_id
            if archive_raw:
                # Keep verbatim source on disk, out of the retrieval path.
                meta["archived"] = 1
                store._conn.execute(
                    "UPDATE memories SET ttl_seconds = NULL, status = 'archived', "
                    "metadata = ? WHERE node_id = ?",
                    (json.dumps(meta), src_id),
                )
            else:
                # Grace TTL: expire GRACE_TTL_DAYS from now (ttl is relative
                # to created_at, so add the row's current age).
                created_dt = store._parse_dt(created_at) or now
                ttl = int((now - created_dt).total_seconds()) + GRACE_TTL_DAYS * 86400
                store._conn.execute(
                    "UPDATE memories SET ttl_seconds = ?, metadata = ? WHERE node_id = ?",
                    (ttl, json.dumps(meta), src_id),
                )
        store._commit()
    store._invalidate_query_cache()

    logger.info("rollup: %s %s -> %s (%d sources)", project, window, node_id, len(rows))
    return {"status": "rolled", "node_id": node_id, "rows": len(rows)}


def rollup_pending(store, min_age_days: int = 30, dry_run: bool = False,
                   max_windows: int = 6,
                   archive_raw: "bool | None" = None) -> Dict[str, Any]:
    """Roll up all pending windows (bounded per run for nightly use)."""
    result: Dict[str, Any] = {"windows_rolled": 0, "rows_rolled": 0,
                              "skipped_llm": 0, "windows": []}
    for win in find_pending_windows(store, min_age_days=min_age_days)[:max_windows]:
        r = rollup_window(store, win["project"], win["window"],
                          min_age_days=min_age_days, dry_run=dry_run,
                          archive_raw=archive_raw)
        entry = {**win, **{k: v for k, v in r.items() if k != "synthesis"}}
        if dry_run and "synthesis" in r:
            entry["synthesis"] = r["synthesis"]
        result["windows"].append(entry)
        if r["status"] in ("rolled", "dry_run"):
            result["windows_rolled"] += 1
            result["rows_rolled"] += r["rows"]
        elif r["status"] == "llm_empty":
            result["skipped_llm"] += 1
    return result
