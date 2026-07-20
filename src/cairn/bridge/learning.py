"""Cross-project learning + legacy reingest for the Cairn bridge.

Surfaces lessons learned across projects and reingests legacy JSONL
memory dumps into the SQLite store. _bridge.CAIRN_HOME and the store singleton
late-bind through the package module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json
from cairn.types import TTLCategory

logger = logging.getLogger("cairn.bridge.learning")


# ---------------------------------------------------------------------------
# Public API -- Cross-project Learning
# ---------------------------------------------------------------------------


def get_cross_project_lessons(
    task: Optional[str] = None,
    exclude_project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve lessons from ALL projects (cross-project knowledge transfer).

    Unlike get_cross_session_lessons which may filter by project,
    this explicitly queries without project scope to find patterns
    that recur across different codebases.
    """
    db = _bridge._get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    project_sets: Dict[str, set] = {}

    try:
        if task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 4)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 4)

        for node in nodes:
            meta = node.metadata or {}
            node_project = meta.get("project", "")

            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if exclude_project and node_project == exclude_project:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()

            if key in seen_content:
                if node_project and key in project_sets:
                    project_sets[key].add(node_project)
                continue

            seen_content.add(key)
            project_sets[key] = {node_project} if node_project else set()

            lessons.append(
                {
                    "content": node.content,
                    "source_project": node_project,
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": meta.get("session_id", ""),
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "projects_seen": 1,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Cross-project lesson query failed: {e}")

    # Enrich with cross-project counts
    for lesson in lessons:
        key = lesson.get("_key", "")
        proj_count = len(project_sets.get(key, set()))
        lesson["projects_seen"] = max(1, proj_count)
        lesson["cross_project"] = proj_count > 1
        lesson.pop("_key", None)

    # Sort by cross-project occurrence, then access count
    lessons.sort(
        key=lambda lesson: (lesson.get("projects_seen", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Reingest (legacy JSONL → SQLite)
# ---------------------------------------------------------------------------


def reingest(
    store_path: Optional[Path] = None,
    batch_size: int = 50,
    skip_types: Optional[set] = None,
) -> Dict[str, Any]:
    """Bulk-load JSONL store entries into SQLite.

    Reads every line from store.jsonl and inserts into the SQLite database.
    Content-hash dedup prevents duplicates automatically.
    """
    db = _bridge._get_store()
    src = store_path or (_bridge.CAIRN_HOME / "store.jsonl")

    if not src.exists():
        return {"error": f"Store file not found: {src}", "ingested": 0}

    skip_types = skip_types or set()
    stats = {"ingested": 0, "skipped": 0, "duplicates": 0, "errors": 0, "total": 0}

    logger.info(f"Reingesting from {src}")

    from cairn.crypto import decrypt_line

    with open(src, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(decrypt_line(line))
            except Exception as e:
                logger.debug("Import line parse failed at line %d: %s", line_num, e)
                stats["errors"] += 1
                continue

            content = entry.get("content", "").strip()
            if not content:
                stats["skipped"] += 1
                continue

            meta = entry.get("metadata", {})
            event_type = meta.get("event_type", "memory")

            if event_type in skip_types:
                stats["skipped"] += 1
                continue

            session_id = meta.get("session_id")
            ttl = TTLCategory.for_event_type(event_type)

            try:
                db.store(
                    content=content[:2000],
                    session_id=session_id,
                    metadata=meta,
                    ttl_seconds=ttl,
                    skip_inference=True,
                )
                stats["ingested"] += 1
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Reingest error line {line_num}: {e}")

            if stats["ingested"] > 0 and stats["ingested"] % batch_size == 0:
                logger.info(f"  Progress: {stats['ingested']} ingested, {stats['total']} processed")

    logger.info(
        f"Reingest complete: {stats['ingested']} ingested, "
        f"{stats['duplicates']} duplicates, {stats['errors']} errors "
        f"out of {stats['total']} entries"
    )
    return stats
