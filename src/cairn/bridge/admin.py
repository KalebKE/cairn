"""Admin & profile (health, export/import, dedup, preferences, profile, cross-session
lessons) — peeled from bridge/__init__.py (Wave 5)."""
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json
from cairn.types import AutoCaptureEventType

logger = logging.getLogger("cairn.bridge")


# ---------------------------------------------------------------------------
# Public API -- Health & Status
# ---------------------------------------------------------------------------


def check_health(
    warn_mb: float = 350,
    critical_mb: float = 800,
    max_nodes: int = 10000,
) -> str:
    """Check Cairn memory health. Returns formatted markdown."""
    db = _bridge._get_store()
    health = db.check_memory_health(warn_mb=warn_mb, critical_mb=critical_mb, max_nodes=max_nodes)

    status_label = health.get("status", "unknown").upper()
    parts = [
        f"Status: {status_label} | Mem: {health.get('memory_mb', 0):.1f}MB"
        f" | DB: {health.get('db_size_mb', 0):.2f}MB"
        f" | Nodes: {health.get('node_count', 0)}",
    ]

    warnings = health.get("warnings", [])
    if warnings:
        parts.append("Warnings: " + "; ".join(warnings))

    recommendations = health.get("recommendations", [])
    if recommendations:
        parts.append("Recs: " + "; ".join(recommendations))

    return "\n".join(parts) + "\n"


def status() -> Dict[str, Any]:
    """Return a machine-readable health/status dict."""
    db = _bridge._get_store()
    try:
        health = db.check_memory_health()
        return {
            "ok": health.get("status") == "healthy",
            "status": health.get("status", "unknown"),
            "node_count": health.get("node_count", 0),
            "memory_mb": health.get("memory_mb", 0),
            "db_size_mb": health.get("db_size_mb", 0),
            "warnings": health.get("warnings", []),
            "store_path": str(_bridge.CAIRN_HOME),
            "backend": "sqlite",
            "vec_enabled": health.get("usage", {}).get("vec_enabled", False),
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def get_dedup_stats() -> Dict[str, Any]:
    """Return deduplication statistics."""
    db = _bridge._get_store()
    return {
        "content_dedup_skips": db.stats.get("dedup_canonical", 0) + db.stats.get("dedup_exact", 0),
        "memory_evolutions": db.stats.get("memory_evolutions", 0),
        "embedding_dedup_skips": db.stats.get("embedding_dedup_skips", 0),
        "node_count": db.node_count(),
    }


# ---------------------------------------------------------------------------
# Public API -- Export / Import
# ---------------------------------------------------------------------------


def export_memories(filepath: str) -> str:
    """Export all Cairn memories to a file."""
    db = _bridge._get_store()
    result = db.export_to_file(Path(filepath))

    output = "# Cairn Export Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Size:** {result.get('file_size_kb', 0):.1f} KB\n"
    output += f"**Exported:** {result.get('exported_at', 'now')}\n"

    logger.info(f"Exported Cairn memories to {filepath}")
    return output


def import_memories(filepath: str, clear_existing: bool = True) -> str:
    """Import Cairn memories from a file."""
    db = _bridge._get_store()
    result = db.import_from_file(Path(filepath), clear_existing=clear_existing)

    output = "# Cairn Import Complete\n\n"
    output += f"**File:** {result.get('filepath', filepath)}\n"
    output += f"**Nodes Imported:** {result.get('node_count', 0)}\n"
    output += f"**Sessions:** {result.get('session_count', 0)}\n"
    output += f"**Cleared Existing:** {'Yes' if clear_existing else 'No'}\n"

    logger.info(f"Imported Cairn memories from {filepath}")
    return output


# ---------------------------------------------------------------------------
# Public API -- Deduplication
# ---------------------------------------------------------------------------


def deduplicate(
    event_type: Optional[str] = "lesson_learned",
    similarity_threshold: float = 0.80,
    dry_run: bool = False,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Find and merge duplicate memories using Jaccard similarity."""
    db = _bridge._get_store()
    result: Dict[str, Any] = {
        "event_type": event_type or "all",
        "similarity_threshold": similarity_threshold,
        "dry_run": dry_run,
        "groups_found": 0,
        "duplicates_removed": 0,
        "memories_kept": 0,
        "details": [],
    }

    # Gather candidates
    if event_type:
        candidates = db.get_by_type(event_type, limit=500)
    else:
        candidates = db.get_recent(limit=500)

    if session_id:
        candidates = [n for n in candidates if (n.metadata or {}).get("session_id") == session_id]

    if len(candidates) < 2:
        result["message"] = f"Only {len(candidates)} memories found, nothing to deduplicate."
        return result

    # Build word sets
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style grouping
    merged_into: Dict[str, str] = {}
    groups: Dict[str, list] = {}

    for i, (node_i, words_i) in enumerate(node_words):
        if node_i.id in merged_into or not words_i:
            continue

        group = [node_i]
        for node_j, words_j in node_words[i + 1:]:
            if node_j.id in merged_into or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                group.append(node_j)
                merged_into[node_j.id] = node_i.id

        if len(group) > 1:
            groups[node_i.id] = group

    result["groups_found"] = len(groups)

    for _rep_id, group in groups.items():
        group.sort(key=lambda n: len(n.content), reverse=True)
        keeper = group[0]
        duplicates = group[1:]
        total_access = sum(getattr(n, "access_count", 0) or 0 for n in group)

        detail = {
            "kept": {
                "id": keeper.id[:12],
                "content_preview": keeper.content[:100],
                "access_count": total_access,
            },
            "removed": [{"id": n.id[:12], "content_preview": n.content[:80]} for n in duplicates],
            "group_size": len(group),
        }
        result["details"].append(detail)

        if not dry_run:
            db.update_node(keeper.id, access_count=total_access)
            for dup in duplicates:
                try:
                    db.delete_node(dup.id)
                    result["duplicates_removed"] += 1
                except Exception as e:
                    logger.warning(f"Failed to remove duplicate {dup.id[:12]}: {e}")
            result["memories_kept"] += 1

    if not dry_run and result["duplicates_removed"] > 0:
        logger.info(
            f"Deduplication complete: {result['groups_found']} groups, "
            f"{result['duplicates_removed']} removed, "
            f"{result['memories_kept']} kept"
        )

    return result


# ---------------------------------------------------------------------------
# Public API -- Preferences
# ---------------------------------------------------------------------------


def extract_preferences(text: str) -> Dict[str, Any]:
    """Extract user preferences from free text and store them."""
    try:
        from cairn.preferences import PreferenceExtractor

        extractor = PreferenceExtractor()
        prefs = extractor.extract(text)
        stored = []
        for pref in prefs:
            _bridge.auto_capture(
                content=f"[Preference] {pref.get('key', 'unknown')}: {pref.get('value', text[:100])}",
                event_type=AutoCaptureEventType.USER_PREFERENCE,
                metadata={"preference_key": pref.get("key"), "preference_value": pref.get("value")},
            )
            stored.append({"key": pref.get("key"), "stored": True})
        return {"success": True, "preferences": stored, "count": len(stored)}
    except ImportError:
        _bridge.auto_capture(
            content=f"[Preference] {text[:500]}",
            event_type=AutoCaptureEventType.USER_PREFERENCE,
            metadata={"source": "raw_text"},
        )
        return {"success": True, "preferences": [{"key": "raw", "stored": True}], "count": 1}
    except Exception as e:
        logger.error(f"Preference extraction failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_preferences() -> List[Dict[str, Any]]:
    """List stored user preferences."""
    db = _bridge._get_store()
    try:
        nodes = db.get_by_type(AutoCaptureEventType.USER_PREFERENCE, limit=100)
        return [
            {
                "id": n.id,
                "content": n.content,
                "created_at": n.created_at.isoformat() if n.created_at else "",
                "metadata": n.metadata or {},
            }
            for n in nodes
        ]
    except Exception as e:
        logger.error(f"list_preferences failed: {e}", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Public API -- Profile
# ---------------------------------------------------------------------------


def get_profile() -> Dict[str, Any]:
    """Get the user profile from the Cairn home directory, augmented with preference memories."""
    profile_path = _bridge.CAIRN_HOME / "profile.json"
    profile: Dict[str, Any] = {}
    try:
        if profile_path.exists():
            with open(profile_path, "r") as f:
                profile = json.loads(f.read())
    except Exception as e:
        logger.debug(f"Failed to load profile: {e}")
    # Augment with preference memories
    try:
        store = _bridge._get_store()
        prefs = store.get_by_type("user_preference", limit=20)
        if prefs:
            profile["preferences_from_memory"] = [
                {
                    "content": m.content,
                    "created": m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at),
                }
                for m in prefs
            ]
    except Exception as e:
        logger.debug(f"Failed to load preference memories: {e}")
    return profile


def save_profile(profile: Dict[str, Any]) -> bool:
    """Persist the user profile to disk (atomic write via temp+rename)."""
    profile_path = _bridge.CAIRN_HOME / "profile.json"
    try:
        profile_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        import tempfile

        fd, tmp_path = tempfile.mkstemp(dir=profile_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(json.dumps(profile, indent=2))
            os.replace(tmp_path, profile_path)
        except BaseException:
            os.unlink(tmp_path)
            raise
        return True
    except Exception as e:
        logger.error(f"Failed to save profile: {e}", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Public API -- Cross-session lessons
# ---------------------------------------------------------------------------


def get_cross_session_lessons(
    task: Optional[str] = None,
    project_path: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Retrieve top lessons from ALL past sessions for injection."""
    db = _bridge._get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    lesson_sessions: Dict[str, set] = {}

    try:
        if task and (context_file or context_tags):
            # Use full query() for contextual re-ranking when context is available
            enhanced = f"lesson_learned {task}"
            if project_path:
                enhanced = f"{Path(project_path).name} {enhanced}"
            raw = db.query(
                enhanced,
                limit=limit * 3,
                context_file=context_file or "",
                context_tags=context_tags,
                project_path=project_path or "",
            )
            nodes = [r for r in raw if (r.metadata or {}).get("event_type") == "lesson_learned"]
        elif task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 3)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 3)

        for node in nodes:
            meta = node.metadata or {}
            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()
            node_session = meta.get("session_id", "")

            if key in seen_content:
                if node_session and key in lesson_sessions:
                    lesson_sessions[key].add(node_session)
                continue

            seen_content.add(key)
            lesson_sessions[key] = {node_session} if node_session else set()

            lessons.append(
                {
                    "content": node.content,
                    "source": "cairn",
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": node_session,
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "verified_count": 0,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Lesson query failed: {e}")

    for lesson in lessons:
        key = lesson.get("_key", "")
        session_count = len(lesson_sessions.get(key, set()))
        if session_count > 1:
            lesson["verified_count"] = max(lesson.get("verified_count", 0), session_count)
        lesson["verified"] = lesson.get("verified_count", 0) > 0
        lesson.pop("_key", None)

    lessons.sort(
        key=lambda lesson: (lesson.get("verified_count", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


