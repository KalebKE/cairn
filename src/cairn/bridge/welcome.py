"""Welcome / session bootstrap — peeled from bridge/__init__.py (Wave 3)."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json

logger = logging.getLogger("cairn.bridge")


# ---------------------------------------------------------------------------
# Public API -- Welcome / Session Bootstrap
# ---------------------------------------------------------------------------

# Welcome cache — keyed by (project or ""), stores (timestamp, result).
# Avoids 10+ DB round-trips on every session start (daemon serves many sessions).
# _welcome_cache / _WELCOME_CACHE_TTL live in bridge/__init__ (reset_memory clears
# the cache); referenced here via _bridge.* so both see the same dict.


def welcome(session_id: Optional[str] = None, project: Optional[str] = None) -> Dict[str, Any]:
    """Generate a session welcome briefing with relevant memories.

    Returns observation_prefix (grouped by type) and project_context
    for Claude to reference throughout the session.

    Uses a 30s cache to avoid 10+ DB round-trips when multiple sessions
    start in quick succession (common with HTTP daemon transport).
    """
    import time as _time_mod

    cache_key = project or ""
    now_mono = _time_mod.monotonic()

    # Fast path: return cached result if fresh
    if cache_key in _bridge._welcome_cache:
        cached_ts, cached_result = _bridge._welcome_cache[cache_key]
        if now_mono - cached_ts < _bridge._WELCOME_CACHE_TTL:
            # Update memory_count (cheap) so it's current
            try:
                db = _bridge._get_store()
                cached_result["memory_count"] = db.node_count()
            except Exception:
                pass
            return dict(cached_result)  # shallow copy

    db = _bridge._get_store()

    # Get recent high-value memories (decisions, lessons, preferences, errors)
    _HIGH_VALUE_TYPES = {"decision", "lesson_learned", "user_preference", "user_fact", "error_pattern", "constraint"}
    recent = []
    recent_activity = []  # Last few memories of any type for freshness
    try:
        candidates = db.get_recent(limit=200)
        for node in candidates:
            meta = node.metadata or {}
            # Skip superseded memories
            if meta.get("superseded"):
                continue
            event_type = meta.get("event_type", "")
            # Track recent activity (useful types only, up to 5)
            _NOISE_TYPES = {"session_respawn"}
            if len(recent_activity) < 5 and event_type not in _NOISE_TYPES:
                recent_activity.append(node)
            if event_type in _HIGH_VALUE_TYPES:
                recent.append(node)
                if len(recent) >= 30:
                    break
        # If no high-value memories found, fall back to most recent of any type
        if not recent:
            recent = candidates[:5]
    except Exception as e:
        logger.debug("Welcome recent memory filtering failed: %s", e)

    # Ensure user_preference and user_fact are always represented
    # These types have 95-98% never-accessed rates because recency-based
    # selection misses older entries. Direct type queries fix this.
    try:
        _entity_id = None
        if project:
            try:
                from cairn_platform.entity.engine import resolve_project_entity
                _entity_id = resolve_project_entity(project)
            except Exception as e:
                logger.debug("Welcome entity resolution failed: %s", e)
        recent_ids = {n.id for n in recent}
        _welcome_types = ("user_preference", "user_fact", "decision", "task_completion", "checkpoint", "session_summary", "behavioral_pattern")
        _limit_per_type = 8
        # Batch query: fetch all 7 types in one SQL call instead of 7 separate queries
        _placeholders = ",".join("?" for _ in _welcome_types)
        if _entity_id:
            _batch_rows = db._conn.execute(
                f"""SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds, event_type
                   FROM memories WHERE event_type IN ({_placeholders})
                   AND (entity_id = ? OR entity_id IS NULL)
                   ORDER BY event_type, created_at DESC""",
                (*_welcome_types, _entity_id),
            ).fetchall()
        else:
            _batch_rows = db._conn.execute(
                f"""SELECT node_id, content, metadata, created_at,
                          access_count, last_accessed, ttl_seconds, event_type
                   FROM memories WHERE event_type IN ({_placeholders})
                   ORDER BY event_type, created_at DESC""",
                _welcome_types,
            ).fetchall()
        # Group by event_type and apply per-type limit
        _type_counts: Dict[str, int] = {}
        for row in _batch_rows:
            _et = row[7]  # event_type column
            _type_counts.setdefault(_et, 0)
            if _type_counts[_et] >= _limit_per_type:
                continue
            _type_counts[_et] += 1
            node = db._row_to_result(row[:7])
            if (node.metadata or {}).get("superseded"):
                continue
            if node.id not in recent_ids:
                recent.append(node)
                recent_ids.add(node.id)
    except Exception as e:
        logger.debug("Welcome type-based enrichment failed: %s", e)

    # Sort by blended score: priority * 0.45 + recency * 0.35 + access_boost * 0.20
    # Recency uses 3-day half-life so decisions from days ago aren't pushed out by noise
    _now_ts = _time_mod.time()
    def _recency_score(n):
        if not n.created_at:
            return 0.0
        age_hours = (_now_ts - n.created_at.timestamp()) / 3600.0
        return max(0.0, 1.0 / (1.0 + age_hours / 72.0))

    import math as _math_mod
    def _access_boost(n):
        ac = n.access_count or 0
        return min(_math_mod.log2(1 + ac), 5.0)

    recent.sort(
        key=lambda n: (
            (n.metadata or {}).get("priority", 3) * 0.45
            + _recency_score(n) * 5.0 * 0.35
            + _access_boost(n) * 0.20
        ),
        reverse=True,
    )

    # Build observation_prefix — grouped by type
    observation_prefix = ""
    try:
        grouped: Dict[str, List[str]] = {}
        type_labels = {
            "constraint": "Active Constraints",
            "user_preference": "User Preferences",
            "user_fact": "User Context",
            "decision": "Active Decisions",
            "lesson_learned": "Key Lessons",
            "error_pattern": "Known Pitfalls",
            "task_completion": "Recent Completions",
            "checkpoint": "Saved Checkpoints",
            "session_summary": "Session History",
            "behavioral_pattern": "Behavioral Patterns",
            "project_status": "Project Status",
        }
        for n in recent[:25]:
            etype = (n.metadata or {}).get("event_type", "")
            label = type_labels.get(etype)
            if not label:
                continue
            text = (n.metadata or {}).get("observation") or n.content[:300]
            if label not in grouped:
                grouped[label] = []
            if len(grouped[label]) < 7:
                grouped[label].append(text)

        if grouped:
            _STABLE_LABELS = ["Project Status", "Active Constraints", "User Preferences", "User Context", "Active Decisions", "Key Lessons", "Known Pitfalls", "Behavioral Patterns"]
            _VOLATILE_LABELS = ["Recent Completions", "Saved Checkpoints", "Session History"]

            stable_sections = []
            for label in _STABLE_LABELS:
                items = grouped.get(label, [])
                if items:
                    items_sorted = sorted(items)
                    section = f"### {label}\n"
                    section += "\n".join(f"- {item}" for item in items_sorted)
                    stable_sections.append(section)

            volatile_sections = []
            for label in _VOLATILE_LABELS:
                items = grouped.get(label, [])
                if items:
                    section = f"### {label}\n"
                    section += "\n".join(f"- {item}" for item in items)
                    volatile_sections.append(section)

            if recent_activity:
                activity_items = []
                for n in recent_activity:
                    meta = n.metadata or {}
                    etype = meta.get("event_type", "unknown")
                    text = meta.get("observation") or n.content[:120]
                    ts_str = n.created_at.strftime("%b %d %H:%M") if n.created_at else ""
                    activity_items.append(f"- [{etype}] {ts_str}: {text}")
                volatile_sections.append("### Recent Activity\n" + "\n".join(activity_items))

            all_sections = stable_sections
            if volatile_sections:
                if stable_sections:
                    all_sections = stable_sections + ["<!-- cairn:cache_breakpoint -->"] + volatile_sections
                else:
                    all_sections = volatile_sections
            observation_prefix = "\n".join(all_sections)
    except Exception as e:
        logger.debug("Welcome observation_prefix failed: %s", e)

    # Build project_context — skip expensive embedding search (db.query) on fast path.
    # Use only direct type lookups and coordination queries.
    project_context = ""
    _entity_id = None
    if project:
        try:
            from cairn_platform.entity.engine import resolve_project_entity
            _entity_id = resolve_project_entity(project)
        except Exception:
            pass

        # Surface latest project_status snapshot
        try:
            status_nodes = db.get_by_type("project_status", limit=3, entity_id=_entity_id)
            project_statuses = [
                n for n in status_nodes
                if (n.metadata or {}).get("project", "") == project
            ]
            if project_statuses:
                latest = project_statuses[0]
                text = latest.content[:400]
                project_context = f"### Project Status\n- {text}\n\n"
        except Exception as e:
            logger.debug("Welcome project_status query failed: %s", e)

        # Surface active coordination decisions for this project
        try:
            from cairn_platform.orchestrator.coordination import get_manager
            cm = get_manager()
            coord_decs = cm.query_decisions(project=project, status="active", limit=5)
            if coord_decs:
                dec_items = []
                for d in coord_decs:
                    dec_text = f"[{d['domain']}] {d['decision'][:200]}"
                    dec_items.append(f"- {dec_text}")
                if dec_items:
                    project_context += "\n### Coordination Decisions\n" + "\n".join(dec_items)
        except Exception as e:
            logger.debug("Welcome coord_decisions query failed: %s", e)

    node_count = db.node_count()

    # Dedup stats summary (in-memory, cheap)
    dedup_prevented = 0
    try:
        dedup_prevented = db.stats.get("content_dedup_skips", 0) + db.stats.get("embedding_dedup_skips", 0)
    except Exception:
        pass

    result = {
        "memory_count": node_count,
        "recent_memories": [
            {
                "id": n.id[:12],
                "content": n.content[:400],
                "type": (n.metadata or {}).get("event_type", "unknown"),
            }
            for n in recent[:5]
        ],
        "observation_prefix": observation_prefix,
        "project_context": project_context,
    }

    if dedup_prevented > 0:
        result["duplicates_prevented"] = dedup_prevented

    # NOTE: access_count bumps, behavioral reinforcement, and predictive
    # prefetch removed from welcome() — they caused 60s of DB write
    # contention when the deferred startup thread holds the write lock.
    # Access counts are still updated on actual queries (cairn_query).

    # NOTE: Supplementary enrichments (weekly digest, advisor suggestions)
    # removed from welcome() hot path — they trigger query_structured which
    # needs embedding computation + SQLite vector search, blocking 100s+ when
    # the deferred startup thread holds the DB lock during integrity_check.
    # Advisor data is surfaced via the hook system's [HANDOFF] blocks instead.

    _bridge._welcome_cache[cache_key] = (now_mono, result)

    return result


def get_session_context(
    project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Gather all data needed for session start briefing.

    Returns a dict with context_items (typed high-value memories),
    memory_count, health status, and last_capture_ago.
    """
    db = _bridge._get_store()
    node_count = db.node_count()

    # Health status
    health_status = "ok"
    try:
        health = db.check_memory_health()
        health_status = health.get("status", "ok")
    except Exception as e:
        logger.warning("Health check failed: %s", e)
        health_status = "unknown"

    # Last capture time
    last_capture_ago = ""
    try:
        recent_all = db.get_recent(limit=1)
        if recent_all:
            last_capture_ago = _bridge._relative_time(recent_all[0].created_at)
    except Exception as e:
        logger.debug("Last capture time query failed: %s", e)

    # Gather typed high-value items for [CONTEXT] section
    from cairn.types import STABLE_EVENT_TYPES
    _TYPE_TAG = {
        "constraint": "RULE",
        "user_preference": "PREF",
        "decision": "DECISION",
        "lesson_learned": "LESSON",
        "error_pattern": "PITFALL",
    }
    context_items: list[Dict[str, str]] = []

    # Always-surface constraints (separate budget, not recency-dependent)
    try:
        constraint_nodes = db.get_by_type("constraint", limit=10)
        for node in constraint_nodes:
            if (node.metadata or {}).get("superseded"):
                continue
            text = (node.metadata or {}).get("observation") or node.content[:300]
            text = text.replace("\n", " ").strip()
            context_items.append({"tag": "RULE", "text": text, "stability": "stable"})
            if len(context_items) >= 3:
                break
    except Exception as e:
        logger.debug("Constraint surfacing failed: %s", e)

    # Regular high-value items (separate budget from constraints)
    # Velocity-adaptive freshness: measure memories/day, adjust window accordingly
    try:
        from datetime import datetime as _dt_ctx, timedelta as _td_ctx, timezone as _tz_ctx
        candidates = db.get_recent(limit=100)
        _now_ctx = _dt_ctx.now(_tz_ctx.utc)

        # Compute velocity: count high-value memories in last 24h
        _24h_ago_check = _now_ctx - _td_ctx(hours=24)
        _velocity = 0
        for node in candidates:
            try:
                ca = node.created_at
                if isinstance(ca, str):
                    dt = _dt_ctx.fromisoformat(ca.replace("Z", "+00:00"))
                else:
                    dt = _dt_ctx.fromtimestamp(ca, tz=_tz_ctx.utc)
                if dt >= _24h_ago_check:
                    _velocity += 1
            except (ValueError, TypeError, OSError):
                pass

        # Adaptive windows based on velocity
        if _velocity >= 15:
            # High velocity: tight 12h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=12)
            _recent_cutoff = _now_ctx - _td_ctx(hours=36)
        elif _velocity >= 5:
            # Moderate velocity: 24h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=24)
            _recent_cutoff = _now_ctx - _td_ctx(hours=72)
        else:
            # Low velocity: wider 72h fresh window
            _fresh_cutoff = _now_ctx - _td_ctx(hours=72)
            _recent_cutoff = _now_ctx - _td_ctx(hours=168)

        def _node_age_bucket(node):
            """Return 0 (fresh), 1 (recent), 2 (stale) based on velocity-adaptive windows."""
            try:
                ca = node.created_at
                if isinstance(ca, str):
                    dt = _dt_ctx.fromisoformat(ca.replace("Z", "+00:00"))
                else:
                    dt = _dt_ctx.fromtimestamp(ca, tz=_tz_ctx.utc)
                if dt >= _fresh_cutoff:
                    return 0
                elif dt >= _recent_cutoff:
                    return 1
                return 2
            except (ValueError, TypeError, OSError):
                return 2

        # Sort candidates: fresh first, then recent, then stale
        candidates_sorted = sorted(candidates, key=_node_age_bucket)

        seen_tags: Dict[str, int] = {}
        regular_count = 0
        for node in candidates_sorted:
            etype = (node.metadata or {}).get("event_type", "")
            if etype == "constraint":
                continue  # Already handled above
            tag = _TYPE_TAG.get(etype)
            if not tag:
                continue
            if seen_tags.get(tag, 0) >= 2:
                continue
            text = (node.metadata or {}).get("observation") or node.content[:300]
            text = text.replace("\n", " ").strip()
            _stability = "stable" if etype in STABLE_EVENT_TYPES else "volatile"
            context_items.append({"tag": tag, "text": text, "stability": _stability})
            seen_tags[tag] = seen_tags.get(tag, 0) + 1
            regular_count += 1
            if regular_count >= limit:
                break
    except Exception as e:
        logger.debug("get_session_context context_items failed: %s", e)

    # Type breakdown for stats line
    type_stats: Dict[str, int] = {}
    try:
        type_stats = db.get_type_stats()
    except Exception as e:
        logger.debug("get_session_context type_stats failed: %s", e)

    # Count memories added in last 7 days
    period_new_7d = 0
    try:
        cutoff_7d = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        row = db._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ?", (cutoff_7d,)
        ).fetchone()
        period_new_7d = row[0] if row else 0
    except Exception as e:
        logger.debug("get_session_context period_new_7d failed: %s", e)

    return {
        "memory_count": node_count,
        "health_status": health_status,
        "last_capture_ago": last_capture_ago or "unknown",
        "context_items": context_items,
        "type_stats": type_stats,
        "period_new_7d": period_new_7d,
    }
