#!/usr/bin/env python3
"""Cairn SessionStop hook — Generate and store session summary on exit."""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def _cairn_home():
    """Cairn data home, honoring CAIRN_HOME (env-inline; no cairn import)."""
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))


# Tool-utilization scoring depended on the removed Pro coordination audit —
# with no live source of session-wide tool calls, the scorecard was removed.


def _log_hook_error(hook_name, error):
    try:
        log_path = _cairn_home() / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        tb = traceback.format_exc()
        data = f"[{timestamp}] {hook_name}: {error}\n{tb}\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _get_activity_counts(session_id: str) -> dict:
    """Count memories by event_type for this session."""
    try:
        from cairn.bridge import _get_store
        store = _get_store()
        return store.get_session_event_counts(session_id)
    except Exception:
        return {}


def _get_surfaced_count(session_id: str) -> int:
    """Read and clean up the surfacing counter file."""
    try:
        marker = _cairn_home() / f"session-{session_id}.surfaced"
        if marker.exists():
            count = marker.stat().st_size
            marker.unlink()
            return count
    except Exception:
        pass
    return 0


def _get_surfaced_details(session_id: str) -> tuple:
    """Read unique memory IDs and file count from surfaced.json."""
    unique_ids = 0
    unique_files = 0
    try:
        json_path = _cairn_home() / f"session-{session_id}.surfaced.json"
        if json_path.exists():
            data = json.loads(json_path.read_text())
            all_ids = set()
            for ids in data.values():
                all_ids.update(ids)
            unique_ids = len(all_ids)
            unique_files = len(data)
    except Exception:
        pass
    return unique_ids, unique_files


def _print_activity_report(session_id: str):
    """Print session memory activity summary with productivity recap."""
    if not session_id:
        return
    counts = _get_activity_counts(session_id)
    surfaced = _get_surfaced_count(session_id)
    surfaced_unique_ids, surfaced_unique_files = _get_surfaced_details(session_id)
    if not counts and surfaced == 0:
        return

    captured = sum(counts.values())
    parts = [f"{captured} captured"]
    _LABELS = {
        "error_pattern": ("error", "errors"),
        "decision": ("decision", "decisions"),
        "lesson_learned": ("lesson learned", "lessons learned"),
    }
    for key, (singular, plural) in _LABELS.items():
        n = counts.get(key, 0)
        if n:
            parts.append(f"{n} {plural if n > 1 else singular}")
    if surfaced:
        parts.append(f"{surfaced} surfaced")
    print(f"\n## Session complete — {' | '.join(parts)}")

    # Unique recall stats
    if surfaced_unique_ids > 0:
        print(f"  Recalled: {surfaced_unique_ids} unique memories across {surfaced_unique_files} file{'s' if surfaced_unique_files != 1 else ''}")

    # Weekly recap
    try:
        from cairn.bridge import _get_store
        store = _get_store()
        total = store.node_count()

        from datetime import timedelta, timezone
        week_cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        row = store._conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM memories "
            "WHERE created_at >= ? AND session_id IS NOT NULL",
            (week_cutoff,),
        ).fetchone()
        weekly_sessions = row[0] if row else 0

        row2 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ?",
            (week_cutoff,),
        ).fetchone()
        weekly_memories = row2[0] if row2 else 0

        # Prior week count for growth
        prev_cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        row3 = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at >= ? AND created_at < ?",
            (prev_cutoff, week_cutoff),
        ).fetchone()
        prev_week_memories = row3[0] if row3 else 0

        recap_parts = []
        if weekly_sessions > 1:
            recap_parts.append(f"{weekly_sessions} sessions this week")
        if weekly_memories > 0:
            recap_parts.append(f"{weekly_memories} memories this week")
        recap_parts.append(f"{total} total")
        print(f"  Recap: {', '.join(recap_parts)}")

        # Week-over-week growth
        if prev_week_memories > 0 and weekly_memories > 0:
            growth_pct = ((weekly_memories - prev_week_memories) / prev_week_memories) * 100
            sign = "+" if growth_pct >= 0 else ""
            print(f"  Growth: {sign}{growth_pct:.0f}% vs last week")
    except Exception:
        pass



def _build_summary(session_id: str, project: str) -> str:
    """Build a session summary from per-type targeted queries.

    Each category is queried independently with event_type filter.
    session_summary type is excluded entirely to prevent circular refs.
    """
    try:
        from cairn.bridge import query_structured
    except ImportError:
        return "Session ended"

    decisions = query_structured(
        query_text="decisions made",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="decision",
    )
    errors = query_structured(
        query_text="errors encountered",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="error_pattern",
    )
    tasks = query_structured(
        query_text="completed tasks",
        limit=3,
        session_id=session_id,
        project=project,
        event_type="task_completion",
    )

    if not decisions and not errors and not tasks:
        return "Session ended (no captured activity)"

    parts = []
    if decisions:
        items = [m.get("content", "")[:120] for m in decisions[:3]]
        parts.append(f"Decisions ({len(decisions)}): " + "; ".join(items))
    if errors:
        items = [m.get("content", "")[:120] for m in errors[:3]]
        parts.append(f"Errors ({len(errors)}): " + "; ".join(items))
    if tasks:
        items = [m.get("content", "")[:120] for m in tasks[:3]]
        parts.append(f"Tasks ({len(tasks)}): " + "; ".join(items))

    if not parts:
        return "Session ended"

    return " | ".join(parts)[:600]


def _get_reflect_store():
    """Lazy import store for reflection. Separated for testability."""
    from cairn.bridge import _get_store
    return _get_store()


# Lazy import: may be None if cairn.reflect is not installed
try:
    from cairn.reflect import find_contradictions
except ImportError:
    find_contradictions = None


def _auto_reflect(session_id: str, project: str) -> dict:
    """Run contradiction detection automatically at session end.
    Returns summary dict with contradictions_found count."""
    try:
        store = _get_reflect_store()
        result = find_contradictions(store, topic="recent decisions", limit=10)

        contradictions = result.get("contradictions", [])
        if contradictions:
            # Store a summary for the next session to see
            summary = f"Auto-reflect found {len(contradictions)} potential contradiction(s):\n"
            for c in contradictions[:3]:
                summary += f"- '{c.get('memory_a_content', '')[:80]}' vs '{c.get('memory_b_content', '')[:80]}'\n"

            try:
                from cairn.bridge import auto_capture
                auto_capture(
                    content=summary,
                    event_type="lesson_learned",
                    session_id=session_id,
                    project=project,
                    metadata={"source": "auto_reflect", "contradiction_count": len(contradictions)},
                )
            except Exception:
                pass

        return {"contradictions_found": len(contradictions)}
    except Exception:
        return {"contradictions_found": 0}


def _auto_feedback_on_surfaced(session_id: str):
    """Auto-record 'helpful' feedback for memories surfaced during active work."""
    if not session_id:
        return
    json_path = _cairn_home() / f"session-{session_id}.surfaced.json"
    if not json_path.exists():
        return
    try:
        data = json.loads(json_path.read_text())
        # Collect all unique memory IDs across all files
        all_ids = set()
        for ids in data.values():
            all_ids.update(ids)

        if not all_ids:
            return

        from cairn.bridge import record_feedback
        count = 0
        for mid in list(all_ids)[:10]:  # Cap at 10 feedback calls
            try:
                record_feedback(mid, "helpful", "Auto: surfaced during active work")
                count += 1
            except Exception:
                pass

        # Clean up the JSON file
        json_path.unlink(missing_ok=True)
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("auto_feedback_surfaced", e)
    finally:
        # Always try to clean up
        try:
            if json_path.exists():
                json_path.unlink()
        except Exception:
            pass


def _build_project_status(session_id: str, project: str):
    """Build a project status snapshot from session activity.

    Returns structured text or None if insufficient data.
    """
    if not project:
        return None
    try:
        from cairn.bridge import query_structured
    except ImportError:
        return None

    decisions = query_structured(
        query_text="decisions made",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="decision",
    )
    tasks = query_structured(
        query_text="completed tasks",
        limit=5,
        session_id=session_id,
        project=project,
        event_type="task_completion",
    )

    if not decisions and not tasks:
        return None  # Not enough activity for a status snapshot

    parts = [f"Project: {Path(project).name}"]
    if decisions:
        items = [m.get("content", "")[:150] for m in decisions[:3]]
        parts.append("Key decisions: " + "; ".join(items))
    if tasks:
        items = [m.get("content", "")[:150] for m in tasks[:3]]
        parts.append("Completed: " + "; ".join(items))

    return " | ".join(parts)[:600]


def main():
    session_id = os.environ.get("SESSION_ID", "")
    project = os.environ.get("PROJECT_DIR", os.getcwd())

    _auto_feedback_on_surfaced(session_id)
    _print_activity_report(session_id)

    # Auto-reflect: detect contradictions (Part C — cairn_reflect has 0 agent calls)
    try:
        reflect_result = _auto_reflect(session_id, project)
        if reflect_result["contradictions_found"] > 0:
            print(f"  Auto-reflect: {reflect_result['contradictions_found']} contradiction(s) detected. Check next session start.")
    except Exception:
        pass

    if os.environ.get("CAIRN_NO_SESSION_SUMMARY", "").strip() == "1":
        return

    summary = _build_summary(session_id, project)

    try:
        from cairn.bridge import auto_capture
        auto_capture(
            content=f"Session summary: {summary}",
            event_type="session_summary",
            metadata={"source": "session_stop_hook", "project": project},
            session_id=session_id,
            project=project,
            ttl_override=3600,  # Match hook server TTL — don't accumulate
        )
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("session_stop", e)
        print(f"Cairn session_stop failed: {e}", file=sys.stderr)

    # Auto-generate project_status (will evolve existing if present)
    project_status_text = _build_project_status(session_id, project)
    if project_status_text:
        try:
            from cairn.bridge import auto_capture as _ac
            _ac(
                content=project_status_text,
                event_type="project_status",
                session_id=session_id,
                project=project,
                metadata={"source": "session_stop_auto", "project": project},
            )
        except ImportError:
            pass
        except Exception as e:
            _log_hook_error("session_stop_project_status", e)


def _log_timing(hook_name, elapsed_ms):
    try:
        log_path = _cairn_home() / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] {hook_name}: OK ({elapsed_ms:.0f}ms)\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("session_stop", (time.monotonic() - _t0) * 1000)
