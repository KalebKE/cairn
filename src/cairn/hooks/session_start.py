#!/usr/bin/env python3
"""Cairn SessionStart hook — Welcome briefing with recent context."""
import os
import time
import traceback
from datetime import datetime
from pathlib import Path


def _cairn_home():
    """Cairn data home, honoring CAIRN_HOME (env-inline; no cairn import)."""
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))



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


# Periodic maintenance (consolidate/compact/backup/...) moved to the
# daemon-side scheduler: src/cairn/scheduler.py (run from mcp_server's
# event loop). The SessionStart hook path is dead in the lean daemon, so
# jobs living here silently never ran.

def main():
    project = os.environ.get("PROJECT_DIR", os.getcwd())
    session_id = os.environ.get("SESSION_ID", "")

    try:
        from cairn.bridge import welcome
        result = welcome(session_id=session_id, project=project)
    except ImportError:
        print("Cairn not installed. Run: pip install cairn && cairn setup")
        return
    except Exception as e:
        _log_hook_error("session_start", e)
        print(f"Cairn welcome failed: {e}")
        return

    memory_count = result.get("memory_count", 0)

    print(f"## Welcome back! Cairn ready — {memory_count} memories")

    # First-time user "Aha" moment
    if memory_count == 0:
        print("")
        print("Cairn captures decisions, lessons, and errors automatically as you work.")
        print("Next session, it surfaces relevant context when you edit the same files.")
        print("")
        print("**Quick start:**")
        print('- Say "remember that we always use TypeScript strict mode" to store a preference')
        print("- Make a decision and Cairn captures it automatically")
        print("- Encounter an error, and Cairn stores the pattern for future recall")
        print("")
        print("After this session ends, you'll see exactly what was captured.")
    elif memory_count <= 10:
        print(f"  Cairn has {memory_count} memories from your first sessions. These will surface when you edit related files.")
        try:
            from cairn.bridge import type_stats as _ts_first
            first_stats = _ts_first()
            stat_parts = []
            for k, v in sorted(first_stats.items(), key=lambda x: x[1], reverse=True):
                if v > 0 and k != "session_summary":
                    stat_parts.append(f"{v} {k.replace('_', ' ')}")
            if stat_parts:
                print(f"  Captured so far: {', '.join(stat_parts[:4])}")
        except Exception:
            pass

    # Health pulse
    try:
        from datetime import timezone
        from cairn.bridge import _get_store, status as cairn_status
        health = cairn_status()
        health_label = "ok" if health.get("ok") else health.get("status", "unknown")

        store = _get_store()
        edge_count = store.edge_count()
        last_ts = store.get_last_capture_time()
        if last_ts:
            last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            delta = datetime.now(timezone.utc) - last_dt
            secs = delta.total_seconds()
            ago = (f"{int(secs)}s ago" if secs < 60
                   else f"{int(secs/60)}m ago" if secs < 3600
                   else f"{int(secs/3600)}h ago" if secs < 86400
                   else f"{int(secs/86400)}d ago")
        else:
            ago = "never"
        node_count = store.count()
        if node_count > 0:
            ratio = edge_count / node_count
            graph_label = "rich" if ratio >= 1.5 else ("good" if ratio >= 0.5 else "sparse")
            graph_info = f" | graph: {graph_label} ({edge_count:,} edges)"
        else:
            graph_info = ""
        print(f"Health: {health_label} | Last capture: {ago}{graph_info}")
    except Exception:
        pass

    # Clean up stale surfacing counter files (both .surfaced and .surfaced.json)
    try:
        cairn_dir = _cairn_home()
        cutoff = time.time() - 86400
        for pattern in ("session-*.surfaced", "session-*.surfaced.json"):
            for f in cairn_dir.glob(pattern):
                if f.stat().st_mtime < cutoff:
                    f.unlink()
    except Exception:
        pass

    # Cross-project lesson surfacing
    try:
        from cairn.bridge import get_cross_project_lessons
        cross_lessons = get_cross_project_lessons(
            task=None,
            exclude_project=project,
            limit=3,
        )
        cross_only = [l for l in cross_lessons if l.get("cross_project")]
        if cross_only:
            print("\n[CROSS-PROJECT] Lessons from other codebases:")
            for l in cross_only[:3]:
                content = l.get("content", "")[:120]
                source_proj = l.get("project", "unknown")
                print(f"  - [{source_proj}] {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("cross_project_lessons", e)

    # Surface top project lessons
    try:
        from cairn.bridge import get_cross_session_lessons
        project_lessons = get_cross_session_lessons(
            task=None,
            project_path=project,
            exclude_session=session_id,
            limit=3,
        )
        top_lessons = [l for l in project_lessons if (l.get("access_count", 0) or 0) > 0]
        if top_lessons:
            print("\n[LESSONS] Top lessons for this project:")
            for l in top_lessons[:3]:
                content = l.get("content", "")[:120]
                print(f"  - {content}")
    except ImportError:
        pass
    except Exception as e:
        _log_hook_error("project_lessons", e)

    # Weekly digest, type stats, preferences, recent memories available on-demand
    # via cairn_weekly_digest, cairn_type_stats, cairn_list_preferences.


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
    _log_timing("session_start", (time.monotonic() - _t0) * 1000)
