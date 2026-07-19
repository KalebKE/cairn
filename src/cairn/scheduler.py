"""Daemon-side periodic maintenance scheduler.

Successor to the SessionStart-hook maintenance jobs, which had been dead
code since the Pro hook server was stripped: the lean daemon no-ops the
``session_start`` hook and ``fast_hook.py`` has no fallback for it, so
consolidate/compact/backup silently stopped running — and rollup/gc/link
never had any scheduler at all (they relied on an untracked nightly cron).

The MCP server's event loop calls :func:`run_due_jobs` on a fixed tick
(see ``mcp_server._maintenance_scheduler``), always inside the shared
``_SQLITE_EXECUTOR`` so DB work never blocks the loop and never races the
tool handlers. Marker files + ``flock`` make the whole thing safe across
multiple concurrent daemon processes: whichever process claims the marker
first runs the job, everyone else skips.

Semantics inherited from the old hook implementation, deliberately:

* The marker is written BEFORE the work runs, so a concurrent process
  can't start the same job. A job that *raises* rolls its marker back
  (retried next tick); a job whose process dies mid-run consumes one
  interval — acceptable for jobs that run daily/weekly.
* ``rollback_marker`` with the empty-string sentinel (first ever run)
  removes the marker entirely.
"""
from __future__ import annotations

import fcntl
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger("cairn.scheduler")


def _cairn_home() -> Path:
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))


def try_acquire_periodic(marker_name: str, min_age_days: int) -> Optional[str]:
    """Atomically check and claim a periodic task via file lock.

    Returns the old marker content (for rollback) if claimed, None if the
    job is not due or another process holds the lock. The marker is
    rewritten BEFORE the caller does the slow work, blocking concurrent
    claims.
    """
    cairn_dir = _cairn_home()
    cairn_dir.mkdir(parents=True, exist_ok=True)
    marker = cairn_dir / marker_name
    lock_path = cairn_dir / f"{marker_name}.lock"
    lock_fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError):
        os.close(lock_fd)
        return None  # Another process holds the lock

    try:
        old_content = None
        if marker.exists():
            old_content = marker.read_text().strip()
            last = datetime.fromisoformat(old_content)
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - last).days
            if age_days < min_age_days:
                return None

        marker.write_text(datetime.now(timezone.utc).isoformat())
        return old_content if old_content else ""
    except Exception:
        return None
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


def rollback_marker(marker_name: str, old_content: str) -> None:
    """Restore a marker to its previous value after a failed job."""
    marker = _cairn_home() / marker_name
    if old_content == "":
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(old_content)


# ---------------------------------------------------------------------------
# Job bodies. Module-level functions referenced BY NAME from the JOBS table
# so tests (and operators) can monkeypatch them individually.
# ---------------------------------------------------------------------------

def _job_rollup() -> None:
    """LLM-assisted episodic rollup (task_completions/session_summaries →
    durable project_history). Never with degraded embeddings: the synthesis
    memory would get an unqueryable hash vector."""
    from cairn.embedding import generate_embedding, is_embedding_degraded
    generate_embedding("rollup health probe")
    if is_embedding_degraded():
        raise RuntimeError("embedding backend degraded — rollup deferred")

    from cairn.bridge import _get_store
    from cairn.rollup import rollup_pending
    result = rollup_pending(_get_store(), min_age_days=30, max_windows=4)
    logger.info("rollup: %d window(s), %d row(s), %d skipped (no LLM)",
                result.get("windows_rolled", 0), result.get("rows_rolled", 0),
                result.get("skipped_llm", 0))


def _job_link() -> None:
    """Discover cross-memory connections over the last 48h of activity."""
    from cairn.bridge import discover_connections
    discover_connections(lookback_hours=48)


def _job_gc() -> None:
    """Storage GC: re-embed hash-tainted rows, drain the write-only
    cloud_delete_queue, checkpoint the WAL, VACUUM."""
    from cairn.bridge import _get_store
    db = _get_store()
    db.reembed_hash_tainted()
    with db._lock:
        db._conn.execute("DELETE FROM cloud_delete_queue")
        db._commit()
    with db._lock:
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db._conn.execute("VACUUM")
    logger.info("gc: reembed + queue drain + VACUUM complete")


def _job_consolidate() -> None:
    from cairn.bridge import consolidate
    consolidate(prune_days=7, max_summaries=30)


_COMPACT_TYPES = ("advisor_insight", "lesson_learned", "decision",
                  "observation", "session_summary", "handoff", "task_completion")


def _job_compact() -> None:
    from cairn.bridge import compact
    for etype in _COMPACT_TYPES:
        compact(event_type=etype, similarity_threshold=0.50, min_cluster_size=2)


def _job_backup() -> None:
    import cairn.bridge as _bridge
    backup_dir = _cairn_home() / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"cairn-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    _bridge.export_memories(filepath=str(dest))
    backups = sorted(backup_dir.glob("cairn-*.json"), key=lambda p: p.name, reverse=True)
    for b in backups[4:]:
        b.unlink()


def _job_behavioral() -> None:
    try:
        from cairn.behavioral import analyze_and_store
    except ImportError:
        return  # optional module
    analyze_and_store()


@dataclass(frozen=True)
class MaintenanceJob:
    name: str
    marker: str
    interval_days: int
    fn_name: str  # resolved via globals() at run time so patches apply
    requires_idle: bool = False


JOBS: List[MaintenanceJob] = [
    MaintenanceJob("rollup", "last-rollup", 1, "_job_rollup"),
    MaintenanceJob("link", "last-link", 1, "_job_link"),
    MaintenanceJob("gc", "last-gc", 7, "_job_gc", requires_idle=True),
    MaintenanceJob("consolidate", "last-consolidate", 3, "_job_consolidate"),
    MaintenanceJob("compact", "last-compact", 3, "_job_compact"),
    MaintenanceJob("backup", "last-backup", 7, "_job_backup"),
    MaintenanceJob("behavioral", "last-behavioral-analysis", 3, "_job_behavioral"),
]


def run_due_jobs(is_idle: Callable[[], bool] = lambda: True,
                 shutting_down: Callable[[], bool] = lambda: False) -> List[str]:
    """Run every due job. Synchronous — call from an executor thread.

    ``is_idle`` gates jobs that hold the DB for a long time (VACUUM);
    skipping for idleness does NOT consume the marker, so the job retries
    on the next tick. ``shutting_down`` is checked between jobs so server
    shutdown isn't stalled behind maintenance.
    """
    ran: List[str] = []
    for job in JOBS:
        if shutting_down():
            break
        if job.requires_idle and not is_idle():
            continue
        old = try_acquire_periodic(job.marker, job.interval_days)
        if old is None:
            continue
        try:
            globals()[job.fn_name]()
            ran.append(job.name)
        except Exception:
            logger.exception("maintenance job %s failed", job.name)
            rollback_marker(job.marker, old)
    return ran
