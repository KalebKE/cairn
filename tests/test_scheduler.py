"""Tests for omega.scheduler — the daemon-side periodic maintenance runner.

Replaces the dead SessionStart-hook jobs (which never executed: the lean
daemon no-ops session_start and fast_hook has no fallback for it) with an
in-process scheduler driven from the MCP server's event loop.
"""
import fcntl
import json
import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def sched(tmp_omega_dir):
    """Import the scheduler with OMEGA_HOME pointed at a tmp dir."""
    import omega.scheduler as s
    return s


def _seed_marker(tmp_omega_dir, name, age_days):
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    (tmp_omega_dir / name).write_text(ts.isoformat())
    return ts.isoformat()


# ---------------------------------------------------------------------------
# Marker primitives
# ---------------------------------------------------------------------------

def test_acquire_first_time_returns_empty_and_writes_marker(sched, tmp_omega_dir):
    old = sched.try_acquire_periodic("last-testjob", 3)
    assert old == ""
    assert (tmp_omega_dir / "last-testjob").exists()


def test_acquire_too_recent_returns_none(sched, tmp_omega_dir):
    _seed_marker(tmp_omega_dir, "last-testjob", age_days=1)
    assert sched.try_acquire_periodic("last-testjob", 3) is None


def test_acquire_due_returns_old_content_and_rewrites_marker(sched, tmp_omega_dir):
    old_ts = _seed_marker(tmp_omega_dir, "last-testjob", age_days=5)
    got = sched.try_acquire_periodic("last-testjob", 3)
    assert got == old_ts
    new_content = (tmp_omega_dir / "last-testjob").read_text()
    assert new_content != old_ts, "marker must be rewritten before the work runs"


def test_acquire_corrupt_marker_returns_none(sched, tmp_omega_dir):
    (tmp_omega_dir / "last-testjob").write_text("not-a-timestamp")
    assert sched.try_acquire_periodic("last-testjob", 3) is None


def test_acquire_contended_lock_returns_none(sched, tmp_omega_dir):
    lock_path = tmp_omega_dir / "last-testjob.lock"
    fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        assert sched.try_acquire_periodic("last-testjob", 3) is None
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def test_rollback_restores_old_content(sched, tmp_omega_dir):
    old_ts = _seed_marker(tmp_omega_dir, "last-testjob", age_days=5)
    sched.try_acquire_periodic("last-testjob", 3)
    sched.rollback_marker("last-testjob", old_ts)
    assert (tmp_omega_dir / "last-testjob").read_text() == old_ts


def test_rollback_removes_marker_when_first_run(sched, tmp_omega_dir):
    sched.try_acquire_periodic("last-testjob", 3)
    sched.rollback_marker("last-testjob", "")
    assert not (tmp_omega_dir / "last-testjob").exists()


# ---------------------------------------------------------------------------
# Job selection / execution
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_jobs(sched, monkeypatch):
    """Replace every job body with a recorder; return the record list."""
    ran = []
    for job in sched.JOBS:
        monkeypatch.setattr(sched, job.fn_name,
                            (lambda n: (lambda: ran.append(n)))(job.name))
    return ran


def test_run_due_jobs_runs_only_due(sched, stub_jobs, tmp_omega_dir):
    _seed_marker(tmp_omega_dir, "last-consolidate", age_days=1)  # 3d interval: not due
    ran = sched.run_due_jobs()
    assert "consolidate" not in ran
    assert "link" in ran  # no marker: due
    assert ran == stub_jobs


def test_job_exception_rolls_back_marker_and_continues(sched, stub_jobs, tmp_omega_dir, monkeypatch):
    old_ts = _seed_marker(tmp_omega_dir, "last-link", age_days=5)

    def boom():
        raise RuntimeError("job blew up")

    monkeypatch.setattr(sched, "_job_link", boom)
    ran = sched.run_due_jobs()
    assert "link" not in ran
    assert (tmp_omega_dir / "last-link").read_text() == old_ts, "failed job must roll back its marker"
    assert "consolidate" in ran, "later jobs must still run after one fails"


def test_gc_requires_idle_and_marker_not_consumed(sched, stub_jobs, tmp_omega_dir):
    ran = sched.run_due_jobs(is_idle=lambda: False)
    assert "gc" not in ran
    assert not (tmp_omega_dir / "last-gc").exists(), "skipped-for-idle must not consume the marker"
    ran2 = sched.run_due_jobs(is_idle=lambda: True)
    assert "gc" in ran2


def test_run_due_jobs_noop_when_shutting_down(sched, stub_jobs, tmp_omega_dir):
    assert sched.run_due_jobs(shutting_down=lambda: True) == []
    assert stub_jobs == []


def test_rollup_skipped_when_embedding_degraded(sched, tmp_omega_dir, monkeypatch):
    import omega.embedding as emb
    monkeypatch.setattr(emb, "generate_embedding", lambda *a, **kw: [0.0])
    monkeypatch.setattr(emb, "is_embedding_degraded", lambda: True)

    import omega.rollup as rollup_mod
    monkeypatch.setattr(
        rollup_mod, "rollup_pending",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not roll up degraded")),
    )
    with pytest.raises(RuntimeError):
        sched._job_rollup()

    # Through run_due_jobs the failure rolls the marker back.
    for job in sched.JOBS:
        if job.name != "rollup":
            monkeypatch.setattr(sched, job.fn_name, lambda: None)
    ran = sched.run_due_jobs()
    assert "rollup" not in ran
    assert not (tmp_omega_dir / "last-rollup").exists()


def test_backup_prunes_to_four(sched, tmp_omega_dir, monkeypatch):
    backup_dir = tmp_omega_dir / "backups"
    backup_dir.mkdir()
    for d in range(6):
        (backup_dir / f"omega-2026-06-{d + 1:02d}.json").write_text("{}")

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "export_memories",
                        lambda filepath: open(filepath, "w").write("{}"))
    sched._job_backup()
    remaining = sorted(p.name for p in backup_dir.glob("omega-*.json"))
    assert len(remaining) == 4
    assert remaining[-1].startswith("omega-2026")  # newest kept


def test_compact_iterates_all_seven_types(sched, monkeypatch):
    import omega.bridge as bridge
    calls = []
    monkeypatch.setattr(bridge, "compact",
                        lambda **kw: calls.append(kw))
    sched._job_compact()
    assert len(calls) == 7
    assert all(c["similarity_threshold"] == 0.50 and c["min_cluster_size"] == 2 for c in calls)
    assert {c["event_type"] for c in calls} == {
        "advisor_insight", "lesson_learned", "decision", "observation",
        "session_summary", "handoff", "task_completion",
    }


# ---------------------------------------------------------------------------
# Event-loop wiring (mcp_server._maintenance_once)
# ---------------------------------------------------------------------------

def test_maintenance_once_submits_to_executor(monkeypatch, tmp_omega_dir):
    import asyncio

    import omega.scheduler as sched
    from omega.server import mcp_server as M

    called = {}
    monkeypatch.setattr(sched, "run_due_jobs",
                        lambda is_idle, shutting_down: called.setdefault("ran", True) and ["x"])
    asyncio.run(M._maintenance_once())
    assert called.get("ran") is True


def test_maintenance_once_swallows_exceptions(monkeypatch, tmp_omega_dir):
    import asyncio

    import omega.scheduler as sched
    from omega.server import mcp_server as M

    def boom(is_idle, shutting_down):
        raise RuntimeError("maintenance exploded")

    monkeypatch.setattr(sched, "run_due_jobs", boom)
    asyncio.run(M._maintenance_once())  # must not raise
