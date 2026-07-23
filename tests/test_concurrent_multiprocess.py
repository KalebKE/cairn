"""Multi-process concurrency against one shared store.

This is the real deployment shape: several parallel git worktrees each run
their own Cairn CLI/MCP *process*, and (because memories are one global store
keyed by project slug) they all write to the same ~/.cairn/cairn.db. Separate
processes each open their own SQLite connection, so correctness here rests on
WAL mode + busy-timeout, not the in-process _SQLITE_EXECUTOR serialization that
test_concurrent_store_no_sigsegv covers.

Asserts: every worker process exits cleanly (no SIGSEGV, no unhandled
"database is locked"), every distinct write lands, and the DB passes
integrity_check.
"""
import subprocess
import sys
import tempfile

import pytest

N_WORKERS = 4
WRITES_PER_WORKER = 25

_WORKER = r"""
import os, sys
os.environ["CAIRN_HOME"] = sys.argv[1]
os.environ.setdefault("CAIRN_SKIP_EMBEDDINGS", "1")
wid = int(sys.argv[2])

from cairn.sqlite_store import SQLiteStore
from cairn import bridge

store = SQLiteStore()
bridge._store = store

for i in range(25):
    # Distinct content per (worker, i) so nothing dedups away — the final
    # count must equal N_WORKERS * WRITES_PER_WORKER exactly.
    bridge.auto_capture(
        content=f"mp worker {wid} distinct write {i} nonce-{wid}-{i}-zzq",
        event_type="observation",
        session_id=f"mp-{wid}",
    )
    if i % 8 == 0:
        store.query(f"nonce-{wid}", limit=3)          # reader contention vs writers
store.close()
print(f"OK worker {wid}")
"""

_CHECK = r"""
import os, sys
os.environ["CAIRN_HOME"] = sys.argv[1]
os.environ.setdefault("CAIRN_SKIP_EMBEDDINGS", "1")
from cairn.sqlite_store import SQLiteStore
s = SQLiteStore()
n = s._conn.execute(
    "SELECT COUNT(*) FROM memories WHERE event_type='observation'").fetchone()[0]
integ = s._conn.execute("PRAGMA integrity_check").fetchone()[0]
print(f"COUNT={n} INTEGRITY={integ}")
"""


@pytest.mark.timeout(180)
def test_multiprocess_shared_store_all_writes_land_and_db_intact():
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            subprocess.Popen(
                [sys.executable, "-c", _WORKER, tmp, str(w)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            for w in range(N_WORKERS)
        ]
        results = [(*p.communicate(timeout=150), p.returncode) for p in procs]

        for out, err, rc in results:
            assert rc == 0, f"worker crashed rc={rc}\nstdout={out!r}\nstderr={err!r}"
            assert "OK worker" in out, f"worker did not finish: {out!r}\n{err!r}"

        check = subprocess.run(
            [sys.executable, "-c", _CHECK, tmp],
            capture_output=True, text=True, timeout=60,
        )
        assert check.returncode == 0, check.stderr
        assert "INTEGRITY=ok" in check.stdout, f"DB corrupt: {check.stdout!r}"
        count = int(check.stdout.split("COUNT=")[1].split()[0])
        assert count == N_WORKERS * WRITES_PER_WORKER, (
            f"expected {N_WORKERS * WRITES_PER_WORKER} writes, found {count} "
            f"(lost writes under multi-process contention)"
        )


def test_wal_retry_survives_locked_first_attempts(monkeypatch, tmp_path):
    """The journal_mode=WAL switch retry loop must actually retry on
    "database is locked" instead of crashing.

    Regression: the retry branch called time.sleep() but the module imports
    the alias _time, so the first contended WAL switch killed the whole
    worker process with NameError — surfacing in CI as a rare
    test_multiprocess_shared_store flake whenever concurrent first-openers
    actually contended.
    """
    import sqlite3

    import cairn.crypto as crypto
    from cairn.sqlite_store import SQLiteStore

    real_connect = crypto.secure_connect
    locked_remaining = {"n": 2}

    class _FlakyConn:
        def __init__(self, conn):
            object.__setattr__(self, "_conn", conn)

        def execute(self, sql, *args, **kwargs):
            if "journal_mode=WAL" in str(sql) and locked_remaining["n"] > 0:
                locked_remaining["n"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return self._conn.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def flaky_connect(db_path, **kwargs):
        return _FlakyConn(real_connect(db_path, **kwargs))

    monkeypatch.setattr(crypto, "secure_connect", flaky_connect)

    store = SQLiteStore(db_path=tmp_path / "retry.db")  # must not raise
    try:
        assert locked_remaining["n"] == 0, "locked branch never exercised"
        node_id = store.store(content="wal retry path exercised", session_id="s1")
        assert node_id
    finally:
        store.close()
