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
