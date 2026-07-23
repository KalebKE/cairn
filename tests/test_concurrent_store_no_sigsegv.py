"""Regression test for issue #58: store() SIGSEGVs under concurrency.

The reporter showed 8 workers × 40 stores = 320 concurrent store() calls
reliably crashed _sqlite3 _pysqlite_query_execute on macOS Python 3.12/3.14.
Root cause: bridge's auto-relate daemon thread called find_similar()/
_vec_query() on the shared sqlite connection without serialization against
the calling-thread writes; sqlite-vec's vec0 C state corrupted under
concurrent access.

Test strategy: SIGSEGV is uncatchable in-process — the test worker dies and
pytest gets an opaque worker-crash report. Run the stress in a subprocess
and assert clean exit instead.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


STRESS_SCRIPT = r"""
import os
import sys
import threading
from pathlib import Path

# Force a non-default CAIRN_HOME so the test never touches a real DB.
os.environ["CAIRN_HOME"] = sys.argv[1]
os.environ.setdefault("CAIRN_SKIP_EMBEDDINGS", "1")  # hash fallback — no ONNX dependency

from cairn import bridge
from cairn.sqlite_store import SQLiteStore

# Reuse one store across workers — that's the real-world MCP/CLI shape and
# the exact configuration the reporter crashed on.
store = SQLiteStore()
bridge._store = store  # bridge module-level singleton

N_WORKERS = 8
PER_WORKER = 40

def _worker(worker_id: int) -> None:
    for i in range(PER_WORKER):
        bridge.auto_capture(
            content=f"stress worker {worker_id} write {i} unique-{worker_id * 1000 + i}",
            event_type="observation",
            session_id=f"stress-{worker_id}",
        )

threads = [threading.Thread(target=_worker, args=(w,)) for w in range(N_WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Wait briefly for any straggling auto-relate daemon threads, then close.
# close() joins registered background threads — this is where stale cursors
# would surface as a use-after-close segfault if locking is wrong.
store.close()
print(f"OK: {N_WORKERS * PER_WORKER} concurrent stores completed cleanly")
"""


@pytest.mark.timeout(120)
def test_concurrent_store_no_sigsegv():
    """8 workers × 40 stores = 320 concurrent calls, subprocess-isolated.

    Regression coverage for issue #58. Lock-only is necessary (single-conn
    architecture); without the find_similar/_vec_query lock, this reliably
    SIGSEGVs on macOS Py3.12+ within seconds.
    """
    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, "-c", STRESS_SCRIPT, tmp],
            capture_output=True,
            text=True,
            timeout=90,
        )

    # SIGSEGV on POSIX = negative returncode -11; any nonzero is a fail.
    assert result.returncode == 0, (
        f"Concurrent store subprocess crashed (returncode={result.returncode}).\n"
        f"stdout: {result.stdout!r}\n"
        f"stderr: {result.stderr!r}"
    )
    assert "OK:" in result.stdout, f"Expected OK marker; got: {result.stdout!r}"


@pytest.mark.timeout(60)
def test_node_count_never_none_under_concurrent_writes(tmp_path):
    """Lockless counter reads returned a NULL row under thread contention.

    Regression: node_count()/edge_count()/get_last_capture_time() executed on
    the shared connection without self._lock. With concurrent locked writers
    (executor workers, auto-relate threads) the unprotected COUNT(*) could
    observe a NULL row — node_count() returned None, which broke
    bridge.maintenance.consolidate ("NoneType - int") in CI. With the reads
    unlocked this hammer reproduces None within ~20s; locked, it stays clean.
    """
    import threading

    from cairn.sqlite_store import SQLiteStore

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(0.000005)  # force aggressive thread interleaving
    bad = []
    stop = threading.Event()
    store = SQLiteStore(db_path=Path(tmp_path) / "race.db")
    emb = [1.0] + [0.0] * 383

    def writer():
        i = 0
        while not stop.is_set():
            nid = store.store(
                content=f"race doc {i} with some content",
                session_id="s",
                metadata={"event_type": "memory"},
                embedding=emb,
                skip_inference=True,
            )
            store.delete_node(nid)
            i += 1

    def reader():
        while not stop.is_set():
            n = store.node_count()
            if not isinstance(n, int):
                bad.append(n)
                stop.set()

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=writer),
        threading.Thread(target=reader),
    ]
    try:
        for t in threads:
            t.start()
        stop.wait(8)
        stop.set()
        for t in threads:
            t.join(timeout=10)
    finally:
        sys.setswitchinterval(old_interval)
        store.close()

    assert not bad, f"node_count returned non-int under concurrency: {bad!r}"
