"""Tests for the lean daemon hook server (omega.server.hook_server).

Covers the wire protocol (surface block, no-op dispatch, batch, empty/watchdog
probe), plus the surfacing logic's budget/relevance/debounce behavior against a
mocked store — without requiring a real daemon or embedding model.
"""
import asyncio
import json
import socket
from pathlib import Path

import pytest

from omega.server import hook_server as H


# A short socket path — macOS caps AF_UNIX paths at ~104 chars, so pytest's
# tmp_path (deep under /private/var/folders) can't be used for the socket.
_TEST_SOCK = Path("/tmp/omega_hook_server_test.sock")


def _client(sock_path, req, timeout=5.0):
    """Blocking UDS client mirroring fast_hook.py's protocol (send + SHUT_WR)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(str(sock_path))
    s.sendall(json.dumps(req).encode())
    s.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
        chunk = s.recv(8192)
        if not chunk:
            break
        buf += chunk
    s.close()
    return json.loads(buf.decode()) if buf else None


def test_file_path_from_payload():
    assert H._file_path_from_payload({"tool_input": json.dumps({"file_path": "/a/b.py"})}) == "/a/b.py"
    assert H._file_path_from_payload({"tool_input": {"file_path": "/a/b.py"}}) == "/a/b.py"
    assert H._file_path_from_payload({"tool_input": "not-json"}) == ""
    assert H._file_path_from_payload({}) == ""


def test_wire_protocol_roundtrip(monkeypatch):
    """surface_memories returns the block; unknown hooks are no-ops; a
    watchdog-style empty connection is handled without crashing the server.

    Wraps the async body in asyncio.run so the test needs no async plugin.
    """
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)
    monkeypatch.setattr(H, "_do_surface", lambda payload: "[MEMORY] mocked block")
    asyncio.run(_wire_protocol_body())


async def _wire_protocol_body():
    await H.start_hook_server()
    try:
        r = await asyncio.to_thread(_client, _TEST_SOCK, {"hook": "surface_memories", "tool_input": "{}"})
        assert r == {"output": "[MEMORY] mocked block", "exit_code": 0}

        # Any non-surface hook is a silent no-op (safe to wire by accident).
        r2 = await asyncio.to_thread(_client, _TEST_SOCK, {"hook": "coord_heartbeat"})
        assert r2 == {"output": "", "exit_code": 0}

        # Batch form.
        rb = await asyncio.to_thread(
            _client, _TEST_SOCK, {"hooks": ["surface_memories", "coord_heartbeat"], "tool_input": "{}"}
        )
        assert rb["results"][0]["output"] == "[MEMORY] mocked block"
        assert rb["results"][1]["output"] == ""

        # Empty connection (the 15s socket watchdog probe): connect + close, no data.
        def _probe():
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(str(_TEST_SOCK))
            s.close()
        await asyncio.to_thread(_probe)

        # Server still serves after the probe.
        r3 = await asyncio.to_thread(_client, _TEST_SOCK, {"hook": "surface_memories", "tool_input": "{}"})
        assert r3["output"] == "[MEMORY] mocked block"
    finally:
        await H.stop_hook_server()
    assert not _TEST_SOCK.exists()


def _base_cfg(**over):
    cfg = {
        "limit": 5, "relevance_floor": 0.30, "preview_chars": 40,
        "debounce_s": 999, "max_block_chars": 500, "project_debounce_s": 999,
    }
    cfg.update(over)
    return cfg


def _reset_debounce():
    H._last_surfaced.clear()
    H._last_project_surfaced.clear()


def test_surface_budget_relevance_and_debounce(monkeypatch):
    """_do_surface applies the relevance floor, per-block char budget, and
    per-file/per-project debounce, using a mocked query_structured."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(max_block_chars=130))

    calls = {"n": 0}

    def fake_query(**kwargs):
        calls["n"] += 1
        return [
            {"relevance": 0.95, "event_type": "decision", "content": "high relevance A" * 5, "id": "aaaaaaaa11"},
            {"relevance": 0.80, "event_type": "lesson_learned", "content": "high relevance B" * 5, "id": "bbbbbbbb22"},
            {"relevance": 0.10, "event_type": "memory", "content": "LOW relevance should be dropped", "id": "cccccccc33"},
        ]

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", fake_query)

    payload = {"tool_input": json.dumps({"file_path": "/proj/Service.kt"}), "project": "/proj"}
    block = H._do_surface(payload)

    assert block.startswith("[MEMORY] Relevant prior context for Service.kt:")
    assert "cccccccc" not in block          # relevance floor dropped the 0.10 item
    assert len(block) <= 130                 # max_block_chars budget respected
    assert calls["n"] == 2                   # pass 1 (file) + pass 2 (project)

    # Second call on the same file within both debounce windows: no re-query.
    block2 = H._do_surface(payload)
    assert block2 == ""
    assert calls["n"] == 2


def test_natural_tokens():
    assert H._natural_tokens("VehicleSpecService.kt") == "vehicle spec service"
    assert H._natural_tokens("sync_cursor_race.py") == "sync cursor race"
    assert H._natural_tokens("billfold-web") == "billfold web"


def test_pass1_query_is_semantic_not_filepath(monkeypatch):
    """Pass 1 must send natural-word queries with the FILE_EDIT profile —
    paths/CamelCase classify as keyword-sufficient and skip the vector
    channel, which made the old hook blind to knowledge memory."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())
    captured = []

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", lambda **kw: captured.append(kw) or [])

    H._do_surface({"tool_input": json.dumps({"file_path": "/repo/service/VehicleSpecService.kt"}),
                   "project": "/repo"})

    from omega.sqlite_store._types import SurfacingContext
    p1 = captured[0]
    assert "/" not in p1["query_text"], "no raw paths in the semantic query"
    assert "VehicleSpecService" not in p1["query_text"], "no CamelCase identifiers"
    assert "vehicle spec service" in p1["query_text"]
    assert p1["surfacing_context"] is SurfacingContext.FILE_EDIT
    assert p1["context_file"] == "/repo/service/VehicleSpecService.kt"


def test_pass2_project_briefing_knowledge_types_only(monkeypatch):
    """Pass 2 surfaces only durable knowledge (decisions/lessons/constraints),
    never episodic exhaust, under the PLANNING profile."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())
    captured = []

    def fake_query(**kw):
        captured.append(kw)
        if len(captured) == 1:  # pass 1: file context
            return [{"relevance": 0.7, "event_type": "memory",
                     "content": "notes about x.py refactor", "id": "f1"}]
        return [  # pass 2: project knowledge
            {"relevance": 0.9, "event_type": "decision", "content": "ship dashboards in August", "id": "d1"},
            {"relevance": 0.8, "event_type": "task_completion", "content": "bumped gradle", "id": "t1"},
        ]

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", fake_query)

    block = H._do_surface({"tool_input": json.dumps({"file_path": "/repo/x.py"}),
                           "project": "/Users/k/Projects/tracqi-web"})

    from omega.sqlite_store._types import SurfacingContext
    assert captured[1]["surfacing_context"] is SurfacingContext.PLANNING
    assert captured[1]["project"] == "/Users/k/Projects/tracqi-web"
    assert "Active project knowledge (tracqi-web):" in block
    assert "ship dashboards" in block
    # decision appears in both passes but is deduped; episodic never appears in pass 2
    p2_section = block.split("Active project knowledge")[1]
    assert "bumped gradle" not in p2_section


def test_pass2_debounced_per_project(monkeypatch):
    """Different files in the same project within the window trigger pass 2
    only once."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(debounce_s=0))
    calls = []

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", lambda **kw: calls.append(kw) or [])

    for f in ("/repo/a.py", "/repo/b.py", "/repo/c.py"):
        H._do_surface({"tool_input": json.dumps({"file_path": f}), "project": "/repo"})

    planning_calls = [c for c in calls if str(c.get("surfacing_context")) .endswith("PLANNING")]
    assert len(planning_calls) == 1, "project briefing must fire once per window"


def _write_transcript(tmp_path):
    """Minimal Claude Code transcript JSONL: mode records, str/list user
    content, assistant blocks with thinking+text+tool_use."""
    lines = [
        {"type": "mode", "mode": "normal"},
        {"type": "user", "message": {"content": "let's fix the catalog enrichment bug"}},
        {"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": "SECRET INTERNAL REASONING"},
            {"type": "text", "text": "Root cause: the sync cursor races the enrichment writer."},
            {"type": "tool_use", "name": "Bash", "input": {"command": "grep ..."}},
        ]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "content": "big tool output blob"},
            {"type": "text", "text": "ship the fix as PR 999"},
        ]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Decision: merged PR 999 behind the queue."},
        ]}},
        # padding so the tail clears _do_compact_capture's minimum-size guard
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Additional context on the enrichment work: " * 20},
        ]}},
    ]
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(l) for l in lines))
    return p


def test_transcript_tail_extraction(tmp_path):
    p = _write_transcript(tmp_path)
    text = H._transcript_tail(str(p), max_chars=5000)
    assert "catalog enrichment bug" in text
    assert "sync cursor races" in text
    assert "ship the fix as PR 999" in text
    assert "merged PR 999" in text
    assert "SECRET INTERNAL REASONING" not in text, "thinking blocks must be excluded"
    assert "big tool output blob" not in text, "tool results must be excluded"
    # cap respected
    assert len(H._transcript_tail(str(p), max_chars=50)) <= 50


def test_pre_compact_dispatch_is_nonblocking(monkeypatch):
    """pre_compact_capture must return immediately (compaction is waiting);
    the LLM digest + store happens in a background executor task."""
    import threading
    done = threading.Event()
    monkeypatch.setattr(H, "_do_compact_capture", lambda payload: done.set())

    async def body():
        resp = await H._dispatch("pre_compact_capture", {"transcript_path": "/nope"})
        assert resp == {"output": "", "exit_code": 0}
        for _ in range(50):
            if done.is_set():
                break
            await asyncio.sleep(0.02)

    asyncio.run(body())
    assert done.is_set(), "background capture must actually run"


def test_do_compact_capture_stores_digest(tmp_path, monkeypatch):
    p = _write_transcript(tmp_path)
    stored = {}

    class FakeStore:
        def store(self, content, metadata=None, **kw):
            stored.update({"content": content, "metadata": metadata, **kw})
            return "mem-fake"

    import omega.bridge as bridge
    import omega.llm as llm
    monkeypatch.setattr(bridge, "_get_store", lambda: FakeStore())
    monkeypatch.setattr(llm, "llm_complete",
                        lambda *a, **k: "Session digest: fixed enrichment race, merged PR 999.")

    H._do_compact_capture({
        "transcript_path": str(p),
        "session_id": "sess-1",
        "project": "/Users/k/Projects/Force-Server",
        "trigger": "auto",
    })
    assert "PR 999" in stored["content"]
    assert stored["metadata"]["event_type"] == "session_summary"
    assert stored["metadata"]["project"] == "/Users/k/Projects/Force-Server"
    assert "pre-compact" in stored["metadata"]["tags"]


def test_do_compact_capture_skips_on_empty_llm(tmp_path, monkeypatch):
    p = _write_transcript(tmp_path)
    import omega.bridge as bridge
    import omega.llm as llm
    called = {"store": False}

    class FakeStore:
        def store(self, *a, **k):
            called["store"] = True

    monkeypatch.setattr(bridge, "_get_store", lambda: FakeStore())
    monkeypatch.setattr(llm, "llm_complete", lambda *a, **k: "")
    H._do_compact_capture({"transcript_path": str(p), "session_id": "s"})
    assert not called["store"], "no digest -> nothing stored (fail-open)"


def test_no_dangling_header_when_budget_cuts_entries(monkeypatch):
    """A header whose entries were all trimmed by the budget must not be
    emitted on its own."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(max_block_chars=60))

    def fake_query(**kw):
        return [{"relevance": 0.9, "event_type": "decision",
                 "content": "some quite long content here", "id": "dddddddd44"}]

    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", fake_query)

    block = H._do_surface({"tool_input": json.dumps({"file_path": "/p/f.py"}), "project": "/p"})
    assert not block.rstrip().endswith(":"), f"dangling header in: {block!r}"


def test_surface_empty_when_no_file(monkeypatch):
    H._last_surfaced.clear()
    monkeypatch.setattr(H, "_cfg", lambda: dict(H._DEFAULTS))
    assert H._do_surface({"tool_input": "{}"}) == ""
    assert H._do_surface({}) == ""


def test_stop_hook_server_accepts_legacy_server_arg():
    """mcp_server.py calls stop_hook_server(hook_srv) on shutdown (both the
    stdio finally and the HTTP lifespan teardown). The clean-room rewrite took
    no args, so every graceful shutdown raised TypeError and leaked the socket.
    stop_hook_server must tolerate the legacy positional argument."""
    asyncio.run(H.stop_hook_server(object()))  # must not raise
    asyncio.run(H.stop_hook_server())          # zero-arg form still works


def test_last_surfaced_debounce_dict_is_bounded(monkeypatch):
    """The per-file debounce dict must not grow without bound in a long-lived
    daemon: one entry per distinct file path edited, never pruned."""
    H._last_surfaced.clear()
    monkeypatch.setattr(H, "_cfg", lambda: {
        "limit": 1, "relevance_floor": 0.30, "preview_chars": 40,
        "debounce_s": 999, "max_block_chars": 200,
    })
    import omega.bridge as bridge
    monkeypatch.setattr(bridge, "query_structured", lambda **kw: [])
    cap = H._MAX_DEBOUNCE_ENTRIES
    for i in range(cap + 50):
        H._do_surface({"tool_input": json.dumps({"file_path": f"/proj/file_{i}.py"})})
    assert len(H._last_surfaced) <= cap


def test_cfg_is_cached_until_config_mtime_changes(tmp_path, monkeypatch):
    """_cfg() ran a filesystem read + json parse on every surfaced edit.
    It must cache on (path, mtime) and only re-read when the file changes."""
    import os
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"surface": {"limit": 7}}))
    monkeypatch.setattr(H, "_omega_home", lambda: tmp_path)
    H._cfg_cache = None  # reset cache state

    reads = {"n": 0}
    orig_read_text = type(cfg_file).read_text

    def counting_read_text(self, *a, **kw):
        if self.name == "config.json":
            reads["n"] += 1
        return orig_read_text(self, *a, **kw)

    monkeypatch.setattr(type(cfg_file), "read_text", counting_read_text)

    assert H._cfg()["limit"] == 7
    assert H._cfg()["limit"] == 7
    assert reads["n"] == 1, "second call must hit the cache"

    cfg_file.write_text(json.dumps({"surface": {"limit": 9}}))
    os.utime(cfg_file, (cfg_file.stat().st_atime, cfg_file.stat().st_mtime + 10))
    assert H._cfg()["limit"] == 9, "mtime change must invalidate the cache"
    assert reads["n"] == 2


# ---------------------------------------------------------------------------
# Socket lifecycle: rebind after external unlink, inode-guarded stop.
#
# Reproduces the wedged-daemon incident: another session's stop_hook_server
# unlinks ~/.omega/hook.sock; the surviving process's watchdog calls
# start_hook_server(), which returns the cached _server without re-binding,
# so the daemon loops "re-creating..." forever while clients get
# FileNotFoundError.
# ---------------------------------------------------------------------------

def _noop_roundtrip():
    """Assert a live listener at H.SOCK_PATH without touching the store."""
    r = _client(H.SOCK_PATH, {"hook": "noop_lifecycle_probe"})
    assert r == {"output": "", "exit_code": 0}


def test_start_rebinds_after_external_unlink(monkeypatch):
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)
    asyncio.run(_rebind_body())


async def _rebind_body():
    H._server = None
    await H.start_hook_server()
    try:
        assert _TEST_SOCK.exists()
        _TEST_SOCK.unlink()  # simulate a sibling session's shutdown
        await H.start_hook_server()
        assert _TEST_SOCK.exists(), "start_hook_server must re-bind when the socket path is gone"
        await asyncio.to_thread(_noop_roundtrip)
    finally:
        await H.stop_hook_server()


def test_watchdog_iteration_restores_socket(monkeypatch):
    """The extracted watchdog iteration must restore a working socket."""
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)
    asyncio.run(_watchdog_body())


async def _watchdog_body():
    from omega.server import mcp_server as M
    H._server = None
    await H.start_hook_server()
    try:
        _TEST_SOCK.unlink()
        await M._socket_watchdog_once()
        assert _TEST_SOCK.exists()
        await asyncio.to_thread(_noop_roundtrip)
    finally:
        await H.stop_hook_server()


def test_stop_does_not_unlink_foreign_socket(monkeypatch):
    """If a newer daemon has taken over the socket path, a stale process's
    stop_hook_server must NOT delete the successor's socket."""
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)
    asyncio.run(_foreign_stop_body())


async def _foreign_stop_body():
    H._server = None
    await H.start_hook_server()
    _TEST_SOCK.unlink()

    # A successor daemon binds a fresh socket at the same path (new inode).
    foreign = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    foreign.bind(str(_TEST_SOCK))
    try:
        await H.stop_hook_server()
        assert _TEST_SOCK.exists(), "stop_hook_server deleted a socket it does not own"
    finally:
        foreign.close()
        if _TEST_SOCK.exists():
            _TEST_SOCK.unlink()


def test_stop_unlinks_own_socket(monkeypatch):
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)

    async def body():
        H._server = None
        await H.start_hook_server()
        assert _TEST_SOCK.exists()
        await H.stop_hook_server()
        assert not _TEST_SOCK.exists()

    asyncio.run(body())


def test_start_returns_same_server_when_healthy(monkeypatch):
    monkeypatch.setattr(H, "SOCK_PATH", _TEST_SOCK)

    async def body():
        H._server = None
        s1 = await H.start_hook_server()
        try:
            s2 = await H.start_hook_server()
            assert s1 is s2, "healthy idempotent start must not re-bind"
        finally:
            await H.stop_hook_server()

    asyncio.run(body())
