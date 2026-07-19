"""Tests for the lean daemon hook server (cairn.server.hook_server).

Covers the wire protocol (surface block, no-op dispatch, batch, empty/watchdog
probe), plus the surfacing logic's budget/relevance/debounce behavior against a
mocked store — without requiring a real daemon or embedding model.
"""
import asyncio
import json
import socket
from pathlib import Path

import pytest

from cairn.server import hook_server as H


# A short socket path — macOS caps AF_UNIX paths at ~104 chars, so pytest's
# tmp_path (deep under /private/var/folders) can't be used for the socket.
_TEST_SOCK = Path("/tmp/cairn_hook_server_test.sock")


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


def _fake_packet(markdown="", used=None):
    return {"packet_markdown": markdown, "memories_used": used or [], "metrics": {}}


def _patch_builder(monkeypatch, fn):
    from cairn.server import context_handlers as ctx
    monkeypatch.setattr(ctx, "build_context_packet", fn)
    import cairn.bridge as bridge
    monkeypatch.setattr(bridge, "_get_store", lambda: object())


def test_surface_budget_floor_and_debounce(monkeypatch):
    """_do_surface renders via build_context_packet, forwards the relevance
    floor, enforces the char-budget backstop, and debounces per file/project."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(max_block_chars=130))

    calls = []

    def fake_builder(db, **kw):
        calls.append(kw)
        md = "[MEMORY_CONTEXT] f.py\nPrior Decisions:\n- `aaaaaaaaaaaa` high relevance A\n- `bbbbbbbbbbbb` high relevance B"
        return _fake_packet(md, ["aaaaaaaaaaaa-full", "bbbbbbbbbbbb-full"])

    _patch_builder(monkeypatch, fake_builder)

    payload = {"tool_input": json.dumps({"file_path": "/proj/Service.kt"}), "project": "/proj"}
    block = H._do_surface(payload)

    assert block.startswith("[MEMORY_CONTEXT]")
    assert len(block) <= 130                 # max_block_chars backstop respected
    assert len(calls) == 2                   # pass 1 (file) + pass 2 (project)
    assert calls[0]["min_seed_relevance"] == pytest.approx(0.30)

    # Second call on the same file within both debounce windows: no re-query.
    block2 = H._do_surface(payload)
    assert block2 == ""
    assert len(calls) == 2


def test_natural_tokens():
    assert H._natural_tokens("VehicleSpecService.kt") == "vehicle spec service"
    assert H._natural_tokens("sync_cursor_race.py") == "sync cursor race"
    assert H._natural_tokens("billfold-web") == "billfold web"


def test_pass1_task_is_semantic_not_filepath(monkeypatch):
    """Pass 1 must hand the builder a natural-word task — paths/CamelCase
    classify as keyword-sufficient and skip the vector channel, which made
    the old hook blind to knowledge memory. (The builder itself also
    naturalizes `files` — see test_context_packet.py.)"""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())
    captured = []
    _patch_builder(monkeypatch,
                   lambda db, **kw: captured.append(kw) or _fake_packet())

    H._do_surface({"tool_input": json.dumps({"file_path": "/repo/service/VehicleSpecService.kt"}),
                   "project": "/repo"})

    p1 = captured[0]
    assert "/" not in p1["task"], "no raw paths in the semantic task"
    assert "VehicleSpecService" not in p1["task"], "no CamelCase identifiers"
    assert "vehicle spec service" in p1["task"]
    assert p1["mode"] == "before_edit"
    assert p1["files"] == ["/repo/service/VehicleSpecService.kt"]
    assert p1["include_receipt"] is False


def test_pass2_project_briefing_knowledge_types_and_dedup(monkeypatch):
    """Pass 2 runs in planning mode restricted to durable knowledge types
    and excludes everything pass 1 already surfaced."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())
    captured = []

    def fake_builder(db, **kw):
        captured.append(kw)
        if len(captured) == 1:  # pass 1: file context
            return _fake_packet("[MEMORY_CONTEXT] x.py\n- `f1` notes about x.py refactor", ["f1-full-id"])
        return _fake_packet("[MEMORY_CONTEXT] tracqi-web\nPrior Decisions:\n- `d1` ship dashboards in August", ["d1-full-id"])

    _patch_builder(monkeypatch, fake_builder)

    block = H._do_surface({"tool_input": json.dumps({"file_path": "/repo/x.py"}),
                           "project": "/Users/k/Projects/tracqi-web"})

    p2 = captured[1]
    assert p2["mode"] == "planning"
    assert p2["event_types"] is H._KNOWLEDGE_SURFACE_TYPES
    assert p2["exclude_ids"] == {"f1-full-id"}
    assert "tracqi web" in p2["task"]  # _natural_tokens splits the hyphen
    assert p2["scope"]["project"] == "/Users/k/Projects/tracqi-web"
    assert "ship dashboards" in block


def test_pass2_debounced_per_project(monkeypatch):
    """Different files in the same project within the window trigger pass 2
    only once."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(debounce_s=0))
    calls = []
    _patch_builder(monkeypatch, lambda db, **kw: calls.append(kw) or _fake_packet())

    for f in ("/repo/a.py", "/repo/b.py", "/repo/c.py"):
        H._do_surface({"tool_input": json.dumps({"file_path": f}), "project": "/repo"})

    planning_calls = [c for c in calls if c.get("mode") == "planning"]
    assert len(planning_calls) == 1, "project briefing must fire once per window"


def test_surface_fail_open_when_builder_raises(monkeypatch):
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())

    def boom(db, **kw):
        raise RuntimeError("packet build exploded")

    _patch_builder(monkeypatch, boom)
    block = H._do_surface({"tool_input": json.dumps({"file_path": "/p/f.py"}), "project": "/p"})
    assert block == ""


def test_surface_empty_packet_returns_empty_string(monkeypatch):
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg())
    _patch_builder(monkeypatch, lambda db, **kw: _fake_packet("ignored markdown", used=[]))
    block = H._do_surface({"tool_input": json.dumps({"file_path": "/p/g.py"}), "project": "/p"})
    assert block == ""


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

    import cairn.bridge as bridge
    import cairn.llm as llm
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
    import cairn.bridge as bridge
    import cairn.llm as llm
    called = {"store": False}

    class FakeStore:
        def store(self, *a, **k):
            called["store"] = True

    monkeypatch.setattr(bridge, "_get_store", lambda: FakeStore())
    monkeypatch.setattr(llm, "llm_complete", lambda *a, **k: "")
    H._do_compact_capture({"transcript_path": str(p), "session_id": "s"})
    assert not called["store"], "no digest -> nothing stored (fail-open)"


def test_no_dangling_header_when_budget_cuts_entries(monkeypatch):
    """A header/section line whose entries were all trimmed by the char
    budget must not be emitted on its own."""
    _reset_debounce()
    monkeypatch.setattr(H, "_cfg", lambda: _base_cfg(max_block_chars=60))

    md = ("[MEMORY_CONTEXT] f.py\nPrior Decisions:\n"
          "- `dddddddddddd` some quite long decision content that will not fit the budget")
    _patch_builder(monkeypatch, lambda db, **kw: _fake_packet(md, ["dddddddddddd-full"]))

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
    _patch_builder(monkeypatch, lambda db, **kw: _fake_packet())
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
    monkeypatch.setattr(H, "_cairn_home", lambda: tmp_path)
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
# unlinks ~/.cairn/hook.sock; the surviving process's watchdog calls
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
    from cairn.server import mcp_server as M
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


# ---------------------------------------------------------------------------
# session_end: fire-and-forget worker that distills the session trajectory
# into a reusable skill_template and stores a final session digest.
# ---------------------------------------------------------------------------

def test_dispatch_session_end_schedules_worker(monkeypatch):
    called = {}
    monkeypatch.setattr(H, "_do_session_end", lambda payload: called.update(payload))

    async def body():
        resp = await H._dispatch("session_end", {"session_id": "s1"})
        assert resp == {"output": "", "exit_code": 0}
        if H._bg_futures:
            await asyncio.gather(*list(H._bg_futures))

    asyncio.run(body())
    assert called.get("session_id") == "s1", "session_end must schedule _do_session_end"


def test_session_end_worker_calls_distill(monkeypatch):
    import cairn.bridge as bridge
    seen = {}
    monkeypatch.setattr(bridge, "distill_trajectory", lambda sid: seen.setdefault("sid", sid))
    monkeypatch.setattr(H, "_do_compact_capture", lambda payload, source="pre-compact": None)
    H._do_session_end({"session_id": "abc", "transcript_path": "/nonexistent"})
    assert seen["sid"] == "abc"


def test_session_end_missing_session_id_fail_open(monkeypatch):
    import cairn.bridge as bridge
    monkeypatch.setattr(
        bridge, "distill_trajectory",
        lambda sid: (_ for _ in ()).throw(AssertionError("must not distill without session_id")),
    )
    monkeypatch.setattr(H, "_do_compact_capture", lambda payload, source="pre-compact": None)
    H._do_session_end({"transcript_path": ""})  # must not raise


def test_session_end_stores_digest_with_session_end_tag(tmp_path, monkeypatch):
    import cairn.bridge as bridge
    import cairn.llm as llm

    transcript = tmp_path / "t.jsonl"
    lines = []
    for i in range(30):
        lines.append(json.dumps({
            "type": "user",
            "message": {"content": f"prompt {i}: please fix the flaky rollup test in cairn"},
        }))
        lines.append(json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": f"reply {i}: adjusted the marker rollback logic"}]},
        }))
    transcript.write_text("\n".join(lines))

    stored = {}

    class _FakeStore:
        def store(self, content, metadata=None, session_id=None, **kw):
            stored.update({"content": content, "metadata": metadata, "session_id": session_id})
            return "node-x"

    monkeypatch.setattr(llm, "llm_complete", lambda *a, **kw: "digest text")
    monkeypatch.setattr(bridge, "_get_store", lambda: _FakeStore())
    monkeypatch.setattr(bridge, "distill_trajectory", lambda sid: None)

    H._do_session_end({
        "session_id": "s9", "transcript_path": str(transcript), "project": "/p",
    })
    assert stored["content"] == "digest text"
    assert "session-end" in stored["metadata"]["tags"]
    assert stored["session_id"] == "s9"


def test_dispatch_unknown_hook_still_noop():
    async def body():
        return await H._dispatch("some_future_hook", {})
    assert asyncio.run(body()) == {"output": "", "exit_code": 0}


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
