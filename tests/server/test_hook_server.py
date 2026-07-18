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
