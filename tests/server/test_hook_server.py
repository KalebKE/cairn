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


def test_surface_budget_relevance_and_debounce(monkeypatch):
    """_do_surface applies the relevance floor, per-block char budget, and
    per-file debounce, using a mocked query_structured (no real store)."""
    H._last_surfaced.clear()

    monkeypatch.setattr(H, "_cfg", lambda: {
        "limit": 5, "relevance_floor": 0.30, "preview_chars": 40,
        "debounce_s": 999, "max_block_chars": 130,
    })

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
    assert calls["n"] == 1

    # Second call on the same file within the debounce window: no re-query, empty.
    block2 = H._do_surface(payload)
    assert block2 == ""
    assert calls["n"] == 1                    # query_structured NOT called again


def test_surface_empty_when_no_file(monkeypatch):
    H._last_surfaced.clear()
    monkeypatch.setattr(H, "_cfg", lambda: dict(H._DEFAULTS))
    assert H._do_surface({"tool_input": "{}"}) == ""
    assert H._do_surface({}) == ""
