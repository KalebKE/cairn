"""Lean OMEGA hook server — fast in-daemon memory surfacing over a Unix socket.

This is a clean-room reimplementation of the daemon-side hook path (the previous
implementation was closed "Pro" code and is absent from the open-core
distribution).  It is deliberately scoped to a single job: answer
``surface_memories`` requests in milliseconds using the already-warm store and
the already-loaded ONNX model, and return a token-budgeted ``[MEMORY]`` block
for the client (``omega/hooks/fast_hook.py``) to print into the model's context.

Design goals (driven by the historical failure mode — see ~/.omega/LESSONS.md):

* **Never block the session.**  ``surface_memories`` is informational; on *any*
  error we return an empty output and exit code 0.  If this socket is down the
  client skips the hook entirely (no slow Python/ONNX fallback stampede).
* **Never block the event loop.**  The DB/embedding query is CPU-bound and is
  submitted to the daemon's shared ``_SQLITE_EXECUTOR`` — the same serialization
  the MCP handlers use — which also avoids the macOS sqlite-vec SIGSEGV race.
* **Token-lean by construction.**  Budgets (result count, relevance floor,
  preview length, total block size, per-file debounce) are config-driven with
  conservative defaults and read from ``~/.omega/config.json`` under ``surface``.
* **Quiet.**  Empty/watchdog probe connections are handled without log noise
  (the old server logged every 15s watchdog ping as ``unknown``, growing
  hooks.log unbounded).

Only ``surface_memories`` is handled; every other hook name returns an empty
no-op response so wiring an unexpected hook can never do harm.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("omega.hook_server")


def _omega_home() -> Path:
    return Path(os.environ.get("OMEGA_HOME", os.path.expanduser("~/.omega")))


# Must resolve to the same path the client uses (fast_hook.py: ~/.omega/hook.sock).
SOCK_PATH: Path = _omega_home() / "hook.sock"

# Conservative defaults; override any subset via ~/.omega/config.json:
#   {"surface": {"limit": 3, "relevance_floor": 0.30, ...}}
_DEFAULTS: Dict[str, Any] = {
    "limit": 3,              # max memories surfaced per edit
    "relevance_floor": 0.30,  # drop anything below this score
    "preview_chars": 100,    # per-memory preview truncation
    "debounce_s": 20.0,      # same file surfaced at most once per window
    "max_block_chars": 600,  # hard cap on the whole injected block
}

# Per-file debounce state (process-local; fine for a single long-lived daemon).
# Bounded: pruned opportunistically on insert so a long-lived daemon that has
# touched tens of thousands of distinct files doesn't grow this forever.
_last_surfaced: Dict[str, float] = {}
_MAX_DEBOUNCE_ENTRIES = 512

# The asyncio server handle (set by start_hook_server, cleared by stop).
_server: "asyncio.AbstractServer | None" = None

# _cfg cache: (config_path, mtime, resolved_dict). The surface hook fires on
# every edit; re-reading + re-parsing config.json each time is wasted I/O.
_cfg_cache: "tuple[Path, float, Dict[str, Any]] | None" = None


def _cfg() -> Dict[str, Any]:
    """Read the ``surface`` config block, falling back to defaults per-key.

    Cached on (path, mtime); the file is only re-read when it changes.
    """
    global _cfg_cache
    path = _omega_home() / "config.json"
    try:
        mtime = path.stat().st_mtime
        if _cfg_cache is not None and _cfg_cache[0] == path and _cfg_cache[1] == mtime:
            return dict(_cfg_cache[2])
        data = json.loads(path.read_text())
        s = data.get("surface", {}) or {}
        resolved = {k: s.get(k, v) for k, v in _DEFAULTS.items()}
        _cfg_cache = (path, mtime, resolved)
        return dict(resolved)
    except Exception:
        return dict(_DEFAULTS)


def _prune_debounce(now: float, debounce_s: float) -> None:
    """Keep the debounce dict bounded: drop expired entries; if still over the
    cap (many files inside one window), drop the oldest."""
    if len(_last_surfaced) < _MAX_DEBOUNCE_ENTRIES:
        return
    expired = [k for k, t in _last_surfaced.items() if now - t >= debounce_s]
    for k in expired:
        del _last_surfaced[k]
    while len(_last_surfaced) >= _MAX_DEBOUNCE_ENTRIES:
        oldest = min(_last_surfaced, key=_last_surfaced.get)
        del _last_surfaced[oldest]


def _file_path_from_payload(payload: Dict[str, Any]) -> str:
    """Extract the edited file path from a Claude Code tool payload."""
    ti = payload.get("tool_input")
    if isinstance(ti, str):
        try:
            ti = json.loads(ti)
        except Exception:
            return ""
    if isinstance(ti, dict):
        return ti.get("file_path") or ti.get("path") or ""
    return ""


def _do_surface(payload: Dict[str, Any]) -> str:
    """Synchronous surfacing query. Runs in the DB executor. Returns a
    ``[MEMORY]`` block, or ``""`` when there is nothing worth injecting."""
    cfg = _cfg()
    file_path = _file_path_from_payload(payload)
    if not file_path:
        return ""

    now = time.monotonic()
    last = _last_surfaced.get(file_path, 0.0)
    if now - last < float(cfg["debounce_s"]):
        return ""

    from omega.bridge import query_structured

    filename = os.path.basename(file_path)
    dirname = os.path.basename(os.path.dirname(file_path))
    results = query_structured(
        query_text=f"{filename} {dirname} {file_path}",
        limit=int(cfg["limit"]),
        session_id=payload.get("session_id") or None,
        project=payload.get("project") or None,
        context_file=file_path,
    ) or []

    floor = float(cfg["relevance_floor"])
    results = [r for r in results if r.get("relevance", 0.0) >= floor]

    # Record the attempt even when empty so we don't re-query the same file
    # every keystroke-triggered edit within the debounce window.
    _prune_debounce(now, float(cfg["debounce_s"]))
    _last_surfaced[file_path] = now
    if not results:
        return ""

    pc = int(cfg["preview_chars"])
    budget = int(cfg["max_block_chars"])
    header = f"[MEMORY] Relevant prior context for {filename}:"
    lines = [header]
    used = len(header)
    for r in results:
        score = r.get("relevance", 0.0)
        etype = r.get("event_type", "memory")
        preview = (r.get("content", "") or "")[:pc].replace("\n", " ").strip()
        nid = (r.get("id", "") or "")[:8]
        line = f"  [{score:.0%}] {etype}: {preview} (id:{nid})"
        if used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1

    return "\n".join(lines) if len(lines) > 1 else ""


async def _dispatch(hook: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route one hook to its handler. Only surface_memories does work."""
    if hook != "surface_memories":
        return {"output": "", "exit_code": 0}
    try:
        loop = asyncio.get_running_loop()
        from omega.server.mcp_server import _SQLITE_EXECUTOR
        block = await loop.run_in_executor(_SQLITE_EXECUTOR, _do_surface, payload)
    except Exception as e:  # never propagate to the session
        logger.debug("surface_memories failed: %s", e)
        block = ""
    return {"output": block or "", "exit_code": 0}


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Serve one client connection. The client sends a JSON request then
    half-closes; we reply with JSON and close (EOF signals end-of-response)."""
    try:
        data = await reader.read()  # client SHUT_WR → read to EOF
        if not data:
            return  # watchdog probe / empty connection — silent no-op
        try:
            req = json.loads(data.decode("utf-8"))
        except Exception:
            writer.write(json.dumps({"output": "", "exit_code": 0}).encode("utf-8"))
            return

        payload = {k: v for k, v in req.items() if k not in ("hook", "hooks")}
        if req.get("hooks"):  # batch form
            results = [await _dispatch(h, payload) for h in req["hooks"]]
            resp: Dict[str, Any] = {"results": results}
        else:
            resp = await _dispatch(req.get("hook", ""), payload)
        writer.write(json.dumps(resp).encode("utf-8"))
    except Exception as e:
        logger.debug("hook connection error: %s", e)
        try:
            writer.write(json.dumps({"output": "", "exit_code": 0}).encode("utf-8"))
        except Exception:
            pass
    finally:
        try:
            await writer.drain()
        except Exception:
            pass
        try:
            writer.close()
        except Exception:
            pass


async def start_hook_server() -> "asyncio.AbstractServer | None":
    """Create (or return the existing) Unix-socket hook server."""
    global _server
    if _server is not None:
        return _server
    try:
        if SOCK_PATH.exists():
            SOCK_PATH.unlink()
    except OSError:
        pass
    SOCK_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _server = await asyncio.start_unix_server(_handle, path=str(SOCK_PATH))
    try:
        SOCK_PATH.chmod(0o600)
    except OSError:
        pass
    logger.info("OMEGA hook server listening at %s", SOCK_PATH)
    return _server


async def stop_hook_server(_legacy_server: object = None) -> None:
    """Close the hook server and remove its socket.

    Accepts (and ignores) one positional argument: mcp_server.py's shutdown
    paths historically pass the server handle returned by start_hook_server.
    """
    global _server
    if _server is not None:
        _server.close()
        try:
            await _server.wait_closed()
        except Exception:
            pass
        _server = None
    try:
        if SOCK_PATH.exists():
            SOCK_PATH.unlink()
    except OSError:
        pass
