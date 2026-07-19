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

Three hooks are handled: ``surface_memories`` (synchronous block render),
``pre_compact_capture`` and ``session_end`` (fire-and-forget background
capture).  Every other hook name returns an empty no-op response so wiring an
unexpected hook can never do harm.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
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
    "project_debounce_s": 600.0,  # project-knowledge briefing at most every 10 min
}

# Per-project debounce for the knowledge-briefing pass (pass 2).
_last_project_surfaced: Dict[str, float] = {}

# Event types worth surfacing in the project-knowledge briefing — durable
# knowledge only, never episodic exhaust.
_KNOWLEDGE_SURFACE_TYPES = frozenset({
    "decision", "lesson_learned", "constraint", "advisor_insight", "user_preference",
})

_TOKEN_SPLIT_RE = re.compile(r"[_\-.\s]+")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _natural_tokens(name: str) -> str:
    """Turn an identifier ("VehicleSpecService.kt", "sync_cursor_race") into
    natural lowercase words ("vehicle spec service", "sync cursor race").

    Matters because the query pipeline classifies CamelCase/snake_case/paths
    as keyword-sufficient and SKIPS the vector channel — which is exactly the
    old failure mode: a filename-lexical query structurally blind to plans,
    decisions, and lessons that lack file locality.
    """
    stem = name.rsplit(".", 1)[0] if "." in name else name
    words = []
    for part in _TOKEN_SPLIT_RE.split(stem):
        words.extend(_CAMEL_SPLIT_RE.sub(" ", part).split())
    return " ".join(w.lower() for w in words if w)

# Per-file debounce state (process-local; fine for a single long-lived daemon).
# Bounded: pruned opportunistically on insert so a long-lived daemon that has
# touched tens of thousands of distinct files doesn't grow this forever.
_last_surfaced: Dict[str, float] = {}
_MAX_DEBOUNCE_ENTRIES = 512

# The asyncio server handle (set by start_hook_server, cleared by stop).
_server: "asyncio.AbstractServer | None" = None

# (st_dev, st_ino) of the socket file THIS process bound. stop_hook_server
# only unlinks the path when it still points at our inode — a successor
# daemon that re-bound the same path owns a different inode and must not
# have its socket deleted by a stale process's shutdown.
_bound_inode: "tuple[int, int] | None" = None

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
    """Synchronous surfacing. Runs in the DB executor. Returns a
    ``[MEMORY_CONTEXT]`` packet block (rendered by
    context_handlers.build_context_packet — one rendering path shared with
    the MCP ``context_packet`` tool), or ``""`` when there is nothing worth
    injecting.

    Two passes, both semantic (vector channel ON — see _natural_tokens):

    1. File context — what do we know that's relevant to this file?
       (FILE_EDIT profile, per-file debounce)
    2. Project knowledge briefing — active decisions/lessons/constraints for
       this project, regardless of file locality. This is what makes business
       and technical *plans* reachable from a coding session.
       (PLANNING profile, knowledge types only, long per-project debounce)
    """
    cfg = _cfg()
    file_path = _file_path_from_payload(payload)
    if not file_path:
        return ""

    now = time.monotonic()
    floor = float(cfg["relevance_floor"])
    budget = int(cfg["max_block_chars"])
    # ~4 chars/token: the packet's own token budget is the primary bound,
    # the char budget below is the backstop.
    packet_budget = max(120, budget // 4)
    project = payload.get("project") or ""
    session_id = payload.get("session_id") or None

    import omega.bridge as _bridge
    from omega.server import context_handlers as _ctx

    lines: list = []
    pass1_used: set = set()

    def _emit(pkt) -> None:
        if pkt.get("memories_used"):
            lines.extend(pkt["packet_markdown"].splitlines())
        try:
            from omega.telemetry import track_context_packet
            track_context_packet(pkt.get("metrics"), surface="hook")
        except Exception:
            pass

    # --- Pass 1: file-context recall (per-file debounce) ---
    if now - _last_surfaced.get(file_path, 0.0) >= float(cfg["debounce_s"]):
        # Record the attempt even when empty so we don't re-query the same
        # file for every keystroke-triggered edit within the window.
        _prune_debounce(now, float(cfg["debounce_s"]))
        _last_surfaced[file_path] = now

        filename = os.path.basename(file_path)
        dirname = os.path.basename(os.path.dirname(file_path))
        try:
            pkt = _ctx.build_context_packet(
                _bridge._get_store(),
                task=f"{_natural_tokens(filename)} {_natural_tokens(dirname)}".strip(),
                files=[file_path],
                scope={"project": project or None, "session_id": session_id},
                mode="before_edit",
                budget_tokens=packet_budget,
                include_receipt=False,
                min_seed_relevance=floor,
            )
            pass1_used = set(pkt.get("memories_used") or [])
            _emit(pkt)
        except Exception as e:
            logger.debug("surface pass 1 failed: %s", e)

    # --- Pass 2: project knowledge briefing (long per-project debounce) ---
    if project and now - _last_project_surfaced.get(project, 0.0) >= float(cfg["project_debounce_s"]):
        _last_project_surfaced[project] = now
        pname = os.path.basename(project.rstrip("/")) or project
        try:
            pkt = _ctx.build_context_packet(
                _bridge._get_store(),
                task=f"{_natural_tokens(pname)} architecture decisions constraints lessons plans",
                scope={"project": project, "session_id": session_id},
                mode="planning",
                budget_tokens=packet_budget,
                include_receipt=False,
                min_seed_relevance=floor,
                event_types=_KNOWLEDGE_SURFACE_TYPES,
                exclude_ids=pass1_used,
            )
            _emit(pkt)
        except Exception as e:
            logger.debug("surface pass 2 failed: %s", e)

    if not lines:
        return ""

    # Enforce the overall character budget across both packets.
    out: list = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        out.append(line)
        used += len(line) + 1
    # Never end on a dangling header/section line with no entries under it.
    while out and (out[-1].startswith("[MEMORY") or out[-1].rstrip().endswith(":")
                   or not out[-1].strip()):
        out.pop()
    return "\n".join(out) if out else ""


# --------------------------------------------------------------------------
# Pre-compaction capture: snapshot the session into a memory BEFORE Claude
# Code compresses its context away. This is the write-path safety net — since
# per-message auto-capture was removed for latency reasons, memory creation
# otherwise depends entirely on the agent remembering to call omega_store.
# Compaction is rare (unlike per-message hooks), and the LLM digest runs in a
# background task so the hook returns immediately.
# --------------------------------------------------------------------------

_TRANSCRIPT_TAIL_CHARS = 16000

# Keep references so background capture tasks aren't garbage-collected.
_bg_futures: set = set()


def _transcript_tail(transcript_path: str, max_chars: int = _TRANSCRIPT_TAIL_CHARS) -> str:
    """Extract the conversational tail (user prompts + assistant prose) from a
    Claude Code transcript JSONL. Thinking blocks, tool calls, and tool
    results are excluded — they're volume, not session knowledge."""
    parts: list = []
    try:
        with open(transcript_path, "r", errors="replace") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                role = d.get("type")
                if role not in ("user", "assistant"):
                    continue
                content = (d.get("message") or {}).get("content")
                texts: list = []
                if isinstance(content, str):
                    texts.append(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block.get("text", ""))
                if texts:
                    tag = "USER" if role == "user" else "ASSISTANT"
                    parts.append(f"[{tag}] " + " ".join(t for t in texts if t))
    except OSError as e:
        logger.debug("pre_compact: cannot read transcript %s: %s", transcript_path, e)
        return ""
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


_COMPACT_SYSTEM_PROMPT = (
    "You are the memory layer of an engineering assistant. The following is "
    "the tail of a coding session that is about to be compressed. Write a "
    "dense session digest capturing what would otherwise be lost: what was "
    "worked on, decisions made and why, problems found with root causes, and "
    "unfinished threads / next steps. Preserve concrete identifiers (files, "
    "PRs, branches, error messages). 200 words maximum, no preamble."
)


def _do_compact_capture(payload: Dict[str, Any], source: str = "pre-compact") -> None:
    """Background worker: digest the transcript tail and store it. All
    failures are logged and swallowed — capture is best-effort by design.

    *source* tags the stored digest ("pre-compact" or "session-end") so
    residual duplicates between the two capture points stay diagnosable.
    """
    try:
        tail = _transcript_tail(payload.get("transcript_path") or "")
        if len(tail) < 500:
            logger.debug("pre_compact: transcript tail too small, skipping")
            return
        import omega.llm as _llm
        digest = _llm.llm_complete(
            tail, _COMPACT_SYSTEM_PROMPT,
            max_tokens=3000, temperature=0.2, timeout=60.0, model_tier="fast",
        )
        if not (digest or "").strip():
            logger.debug("pre_compact: empty digest, skipping store")
            return
        import omega.bridge as _bridge
        store = _bridge._get_store()
        store.store(
            content=digest.strip(),
            metadata={
                "event_type": "session_summary",
                "project": payload.get("project") or payload.get("cwd") or "",
                "tags": [source, "session-capture"],
                "compact_trigger": payload.get("trigger", ""),
            },
            session_id=payload.get("session_id") or None,
        )
        logger.info("pre_compact: session digest stored (%d chars)", len(digest))
    except Exception as e:
        logger.debug("pre_compact capture failed: %s", e)


def _do_session_end(payload: Dict[str, Any]) -> None:
    """Session-end worker: distill the session's trajectory into a reusable
    skill_template (quality-gated inside distill_trajectory) and store a
    final session digest. Fail-open — a session ending must never block."""
    session_id = payload.get("session_id") or ""
    if session_id:
        try:
            import omega.bridge as _bridge
            _bridge.distill_trajectory(session_id)
        except Exception as e:
            logger.debug("session_end distill failed: %s", e)
    try:
        _do_compact_capture(payload, source="session-end")
    except Exception as e:
        logger.debug("session_end digest failed: %s", e)


async def _dispatch(hook: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Route one hook to its handler."""
    if hook == "session_end":
        # Fire-and-forget, same shape as pre_compact_capture below.
        try:
            loop = asyncio.get_running_loop()
            from omega.server.mcp_server import _SQLITE_EXECUTOR
            fut = loop.run_in_executor(_SQLITE_EXECUTOR, _do_session_end, payload)
            _bg_futures.add(fut)
            fut.add_done_callback(_bg_futures.discard)
        except Exception as e:
            logger.debug("session_end dispatch failed: %s", e)
        return {"output": "", "exit_code": 0}
    if hook == "pre_compact_capture":
        # Fire-and-forget: compaction is blocked on this hook returning, so
        # the digest+store runs in the executor and we answer immediately.
        try:
            loop = asyncio.get_running_loop()
            from omega.server.mcp_server import _SQLITE_EXECUTOR
            fut = loop.run_in_executor(_SQLITE_EXECUTOR, _do_compact_capture, payload)
            _bg_futures.add(fut)
            fut.add_done_callback(_bg_futures.discard)
        except Exception as e:
            logger.debug("pre_compact dispatch failed: %s", e)
        return {"output": "", "exit_code": 0}
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
    """Create (or return the existing, healthy) Unix-socket hook server.

    Self-healing: if the socket path was unlinked externally (a sibling
    session's shutdown), the cached server is listening on an orphaned
    inode that no client can reach — close it and bind a fresh socket.
    """
    global _server, _bound_inode
    if _server is not None:
        if SOCK_PATH.exists():
            return _server  # healthy — idempotent no-op
        old, _server = _server, None
        old.close()
        try:
            await old.wait_closed()
        except Exception:
            pass
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
    try:
        st = SOCK_PATH.stat()
        _bound_inode = (st.st_dev, st.st_ino)
    except OSError:
        _bound_inode = None
    logger.info("OMEGA hook server listening at %s", SOCK_PATH)
    return _server


async def stop_hook_server(_legacy_server: object = None) -> None:
    """Close the hook server and remove its socket.

    Accepts (and ignores) one positional argument: mcp_server.py's shutdown
    paths historically pass the server handle returned by start_hook_server.
    """
    global _server, _bound_inode
    if _server is not None:
        _server.close()
        try:
            await _server.wait_closed()
        except Exception:
            pass
        _server = None
    try:
        if SOCK_PATH.exists():
            st = SOCK_PATH.stat()
            if _bound_inode is None or (st.st_dev, st.st_ino) == _bound_inode:
                SOCK_PATH.unlink()
    except OSError:
        pass
    _bound_inode = None
