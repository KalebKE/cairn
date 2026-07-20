"""Cairn MCP Server -- MCP server for Claude Code (stdio or HTTP daemon).

Supports two transports:
- **stdio** (default): One process per Claude Code session. Zero-config.
- **http** (daemon): One shared process serving all sessions via Streamable HTTP.
  Set CAIRN_TRANSPORT=http or use `cairn serve --daemon`.

Requires the 'server' extra: pip install cairn[server]
"""

import atexit
from cairn.paths import cairn_home
import asyncio
import collections
import logging
import os
import socket
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
except ImportError:
    print(
        "Error: MCP server requires the 'mcp' package.\n"
        "Install with: pip install cairn[server]\n"
        "Or directly: pip install mcp>=1.0.0",
        file=sys.stderr,
    )
    sys.exit(1)

# --- Transport configuration ---
_TRANSPORT = os.environ.get("CAIRN_TRANSPORT", "stdio").lower()
_HTTP_HOST = os.environ.get("CAIRN_HTTP_HOST", "127.0.0.1")
_HTTP_PORT = int(os.environ.get("CAIRN_HTTP_PORT", "8377"))
_start_time = time.monotonic()

from cairn.server.tool_schemas import TOOL_SCHEMAS as _CORE_SCHEMAS, get_condensed_schemas
from cairn.server.handlers import HANDLERS as _CORE_HANDLERS
from cairn.server import handlers as _handlers_module

# Start with core memory tools
TOOL_SCHEMAS = list(_CORE_SCHEMAS)
HANDLERS = dict(_CORE_HANDLERS)

# Fork note: the upstream Pro loader (builtin modules gated on a `pro_tools`
# capability, plus its "run 'cairn activate'" upsell log) was removed — this
# self-hosted fork ships core tools plus discovered plugins only.

# Discover external plugins (e.g. cairn-pro)
from cairn.plugins import discover_plugins

_discovered_plugins = discover_plugins()
for _plugin in _discovered_plugins:
    if _plugin.TOOL_SCHEMAS:
        TOOL_SCHEMAS = TOOL_SCHEMAS + _plugin.TOOL_SCHEMAS
    if _plugin.HANDLERS:
        HANDLERS = {**HANDLERS, **_plugin.HANDLERS}

# ---------------------------------------------------------------------------
# Condensed Mode (CodeMode-inspired) — expose 2 meta-tools + 3 standalone
# instead of 60+ individual tools to save ~80% context tokens.
# On by default. Disable with CAIRN_CONDENSED=0 if needed.
# ---------------------------------------------------------------------------
_CONDENSED_MODE = os.environ.get("CAIRN_CONDENSED", "1") != "0"

# Give handlers access to the full merged schema list and handler registry
# so cairn_tools and cairn_call can discover and dispatch all tools.
_handlers_module._ALL_SCHEMAS = TOOL_SCHEMAS
_handlers_module._ALL_HANDLERS.update(HANDLERS)

# Wire plugin retrieval profiles and score modifiers to SQLiteStore (lazy)
def _wire_plugin_retrieval():
    """Register plugin retrieval profiles and score modifiers on the store."""
    try:
        from cairn.bridge import _get_store
        store = _get_store()
        for plugin in _discovered_plugins:
            if getattr(plugin, "RETRIEVAL_PROFILES", None):
                store.register_plugin_profiles(plugin.RETRIEVAL_PROFILES)
            for modifier in getattr(plugin, "SCORE_MODIFIERS", []):
                store.register_score_modifier(modifier)
    except Exception as e:
        logger.debug("Plugin profile registration failed: %s", e)

# ---------------------------------------------------------------------------
# Dedicated SQLite executor — serializes all blocking DB/hook work onto a
# single thread, preventing the GIL+GC race in _pysqlite_query_execute that
# caused SIGSEGV crashes under concurrent multi-thread SQLite access.
# Also used as the default asyncio executor to cap thread growth.
# ---------------------------------------------------------------------------
_SQLITE_EXECUTOR = ThreadPoolExecutor(
    max_workers=4 if os.environ.get("CAIRN_TRANSPORT", "").lower() == "http" else 2,
    thread_name_prefix="cairn-db",
)

# RSS memory watchdog threshold (bytes). Default 1 GB for stdio, 4 GB for HTTP daemon.
# Override with CAIRN_RSS_LIMIT_MB env var. HTTP daemon serves 8-10 concurrent Claude
# Code sessions; 2 GB is normal operating load. Self-killing at 2 GB is worse than
# running hot — it breaks ALL connected sessions simultaneously.
_RSS_LIMIT_DEFAULT = "8192" if _TRANSPORT == "http" else "1024"
_RSS_LIMIT_BYTES = int(os.environ.get("CAIRN_RSS_LIMIT_MB", _RSS_LIMIT_DEFAULT)) * 1024 * 1024

# Idle watchdog: exit after this many seconds without a tool call.
# Override with CAIRN_IDLE_TIMEOUT env var. 0 = disabled.
_IDLE_TIMEOUT = int(os.environ.get("CAIRN_IDLE_TIMEOUT", "3600"))
_last_activity: float = time.monotonic()

# Shutdown flag — set True during graceful shutdown to reject new tool calls
_shutting_down: bool = False

# Periodic maintenance (cairn.scheduler): tick interval and the no-tool-call
# quiet period required before DB-holding jobs (VACUUM) may run. 0 disables.
_MAINT_INTERVAL_S = int(os.environ.get("CAIRN_MAINT_INTERVAL_S", "900"))
_MAINT_IDLE_THRESHOLD_S = 120


def _close_on_exit():
    """Close SQLite store and unregister PID when the MCP server process exits."""
    global _shutting_down
    _shutting_down = True
    try:
        from cairn.server.pid_registry import unregister_pid
        unregister_pid()
    except Exception:
        pass
    try:
        from cairn.bridge import _close_store

        _close_store()
    except Exception as e:
        logger.debug("Store close at exit failed: %s", e)
    # Shutdown background executor to release threads
    try:
        _SQLITE_EXECUTOR.shutdown(wait=False)
    except Exception:
        pass


atexit.register(_close_on_exit)

logger = logging.getLogger("cairn.server")


_MCP_INSTRUCTIONS = """\
Cairn gives you persistent memory across conversations. \
You MUST use it automatically in every conversation.

REQUIRED at the start of EVERY conversation:
1. Call cairn_welcome() immediately for context briefing
2. Call cairn_protocol() for your operating instructions
3. Follow the protocol it returns

During conversations:
- Before non-trivial tasks: call cairn_query() to check for prior context and decisions
- After completing tasks: call cairn_store(content, "decision") to save key outcomes
- When the user says "remember": call cairn_store(text, "user_preference")
- When context is getting full: call cairn_checkpoint() to save state

These tools are your memory. Use them proactively without being asked.\
"""

_MCP_INSTRUCTIONS_CONDENSED = """\
Cairn gives you persistent memory across conversations. \
You MUST use it automatically in every conversation.

REQUIRED at the start of EVERY conversation:
1. Call cairn_welcome() immediately for context briefing
2. Call cairn_protocol() for your operating instructions
3. Follow the protocol it returns

During conversations:
- Use cairn_store() directly to save memories (decisions, lessons, preferences)
- Use cairn_tools() to discover available tools and their parameters
- Use cairn_call(tool='name', args={...}) to execute any other Cairn tool
- When context is getting full: cairn_call(tool='cairn_checkpoint', args={...})

These tools are your memory. Use them proactively without being asked.\
"""


server = Server(
    "cairn",
    instructions=_MCP_INSTRUCTIONS_CONDENSED if _CONDENSED_MODE else _MCP_INSTRUCTIONS,
)

# ---------------------------------------------------------------------------
# Rate limiting — sliding-window counters (no new deps)
# ---------------------------------------------------------------------------
_GLOBAL_RATE_LIMIT = int(os.environ.get("CAIRN_RATE_LIMIT_GLOBAL", "300"))  # per minute
_WRITE_RATE_LIMIT = int(os.environ.get("CAIRN_RATE_LIMIT_WRITE", "60"))  # per minute
_RATE_WINDOW_S = 60.0

_global_timestamps: collections.deque = collections.deque()
_write_timestamps: collections.deque = collections.deque()

_WRITE_TOOLS = frozenset({
    # Core write tools (handlers.py)
    "cairn_store", "cairn_remember", "cairn_checkpoint", "cairn_remind",
    "cairn_memory", "cairn_maintain", "cairn_reflect",
    # Condensed mode meta-tool (rate-limited by inner tool name in call_tool)
    "cairn_call",
})


def _check_rate_limit(tool_name: str) -> str | None:
    """Return an error message if rate limit exceeded, else None."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW_S

    # Prune expired entries
    while _global_timestamps and _global_timestamps[0] < cutoff:
        _global_timestamps.popleft()

    if len(_global_timestamps) >= _GLOBAL_RATE_LIMIT:
        logger.warning("Global rate limit exceeded (%d calls/min)", _GLOBAL_RATE_LIMIT)
        return f"Rate limit exceeded: {_GLOBAL_RATE_LIMIT} calls/min globally. Try again shortly."

    _global_timestamps.append(now)

    # Write-tool tier
    if tool_name in _WRITE_TOOLS:
        while _write_timestamps and _write_timestamps[0] < cutoff:
            _write_timestamps.popleft()
        if len(_write_timestamps) >= _WRITE_RATE_LIMIT:
            logger.warning("Write rate limit exceeded (%d calls/min) for tool=%s", _WRITE_RATE_LIMIT, tool_name)
            return f"Rate limit exceeded: {_WRITE_RATE_LIMIT} write calls/min. Try again shortly."
        _write_timestamps.append(now)

    return None


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return all Cairn tools (or condensed set if CAIRN_CONDENSED=1)."""
    schemas = get_condensed_schemas(TOOL_SCHEMAS) if _CONDENSED_MODE else TOOL_SCHEMAS
    return [
        Tool(
            name=schema["name"],
            description=schema["description"],
            inputSchema=schema["inputSchema"],
        )
        for schema in schemas
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch tool call to the appropriate handler."""
    global _last_activity
    _last_activity = time.monotonic()

    # Reject new tool calls during shutdown to prevent docling/thread pool errors
    if _shutting_down:
        return [TextContent(type="text", text="Server is shutting down, please retry.")]

    # Rate limiting — for cairn_call, rate-limit against the inner tool name
    rate_name = name
    if name == "cairn_call" and arguments.get("tool"):
        rate_name = arguments["tool"]
    rate_err = _check_rate_limit(rate_name)
    if rate_err:
        return [TextContent(type="text", text=rate_err)]

    handler = HANDLERS.get(name)
    if not handler:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    try:
        # Handlers are async def wrapping synchronous DB/embedding work.
        # Call them directly — the previous run_in_executor + asyncio.run()
        # pattern created a new event loop per call, causing ~1.8 GB/hr memory
        # growth from fragmentation and retained references.
        result = await handler(arguments)
        # Extract text from MCP response format
        content_list = result.get("content", [{}])
        text = content_list[0].get("text", str(result)) if content_list else str(result)

        # Release freed native memory after each tool call to prevent
        # MALLOC_LARGE_REUSABLE accumulation (macOS holds freed pages)
        _force_malloc_release()

        return [TextContent(type="text", text=text)]
    except Exception as e:
        logger.error("Tool %s failed: %s", name, e, exc_info=True)
        error_msg = str(e)
        if "database is locked" in error_msg:
            try:
                from cairn.server.pid_registry import format_lock_diagnostic
                diag = format_lock_diagnostic()
                error_msg = (
                    f"Database locked. {diag}. "
                    f"Check `ps aux | grep cairn` for stale processes and kill them."
                )
            except Exception:
                error_msg = (
                    "Database locked. Another Cairn process may be holding the WAL lock. "
                    "Check `ps aux | grep cairn` for stale processes."
                )
        return [TextContent(type="text", text=f"Error in {name}: {error_msg}")]


async def _idle_watchdog():
    """Exit the process if no tool call has been received within the timeout."""
    while True:
        await asyncio.sleep(30)
        idle = time.monotonic() - _last_activity
        if idle >= _IDLE_TIMEOUT:
            logger.warning("Idle for %.0fs (limit %ds), shutting down.", idle, _IDLE_TIMEOUT)
            _close_on_exit()
            os._exit(0)


async def _socket_watchdog_once():
    """One watchdog iteration: re-create the hook socket if deleted or stale.

    Accesses hook_server attributes via the module so tests (and runtime
    reconfiguration) that patch hook_server.SOCK_PATH are honored. Never
    raises — a failed iteration must not kill the watchdog loop.
    """
    try:
        from cairn.server import hook_server as hs
    except ImportError:
        return
    try:
        sock_path = hs.SOCK_PATH
        if not sock_path:
            return
        if not sock_path.exists():
            logger.warning("Hook socket deleted, re-creating...")
            await hs.start_hook_server()
        else:
            # Validate socket is actually ours and responsive
            try:
                r, w = await asyncio.wait_for(
                    asyncio.open_unix_connection(path=str(sock_path)), timeout=2.0
                )
                w.close()
                await w.wait_closed()
            except (OSError, asyncio.TimeoutError):
                logger.warning("Hook socket unresponsive, re-creating...")
                try:
                    sock_path.unlink()
                except OSError:
                    pass
                await hs.start_hook_server()
    except Exception as e:
        logger.warning("Socket watchdog iteration failed: %s", e)


async def _maintenance_once():
    """One maintenance tick: run due jobs in the SQLite executor.

    Never raises. Uses the shared executor deliberately — running SQLite
    work on a second thread pool would reintroduce the concurrent-access
    SIGSEGV race the shared executor exists to prevent.
    """
    try:
        from cairn.scheduler import run_due_jobs
        loop = asyncio.get_running_loop()

        def _is_idle() -> bool:
            return (time.monotonic() - _last_activity) >= _MAINT_IDLE_THRESHOLD_S

        ran = await loop.run_in_executor(
            _SQLITE_EXECUTOR, run_due_jobs, _is_idle, lambda: _shutting_down)
        if ran:
            logger.info("Maintenance ran: %s", ", ".join(ran))
    except Exception as e:
        logger.warning("Maintenance tick failed: %s", e)


async def _maintenance_scheduler():
    """Periodic maintenance loop (rollup/link/gc/consolidate/compact/backup).

    Initial delay keeps the first tick clear of prewarm, plugin wiring and
    the startup integrity check, which share the same executor.
    """
    await asyncio.sleep(300)
    while True:
        if not _shutting_down:
            await _maintenance_once()
        await asyncio.sleep(_MAINT_INTERVAL_S)


async def _socket_watchdog():
    """Re-create the hook socket if deleted or stale (unresponsive)."""
    try:
        from cairn.server import hook_server  # noqa: F401 — availability probe
    except ImportError:
        logger.debug("hook_server not available, socket watchdog disabled")
        return

    while True:
        await asyncio.sleep(15)
        await _socket_watchdog_once()


def _configure_logging():
    """Set up logging with both stderr and rotating file handler."""
    from logging.handlers import RotatingFileHandler
    from pathlib import Path

    log_dir = cairn_home() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    log_file = log_dir / "cairn.log"

    # Root logger: WARNING to stderr (default for MCP)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # File handler: captures WARNING+ with rotation (5MB, 3 backups)
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=5 * 1024 * 1024, backupCount=3,
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)


# --- Mach task_info objects (hoisted to module level to avoid per-call leaks) ---
_mach_libc = None
_mach_task_port = None
_MACH_TASK_BASIC_INFO = 20

if sys.platform == "darwin":
    try:
        import ctypes
        import ctypes.util

        _mach_libc = ctypes.CDLL(ctypes.util.find_library("c"))
        _mach_task_port = _mach_libc.mach_task_self()

        class _TaskBasicInfo(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_max", ctypes.c_uint64),
                ("user_time_seconds", ctypes.c_int32),
                ("user_time_microseconds", ctypes.c_int32),
                ("system_time_seconds", ctypes.c_int32),
                ("system_time_microseconds", ctypes.c_int32),
                ("policy", ctypes.c_int32),
                ("suspend_count", ctypes.c_int32),
            ]

        _mach_info_size_words = ctypes.sizeof(_TaskBasicInfo()) // 4

        # malloc_zone_pressure_relief — forces macOS malloc to return
        # MALLOC_LARGE_REUSABLE pages to the kernel. Without this, freed
        # large allocations (httpx responses, ONNX tensors, JSON parsing)
        # stay resident as "reusable" pages, inflating RSS by 2-4 GB.
        _malloc_lib = ctypes.CDLL(ctypes.util.find_library("System"))
        _malloc_lib.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _malloc_lib.malloc_zone_pressure_relief.restype = ctypes.c_size_t

        def _force_malloc_release() -> int:
            """Release freed malloc pages back to OS. Returns bytes released."""
            try:
                return _malloc_lib.malloc_zone_pressure_relief(None, 0)
            except Exception:
                return 0
    except Exception:
        _mach_libc = None

        def _force_malloc_release() -> int:
            return 0
else:
    def _force_malloc_release() -> int:
        return 0


def _get_current_rss_bytes() -> int:
    """Get current RSS in bytes using the most accurate OS-specific method.

    On macOS, resource.getrusage ru_maxrss returns *peak* RSS, not current.
    This function uses Mach task_info for accurate current RSS on macOS,
    falling back to ru_maxrss if the Mach call fails.
    """
    if sys.platform == "darwin" and _mach_libc is not None:
        try:
            info = _TaskBasicInfo()
            count = ctypes.c_uint32(_mach_info_size_words)
            ret = _mach_libc.task_info(
                _mach_task_port, _MACH_TASK_BASIC_INFO,
                ctypes.byref(info), ctypes.byref(count),
            )
            if ret == 0:  # KERN_SUCCESS
                return info.resident_size
        except Exception:
            pass

    # Fallback: getrusage (peak RSS, not current — but better than nothing)
    try:
        import resource
    except ModuleNotFoundError:
        return 0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        rss *= 1024  # KB to bytes on Linux
    return rss


async def _rss_watchdog():
    """Periodically check RSS and gracefully exit if it exceeds the limit.

    Prevents runaway memory growth from crashing the terminal. The MCP
    client (Claude Code) will automatically restart the server on next
    tool call.
    """
    import gc

    while True:
        await asyncio.sleep(15)
        try:
            rss = _get_current_rss_bytes()

            # Proactive memory release when RSS exceeds 50% of limit.
            # macOS malloc holds freed large allocations as MALLOC_LARGE_REUSABLE
            # pages (2-4 GB observed). malloc_zone_pressure_relief forces release.
            if rss > _RSS_LIMIT_BYTES * 0.5:
                gc.collect()
                released = _force_malloc_release()
                rss_after = _get_current_rss_bytes()
                logger.warning(
                    "Memory pressure relief: RSS %.1f MB -> %.1f MB "
                    "(malloc released %d bytes, limit %.0f MB)",
                    rss / 1024**2, rss_after / 1024**2,
                    released, _RSS_LIMIT_BYTES / 1024**2,
                )
                rss = rss_after

            # Tracemalloc snapshots: log top allocations periodically when RSS is high
            if os.environ.get("CAIRN_TRACEMALLOC") and rss > _RSS_LIMIT_BYTES * 0.3:
                import tracemalloc as _tm
                if _tm.is_tracing():
                    # Throttle snapshots to once per 60s
                    _now_snap = time.monotonic()
                    if not hasattr(_rss_watchdog, '_last_snap') or (_now_snap - _rss_watchdog._last_snap) > 60:
                        _rss_watchdog._last_snap = _now_snap
                        snapshot = _tm.take_snapshot()
                        top_stats = snapshot.statistics("lineno")
                        logger.warning("tracemalloc top 15 allocations (RSS %.0f MB):", rss / 1024**2)
                        for stat in top_stats[:15]:
                            logger.warning("  %s", stat)

            if rss > _RSS_LIMIT_BYTES:
                global _shutting_down
                _shutting_down = True
                msg = (
                    "RSS %.0f MB exceeds limit %.0f MB, shutting down to prevent crash."
                    % (rss / 1024**2, _RSS_LIMIT_BYTES / 1024**2)
                )
                # Final tracemalloc snapshot before exit (top 25)
                if os.environ.get("CAIRN_TRACEMALLOC"):
                    import tracemalloc as _tm
                    if _tm.is_tracing():
                        snapshot = _tm.take_snapshot()
                        top_stats = snapshot.statistics("lineno")
                        logger.warning("FINAL tracemalloc top 25 (before kill at RSS %.0f MB):", rss / 1024**2)
                        for stat in top_stats[:25]:
                            logger.warning("  %s", stat)
                # Force flush to stderr (unbuffered) before os._exit
                sys.stderr.write(f"WARNING cairn.server: {msg}\n")
                sys.stderr.flush()
                logger.warning(msg)
                # Flush all log handlers before hard exit
                for handler in logging.getLogger().handlers:
                    try:
                        handler.flush()
                    except Exception:
                        pass
                _close_on_exit()
                os._exit(0)
        except Exception as e:
            # Log watchdog errors instead of silently swallowing them
            logger.debug("RSS watchdog check failed: %s", e)


def _check_port_available(host: str, port: int) -> bool:
    """Check if a TCP port is available for binding."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


async def _run_http_transport(hook_srv) -> None:
    """Run the MCP server as a Streamable HTTP daemon via uvicorn.

    Uses StreamableHTTPSessionManager to handle multiple concurrent
    Claude Code sessions over a single shared process.
    """
    try:
        import uvicorn
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    except ImportError as e:
        print(
            f"Error: HTTP transport requires additional packages: {e}\n"
            "Install with: pip install cairn[server] starlette uvicorn",
            file=sys.stderr,
        )
        sys.exit(1)

    # Fail fast if port is already bound (another daemon running)
    if not _check_port_available(_HTTP_HOST, _HTTP_PORT):
        print(
            f"Error: Port {_HTTP_PORT} already in use on {_HTTP_HOST}.\n"
            f"Another Cairn daemon may be running. Check: curl http://{_HTTP_HOST}:{_HTTP_PORT}/health",
            file=sys.stderr,
        )
        sys.exit(1)

    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=False,
        # Stateless mode: each request is self-contained. Cairn tool calls are
        # already stateless (state lives in SQLite). This makes daemon restarts
        # invisible to clients — no more "Session not found" errors after crash.
        stateless=True,
    )

    async def health(request):
        """Health check endpoint with process diagnostics."""
        rss = _get_current_rss_bytes()
        return JSONResponse({
            "status": "ok",
            "pid": os.getpid(),
            "rss_mb": round(rss / 1024**2, 1),
            "uptime_s": round(time.monotonic() - _start_time, 1),
            "tool_count": len(TOOL_SCHEMAS),
            "transport": "http",
        })

    import contextlib

    try:
        from cairn.server.hook_server import stop_hook_server as _stop_hook_srv
    except ImportError:
        async def _stop_hook_srv(*args, **kwargs):
            pass

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            logger.info(
                "Cairn MCP daemon listening on http://%s:%d/mcp",
                _HTTP_HOST, _HTTP_PORT,
            )
            yield
        await _stop_hook_srv(hook_srv)

    app = Starlette(
        routes=[
            Route("/health", health, methods=["GET"]),
            Mount("/mcp", app=session_manager.handle_request),
        ],
        lifespan=lifespan,
    )

    config = uvicorn.Config(
        app,
        host=_HTTP_HOST,
        port=_HTTP_PORT,
        log_level="warning",
        timeout_graceful_shutdown=5,
    )
    uv_server = uvicorn.Server(config)

    # Handle SIGTERM gracefully (launchd sends this on unload)
    # Track shutdown task to prevent duplicate asyncio tasks on repeated signals
    _shutdown_task = None

    def _handle_shutdown_signal():
        nonlocal _shutdown_task
        global _shutting_down
        _shutting_down = True
        if _shutdown_task is None or _shutdown_task.done():
            _shutdown_task = asyncio.ensure_future(uv_server.shutdown())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _handle_shutdown_signal)

    await uv_server.serve()


async def main():
    """Entry point for the Cairn MCP server."""
    _configure_logging()
    logger.info("Starting Cairn MCP server...")

    # One-time legacy data migration (~/.omega -> ~/.cairn) before the store
    # opens. No-op once migrated or when CAIRN_HOME is set. Never destructive.
    try:
        from cairn._compat import migrate_home, needs_home_migration
        if needs_home_migration():
            logger.warning("Legacy ~/.omega store found — migrating to ~/.cairn...")
            migrate_home()
    except Exception as e:
        logger.warning("Home migration check failed (non-fatal): %s", e)

    # --- Startup memory baseline ---
    _startup_rss = _get_current_rss_bytes()
    logger.warning(
        "Startup RSS: %.1f MB, PID: %d, transport: %s",
        _startup_rss / 1024**2, os.getpid(), _TRANSPORT,
    )

    # Optional tracemalloc for memory leak diagnosis (enable via env var)
    if os.environ.get("CAIRN_TRACEMALLOC"):
        import tracemalloc
        tracemalloc.start(10)
        logger.warning("tracemalloc enabled for memory leak diagnosis")

    # --- P1 fixes: prevent Metal shader cache growth and cap thread pool ---
    # PyTorch MPS (Metal) allocates an unbounded shader compilation cache that
    # grows ~200 MB/min. Disable it since we only use ONNX for embeddings.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    # Disable in-process cross-encoder reranker in HTTP daemon mode.
    # The bge-reranker-v2-m3 ONNX model costs ~2 GB RSS. With multiple
    # concurrent sessions the daemon easily hits the RSS limit and enters
    # a crash loop. Embedding similarity alone is sufficient for query quality.
    if _TRANSPORT == "http":
        os.environ.setdefault("CAIRN_CROSS_ENCODER", "0")
    # Cap asyncio's default executor — prevents unbounded thread growth from
    # run_in_executor(None, ...) calls (was reaching 49-64 threads at crash).
    # Note: the surfacing hook (hook_server) shares _SQLITE_EXECUTOR with MCP
    # tools so all blocking SQLite work stays serialized on one pool.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=2, thread_name_prefix="cairn-default"))

    # --- Kill orphaned MCP servers before touching the DB ---
    # Orphans from dead Claude sessions hold DB locks; must be killed first.
    # Skip in HTTP daemon mode — daemon runs under launchd with ppid=1 by design.
    if _TRANSPORT != "http":
        try:
            from cairn.server.pid_registry import kill_orphaned_servers
            killed = kill_orphaned_servers()
            if killed:
                logger.warning("Killed %d orphaned MCP server(s) at startup", killed)
        except Exception as e:
            logger.debug("Orphan cleanup at startup failed: %s", e)

    # Fork note: the upstream crash-recovery pass (_cleanup_dead_sessions) is
    # gone — it operated on coord_* tables that don't exist in this build and
    # pushed session status to the removed cloud sync.

    # Register this process for lock diagnostics
    try:
        from cairn.server.pid_registry import register_pid
        register_pid(
            transport=_TRANSPORT,
            port=_HTTP_PORT if _TRANSPORT == "http" else None,
        )
    except Exception:
        pass

    # Start UDS hook server for fast hook dispatch
    try:
        from cairn.server.hook_server import start_hook_server, stop_hook_server
    except ImportError:
        async def start_hook_server(*args, **kwargs):
            return None
        async def stop_hook_server(*args, **kwargs):
            pass

    hook_srv = await start_hook_server()

    # Prewarm embedding: try shared daemon first, fall back to in-process ONNX.
    # Daemon eliminates per-process model duplication (~170MB each).
    async def _prewarm():
        # Try daemon with retries (daemon may be busy handling another MCP server)
        for _attempt in range(3):
            try:
                from cairn.embedding_client import get_client

                client = get_client()
                if client is not None:
                    health = client.health()
                    if health and health.get("status") == "ok":
                        logger.info("Embedding daemon available, skipping local model load")
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
        # Fall back to in-process preload
        try:
            from cairn.embedding import preload_embedding_model_async
            await preload_embedding_model_async()
        except Exception as e:
            logger.debug("Embedding model prewarm failed: %s", e)

    _prewarm_task = asyncio.create_task(_prewarm())

    # Wire plugin retrieval profiles/modifiers to SQLiteStore.
    # Run in executor to avoid blocking the event loop during store init
    # (store init triggers PRAGMA integrity_check which can take 30+ seconds).
    async def _wire_plugins_async():
        await loop.run_in_executor(_SQLITE_EXECUTOR, _wire_plugin_retrieval)

    _wire_plugins_task = asyncio.create_task(_wire_plugins_async())

    # Enrich MCP instructions with memory stats (runs once at startup)
    def _enrich_instructions():
        try:
            from cairn.bridge import _get_store
            store = _get_store()
            count = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            type_count = store._conn.execute(
                "SELECT COUNT(DISTINCT json_extract(metadata, '$.event_type')) FROM memories"
            ).fetchone()[0]
            stats_line = f"\n\nCurrent state: {count} memories across {type_count} types."
            base = _MCP_INSTRUCTIONS_CONDENSED if _CONDENSED_MODE else _MCP_INSTRUCTIONS
            server.instructions = base + stats_line
        except Exception as e:
            logger.debug("MCP instructions enrichment failed (non-fatal): %s", e)

    async def _enrich_async():
        await _wire_plugins_task  # Wait for store to be initialized
        await loop.run_in_executor(_SQLITE_EXECUTOR, _enrich_instructions)

    _enrich_task = asyncio.create_task(_enrich_async())

    # Start idle watchdog (unless disabled).
    # IMPORTANT: Save reference — unref'd tasks get silently GC'd by asyncio.
    # In HTTP daemon mode, idle timeout should be 0 (managed by launchd).
    if _IDLE_TIMEOUT > 0 and _TRANSPORT != "http":
        _watchdog_task = asyncio.create_task(_idle_watchdog())

    # Socket watchdog — re-creates hook.sock if deleted by another session's stop
    _sock_watchdog_task = asyncio.create_task(_socket_watchdog())

    # Periodic maintenance (both transports; flock-guarded markers make it
    # safe when several daemon processes run concurrently)
    if _MAINT_INTERVAL_S > 0:
        _maint_task = asyncio.create_task(_maintenance_scheduler())

    # Fork note: the upstream background coordination loop was removed — it
    # depended entirely on the absent Pro coordination module and woke every
    # 60-90s only to swallow an ImportError.

    # RSS memory watchdog — graceful exit before memory pressure causes SIGSEGV
    _rss_watchdog_task = asyncio.create_task(_rss_watchdog())

    if _TRANSPORT == "http":
        await _run_http_transport(hook_srv)
    else:
        try:
            async with stdio_server() as (read_stream, write_stream):
                await server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
        finally:
            await stop_hook_server(hook_srv)


if __name__ == "__main__":
    asyncio.run(main())
