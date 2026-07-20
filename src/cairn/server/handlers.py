"""
Cairn MCP Handlers -- Maps tool names to async handler functions.

Each handler delegates to cairn.bridge for actual operations and returns
MCP-compatible response dicts.
"""

# HANDLERS is provided lazily via module __getattr__ (see bottom of file); it is
# derived from the ToolSpec table in cairn.server.registry.

import json
import logging
import os
import time
from pathlib import Path
from cairn.paths import cairn_home
from typing import Any, Dict

logger = logging.getLogger("cairn.server.handlers")

# ---------------------------------------------------------------------------
# Deploy gate tracking — file-based so it works in daemon + fallback modes
# ---------------------------------------------------------------------------

def _gate_dir() -> Path:
    """Directory for deploy-gate / coord markers.

    Resolved at call time via ``cairn_home()`` so it follows a CAIRN_HOME
    change; freezing it at import time was a stale-state bug (a relocated
    home, or a per-case home in tests, kept writing to the original dir).
    """
    return cairn_home() / "gates"


def _mark_deploy_gate_cleared(session_id: str | None = None) -> None:
    """Mark the deploy gate as cleared for a session."""
    try:
        _gate_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
        key = session_id or "default"
        gate_file = _gate_dir() / f"{key}.gate"
        gate_file.write_text(str(time.time()))
    except Exception as e:
        logger.debug("Deploy gate write failed: %s", e)


def _mark_coord_status_checked(session_id: str | None = None) -> None:
    """Mark that coord_status was checked for a session."""
    try:
        _gate_dir().mkdir(parents=True, exist_ok=True, mode=0o700)
        key = session_id or "default"
        gate_file = _gate_dir() / f"{key}.coord"
        gate_file.write_text(str(time.time()))
    except Exception as e:
        logger.debug("Coord status write failed: %s", e)


def _is_coord_status_checked(session_id: str | None = None, max_age_sec: int = 1800) -> bool:
    """Check if coord_status was checked recently (default: 30 min)."""
    try:
        candidates = []
        if session_id:
            candidates.append(_gate_dir() / f"{session_id}.coord")
        candidates.append(_gate_dir() / "default.coord")
        for gate_file in candidates:
            if gate_file.exists():
                ts = float(gate_file.read_text().strip())
                if (time.time() - ts) < max_age_sec:
                    return True
        return False
    except Exception as e:
        logger.debug("Coord status read failed: %s", e)
        return False  # fail-closed


def _is_pro_available() -> bool:
    """Check if an installed extension advertises Pro capabilities."""
    try:
        from cairn.plugins import has_capability
        return has_capability("pro_tools")
    except Exception:
        return False


# Fork note: the upstream "nagware" block (periodic $19/mo upsell messages
# appended to cairn_query/cairn_welcome/cairn_store results) was removed —
# this is a self-hosted install; injecting marketing copy into model context
# costs tokens and pollutes answers.


def _full_retrieval_available() -> bool:
    """Check whether a full-retrieval extension capability is available."""
    try:
        from cairn.plugins import has_capability
        return has_capability("full_retrieval")
    except Exception:
        return False


def is_deploy_gate_cleared(session_id: str | None = None, max_age_sec: int = 1800) -> bool:
    """Check if the deploy gate was cleared recently (default: 30 min).

    Requires cairn_query(event_type="decision") to have been called.
    Also requires cairn_coord_status if pro modules are available.
    Checks session-specific markers first, then 'default'.
    """
    try:
        # Check decision query marker
        decision_ok = False
        candidates = []
        if session_id:
            candidates.append(_gate_dir() / f"{session_id}.gate")
        candidates.append(_gate_dir() / "default.gate")
        for gate_file in candidates:
            if gate_file.exists():
                ts = float(gate_file.read_text().strip())
                if (time.time() - ts) < max_age_sec:
                    decision_ok = True
                    break

        if not decision_ok:
            return False

        # Require coord_status check only when pro is available
        if not _is_pro_available():
            return True

        return _is_coord_status_checked(session_id, max_age_sec)
    except Exception as e:
        logger.debug("Deploy gate check failed: %s", e)
        return False  # fail-closed for safety


def _clamp_int(value, default: int, min_val: int = 1, max_val: int = 10000) -> int:
    """Clamp a numeric argument to safe bounds."""
    try:
        v = int(value)
        return max(min_val, min(v, max_val))
    except (TypeError, ValueError):
        return default


# Safe directory for export/import operations — resolved at call time so it
# follows CAIRN_HOME (import-time freezing was a stale-state bug).
def _safe_export_dir() -> Path:
    """Root that export/import paths must stay under, resolved at call time."""
    return cairn_home()


# ---------------------------------------------------------------------------
# Input validation helpers — prevent path traversal and injection
# ---------------------------------------------------------------------------

import re as _re

_SAFE_ID_RE = _re.compile(r"^[a-zA-Z0-9._-]+$")


def _validate_session_id(session_id: str | None) -> str | None:
    """Validate session_id to prevent path traversal."""
    if not session_id:
        return session_id
    if ".." in session_id or "/" in session_id or "\\" in session_id:
        logger.warning("Rejected session_id with path traversal: %s", session_id[:50])
        return None
    if not _SAFE_ID_RE.match(session_id):
        logger.warning("Rejected session_id with invalid chars: %s", session_id[:50])
        return None
    return session_id


def _validate_entity_id(entity_id: str | None) -> str | None:
    """Validate entity_id format (alphanumeric, hyphens, dots, underscores)."""
    if not entity_id:
        return entity_id
    if not _SAFE_ID_RE.match(entity_id):
        logger.warning("Rejected entity_id with invalid chars: %s", entity_id[:50])
        return None
    return entity_id


from cairn.server.responses import mcp_response, mcp_error  # noqa: E402


# ---------------------------------------------------------------------------
# Post-write validation guard (arxiv 2602.19320 §5.2 — backbone resilience)
# ---------------------------------------------------------------------------
# Smaller backbone models produce malformed metadata (17-30% format error
# rate).  This guard normalizes inputs before they reach the store.

_KNOWN_EVENT_TYPES = frozenset({
    "memory", "decision", "lesson_learned", "error_pattern", "observation",
    "user_preference", "behavioral_pattern", "constraint", "reminder",
    "session_summary", "code_pattern", "entity_update", "infrastructure",
    "session_end", "context", "progress",
})


def _validate_memory_write(content: str, event_type: str, metadata: Any) -> tuple:
    """Validate and normalize memory write inputs.

    Returns (event_type, metadata, errors) where errors is a list of
    format issues that were auto-corrected.
    """
    errors: list = []

    # Metadata must be a dict — string/list/int are common backbone errors
    if metadata is None:
        metadata = {}
    elif isinstance(metadata, str):
        # Backbone emitted metadata as JSON string instead of dict
        errors.append("metadata was str, attempted JSON parse")
        try:
            import json as _json
            parsed = _json.loads(metadata)
            if isinstance(parsed, dict):
                metadata = parsed
            else:
                metadata = {"_raw": metadata}
                errors.append("parsed JSON was not a dict, wrapped in _raw")
        except Exception:
            metadata = {"_raw": metadata}
            errors.append("metadata JSON parse failed, wrapped in _raw")
    elif not isinstance(metadata, dict):
        errors.append(f"metadata was {type(metadata).__name__}, replaced with empty dict")
        metadata = {}

    # Event type normalization
    if not isinstance(event_type, str) or not event_type:
        errors.append(f"event_type was {type(event_type).__name__}({event_type!r}), defaulted to 'memory'")
        event_type = "memory"
    elif event_type not in _KNOWN_EVENT_TYPES:
        # Allow unknown types but log — don't block extensibility
        errors.append(f"event_type '{event_type}' not in known set (allowed)")

    return event_type, metadata, errors


# ============================================================================
# Handler: cairn_store (also handles cairn_remember as alias)
# ============================================================================


def _broadcast_decision(session_id: str, project: str, content: str):
    """Best-effort broadcast of a stored decision to active peers."""
    try:
        from cairn_platform.orchestrator.coordination import get_manager
        mgr = get_manager()

        # Only broadcast if there are active peers
        sessions = mgr.list_sessions(auto_clean=False)
        peers = [s for s in sessions if s.get("session_id") != session_id]
        if not peers:
            return

        # Truncate to first meaningful line for the subject
        first_line = content.split("\n")[0].strip()[:120]
        mgr.send_message(
            from_session=session_id,
            subject=f"Decision stored: {first_line}",
            msg_type="inform",
            project=project,
            ttl_minutes=120,
        )
    except Exception as e:
        logger.debug("Decision broadcast failed: %s", e)


# Domain keywords for auto-classification of decisions
_DOMAIN_KEYWORDS = {
    "auth": ["auth", "login", "password", "session", "token", "oauth", "credential"],
    "deploy": ["deploy", "vercel", "netlify", "docker", "k8s", "ci/cd", "pipeline"],
    "testing": ["test", "pytest", "jest", "coverage", "e2e", "unit test"],
    "database": ["database", "postgres", "mysql", "sqlite", "supabase", "migration", "schema"],
    "api": ["api", "endpoint", "route", "rest", "graphql"],
    "frontend": ["frontend", "react", "next.js", "tailwind", "component", "ui", "ux"],
    "architecture": ["architecture", "refactor", "module", "pattern", "structure"],
}


def _extract_decision_domain(content: str) -> str:
    """Extract a domain from decision content using keyword matching."""
    lower = content.lower()
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return domain
    return "general"


def _auto_register_decision(
    mgr,
    session_id: str,
    project: str,
    content: str,
    entity_id=None,
):
    """Auto-register a decision in coordination when cairn_store gets a decision type.
    Returns the registered decision dict or None if skipped/failed."""
    if mgr is None:
        return None

    try:
        domain = _extract_decision_domain(content)
        return mgr.register_decision(
            session_id=session_id,
            project=project or "",
            domain=domain,
            decision=content[:500],
            rationale="Auto-registered from cairn_store(event_type='decision')",
        )
    except Exception:
        return None


async def handle_cairn_store(arguments: dict) -> dict:
    """Store a memory with optional type and metadata.

    Accepts 'text' as alias for 'content' for backward compat with cairn_remember.
    Defaults event_type to 'memory' when not provided.
    """
    # Batch mode: store multiple items at once
    items = arguments.get("items")
    if items is not None:
        if not isinstance(items, list):
            return mcp_error("items must be a list")
        if not items:
            return mcp_response({"ids": [], "count": 0})
        try:
            from cairn.bridge import batch_store
            result = batch_store(items)
            return mcp_response(result)
        except Exception as e:
            logger.error("batch_store failed: %s", e, exc_info=True)
            return mcp_error("Batch store failed")

    content = arguments.get("content", "").strip()
    # Support 'text' as alias for 'content' (backward compat with cairn_remember)
    if not content:
        content = arguments.get("text", "").strip()
    if not content:
        return mcp_error("content (or text) is required")

    raw_event_type = arguments.get("event_type", "memory")
    raw_metadata = arguments.get("metadata", {})
    event_type, metadata, format_errors = _validate_memory_write(
        content, raw_event_type, raw_metadata,
    )
    if format_errors:
        logger.info("Memory write format corrections: %s", format_errors)
        try:
            from cairn.bridge import _get_store
            _store = _get_store()
            for err in format_errors:
                _store.record_format_error("cairn_store", err)
        except Exception:
            pass

    session_id = _validate_session_id(arguments.get("session_id"))
    project = arguments.get("project") or (metadata or {}).get("project") or os.getcwd()
    entity_id = _validate_entity_id(arguments.get("entity_id"))
    agent_type = arguments.get("agent_type")

    # Wire through priority if provided
    priority = arguments.get("priority")
    if priority is not None:
        try:
            priority = max(1, min(5, int(priority)))
            metadata = dict(metadata or {})
            metadata["priority"] = priority
        except (TypeError, ValueError):
            pass

    # Context graph fields
    derived_from = arguments.get("derived_from")
    source_uri = arguments.get("source_uri")
    status = arguments.get("status")

    # Wire context graph fields into metadata for bridge passthrough
    if derived_from:
        metadata = dict(metadata or {})
        metadata["derived_from"] = derived_from
    if source_uri:
        metadata = dict(metadata or {})
        metadata["source_uri"] = source_uri
    if status:
        metadata = dict(metadata or {})
        metadata["status"] = status

    try:
        from cairn.bridge import store

        result = store(
            content=content,
            event_type=event_type,
            metadata=metadata,
            session_id=session_id,
            project=project,
            entity_id=entity_id,
            agent_type=agent_type,
        )

        # Broadcast decisions to active peers for real-time awareness
        if event_type == "decision" and session_id and project:
            _broadcast_decision(session_id, project, content)

        # Auto-register decisions in coordination (Part C of utilization boost)
        if event_type == "decision" and session_id:
            try:
                from cairn_platform.orchestrator.coordination import get_manager
                mgr = get_manager()
                _auto_register_decision(mgr, session_id, project, content, entity_id)
            except Exception:
                pass  # Non-critical

        # Surface prior decision trail for consistency awareness
        if event_type == "decision" and content:
            try:
                from cairn.bridge import query_structured
                from cairn.server.hook_server.cards import format_decision_trail_card

                prior = query_structured(
                    query_text=content[:200],
                    event_type="decision",
                    limit=5,
                    project=project,
                    entity_id=entity_id,
                )
                # Exclude the memory we just stored (result contains its ID)
                new_id = ""
                if result and "mem-" in result:
                    import re as _re
                    _id_match = _re.search(r"(mem-[a-f0-9]+)", result)
                    if _id_match:
                        new_id = _id_match.group(1)
                prior_filtered = [
                    d for d in (prior or [])
                    if d.get("id") != new_id and d.get("relevance", 0) >= 0.30
                ]
                if prior_filtered:
                    # Build trail format: need date + content + status
                    trail_decisions = []
                    for d in prior_filtered[:5]:
                        created = d.get("created_at", "")[:10] or "unknown"
                        trail_decisions.append({
                            "date": created,
                            "content": d.get("content", ""),
                            "status": "active",
                        })
                    topic = content[:60].replace("\n", " ").strip()
                    trail = format_decision_trail_card(topic=topic, decisions=trail_decisions)
                    if trail:
                        result = result + "\n\n" + trail
            except Exception as e:
                logger.debug("decision trail surfacing failed: %s", e)

        # Attach finding to active intent ("already explored" signal)
        if event_type in ("decision", "lesson_learned") and session_id:
            try:
                from cairn_platform.orchestrator.coordination import get_manager
                mgr = get_manager()
                mgr.attach_finding(session_id, content[:300])
            except Exception as e:
                logger.debug("attach_finding skipped: %s", e)

        # Track tool call for telemetry
        try:
            from cairn.telemetry import track_tool_call
            track_tool_call("cairn_store")
        except Exception:
            pass

        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_store failed: %s", e, exc_info=True)
        return mcp_error(f"Failed to store memory: {e}")


# ============================================================================
# Handler: cairn_query
# ============================================================================


async def handle_cairn_query(arguments: dict) -> dict:
    """Search memories — semantic (default), exact phrase match, timeline, or browse."""
    mode = arguments.get("mode", "semantic")

    # Timeline mode — delegate to handle_cairn_timeline
    if mode == "timeline":
        return await handle_cairn_timeline(arguments)

    # Browse mode — list memories by type, session, or recent
    if mode == "browse":
        return await handle_cairn_browse(arguments)

    # Trace mode — session tool call timeline
    if mode == "trace":
        return await handle_cairn_trace(arguments)

    # Unified mode — cross-search memories + knowledge documents
    if mode == "unified":
        query_text = arguments.get("query", "").strip()
        if not query_text:
            return mcp_error("query is required for unified mode")
        limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=100)
        project = arguments.get("project")
        entity_id = _validate_entity_id(arguments.get("entity_id"))
        results = []
        # Memory search
        try:
            from cairn.bridge import query as memory_query
            mem_result = memory_query(query_text=query_text, limit=limit, project=project, entity_id=entity_id)
            if isinstance(mem_result, str):
                results.append({"source": "memory", "data": mem_result})
            elif isinstance(mem_result, dict):
                results.append({"source": "memory", **mem_result})
        except Exception as e:
            logger.warning("unified: memory search failed: %s", e)
            results.append({"source": "memory", "error": str(e)})
        # Knowledge document search
        try:
            from cairn_platform.knowledge.engine import search_documents
            doc_result = search_documents(query=query_text, limit=limit, entity_id=entity_id)
            results.append({"source": "document", "data": doc_result})
        except ImportError:
            results.append({"source": "document", "note": "Knowledge module not available"})
        except Exception as e:
            logger.warning("unified: document search failed: %s", e)
            results.append({"source": "document", "error": str(e)})
        return mcp_response({"mode": "unified", "results": results})

    query_text = arguments.get("query", "").strip()
    if not query_text:
        return mcp_error("query is required")

    # Phrase mode — delegate to bridge.phrase_search
    if mode == "phrase":
        limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=1000)
        event_type = arguments.get("event_type")
        project = arguments.get("project")
        case_sensitive = arguments.get("case_sensitive", False)
        try:
            from cairn.bridge import phrase_search

            result = phrase_search(
                phrase=query_text,
                limit=limit,
                event_type=event_type,
                project=project,
                case_sensitive=case_sensitive,
            )
            return mcp_response(result)
        except Exception as e:
            logger.error("cairn_query (phrase) failed: %s", e, exc_info=True)
            return mcp_error("Phrase search failed")

    # Regex mode — Python-side re over a bounded recency scan
    if mode == "regex":
        limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=1000)
        try:
            from cairn.bridge import regex_search

            result = regex_search(
                pattern=query_text,
                limit=limit,
                event_type=arguments.get("event_type"),
                project=arguments.get("project"),
                case_sensitive=arguments.get("case_sensitive", False),
            )
            return mcp_response(result)
        except ValueError as e:
            return mcp_error(str(e))
        except Exception as e:
            logger.error("cairn_query (regex) failed: %s", e, exc_info=True)
            return mcp_error("Regex search failed")

    # Semantic mode (default)
    limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=1000)
    event_type = arguments.get("event_type")
    project = arguments.get("project")
    session_id = _validate_session_id(arguments.get("session_id"))
    context_file = arguments.get("context_file")
    context_tags = arguments.get("context_tags")
    filter_tags = arguments.get("filter_tags")
    raw_temporal = arguments.get("temporal_range")
    temporal_range = tuple(raw_temporal) if raw_temporal and len(raw_temporal) == 2 else None
    entity_id = _validate_entity_id(arguments.get("entity_id"))
    agent_type = arguments.get("agent_type")
    scope = arguments.get("scope")  # "session" to restrict to own session, None for all
    perspective = arguments.get("perspective")  # Behavioral diversity: implementation/critique/verification
    strength_min = arguments.get("strength_min")
    if strength_min is not None:
        strength_min = max(0.0, min(1.0, float(strength_min)))
    memory_type = arguments.get("memory_type")
    if memory_type and memory_type not in ("episodic", "semantic", "procedural"):
        memory_type = None
    include_contradicted = arguments.get("include_contradicted", False)
    valid_at = arguments.get("valid_at")
    status_filter = arguments.get("status")

    # Map context param to SurfacingContext enum
    surfacing_context = None
    context_param = arguments.get("context")
    if context_param:
        try:
            from cairn.sqlite_store import SurfacingContext
            _context_map = {
                "general": SurfacingContext.GENERAL,
                "error_debug": SurfacingContext.ERROR_DEBUG,
                "file_edit": SurfacingContext.FILE_EDIT,
                "planning": SurfacingContext.PLANNING,
                "review": SurfacingContext.REVIEW,
            }  # SESSION_START excluded — internal use only
            surfacing_context = _context_map.get(context_param)
        except ImportError:
            pass

    try:
        from cairn.bridge import query

        result = query(
            query_text=query_text,
            limit=limit,
            event_type=event_type,
            project=project,
            session_id=session_id,
            context_file=context_file,
            context_tags=context_tags,
            filter_tags=filter_tags,
            temporal_range=temporal_range,
            entity_id=entity_id,
            agent_type=agent_type,
            scope=scope,
            surfacing_context=surfacing_context,
            perspective=perspective,
            strength_min=strength_min,
            memory_type=memory_type,
            include_contradicted=include_contradicted,
            valid_at=valid_at,
            status=status_filter,
        )

        # Mark deploy gate as cleared when querying decisions
        if event_type == "decision":
            _mark_deploy_gate_cleared(session_id)

        # Track tool call for telemetry
        try:
            from cairn.telemetry import track_tool_call
            track_tool_call("cairn_query")
        except Exception:
            pass

        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_query failed: %s", e, exc_info=True)
        return mcp_error("Query failed")


# ============================================================================
# Handler: cairn_query mode=trace
# ============================================================================


async def handle_cairn_trace(arguments: dict) -> dict:
    """Format a session's tool call trace as a timeline."""
    session_id = arguments.get("session_id", "").strip()
    if not session_id:
        return mcp_error("session_id is required for trace mode")

    try:
        from cairn_platform.orchestrator.coordination import CoordinationManager

        mgr = CoordinationManager.get_instance()
        rows = mgr.query_audit(session_id=session_id, limit=500)

        if not rows:
            return mcp_response(f"No trace data for session {session_id}")

        # Sort by call_index (ascending) if available, else by created_at
        rows.sort(key=lambda r: (r.get("call_index") or 0, r.get("created_at", "")))

        error_count = sum(1 for r in rows if r.get("result_status") == "error")
        total_latency = sum(r.get("latency_ms") or 0 for r in rows)

        lines = [f"Session {session_id[:12]} -- {len(rows)} tool calls, {total_latency/1000:.1f}s total, {error_count} errors\n"]

        for r in rows:
            idx = r.get("call_index") or "-"
            lat = f"{r.get('latency_ms') or 0}ms"
            tool = r.get("tool_name", "?")
            status = r.get("result_status") or "ok"
            size = r.get("input_size") or 0
            size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"

            lines.append(f" #{idx:<4} {lat:<8} {tool:<12} {status:<8} {size_str}")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_query (trace) failed: %s", e, exc_info=True)
        return mcp_error(f"Trace query failed: {e}")


# ============================================================================
# Handler: cairn_query mode=browse
# ============================================================================


async def handle_cairn_browse(arguments: dict) -> dict:
    """Browse memories by type, session, or most recent."""
    browse_by = arguments.get("browse_by", "recent")
    limit = _clamp_int(arguments.get("limit", 20), default=20, max_val=200)

    try:
        from cairn.bridge import _get_store

        db = _get_store()

        if browse_by == "type":
            event_type = arguments.get("event_type")
            if not event_type:
                return mcp_error("event_type is required when browse_by='type'")
            results = db.get_by_type(event_type, limit=limit)
            title = f"Memories of type '{event_type}'"
        elif browse_by == "session":
            session_id = _validate_session_id(arguments.get("session_id"))
            if not session_id:
                return mcp_error("session_id is required when browse_by='session'")
            results = db.get_by_session(session_id, limit=limit)
            title = f"Memories from session '{session_id[:16]}...'"
        else:  # recent
            results = db.get_recent(limit=limit)
            title = "Most recent memories"

        if not results:
            return mcp_response(f"# {title}\n\n*No memories found.*")

        output = f"# {title} ({len(results)} results)\n\n"
        for i, node in enumerate(results, 1):
            etype = (node.metadata or {}).get("event_type", "memory")
            preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
            created = node.created_at.isoformat()[:16] if node.created_at else ""
            output += f"## {i}. [{etype}] `{node.id}`\n"
            output += f"{preview}\n"
            output += f"*{created}*\n\n"

        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_browse failed: %s", e, exc_info=True)
        return mcp_error("Browse failed")


# ============================================================================
# Handler: cairn_welcome
# ============================================================================


async def handle_cairn_welcome(arguments: dict) -> dict:
    """Get a session welcome briefing with recent relevant memories."""
    session_id = _validate_session_id(arguments.get("session_id"))
    project = arguments.get("project")

    try:
        from cairn.server.hook_server import mark_protocol_call
        mark_protocol_call(session_id, "cairn_welcome")
    except Exception as e:
        logger.debug("mark_protocol_call (welcome) failed: %s", e)

    # Track session start for telemetry
    try:
        from cairn.telemetry import track_event
        track_event("session_start")
    except Exception:
        pass

    # Register this session in coordination — the MCP handler is the most
    # reliable registration path because it runs in-process (no subprocess
    # timeout, correct PID).  The coord_session_start hook often times out
    # under SQLite contention with many concurrent agents.
    try:
        from cairn_platform.orchestrator.coordination import get_manager
        import os as _os

        mgr = get_manager()
        # For stdio transport the MCP server is a child of the Claude process,
        # so getppid() gives the Claude PID.  For HTTP daemon mode, use own PID
        # as a fallback (the hook daemon will update it via heartbeat).
        from cairn.server.mcp_server import _TRANSPORT
        caller_pid = _os.getppid() if _TRANSPORT == "stdio" else _os.getpid()
        mgr.register_session(
            session_id=session_id,
            pid=caller_pid,
            project=project or _os.getcwd(),
            metadata={"client": "claude-code", "mcp_transport": _TRANSPORT},
        )
    except Exception as e:
        logger.debug("register_session in cairn_welcome failed: %s", e)

    try:
        from cairn.bridge import welcome

        briefing = welcome(session_id=session_id, project=project)

        # Format as readable markdown — stable content first, volatile after breakpoint
        stable_parts = []
        volatile_parts = []

        stable_parts.append(f"# Welcome Briefing ({briefing.get('memory_count', 0)} memories)\n")

        # Observation prefix already has internal cache breakpoint from bridge.py
        obs = briefing.get("observation_prefix", "")
        if obs:
            stable_parts.append(obs)

        # Project context is already markdown
        proj = briefing.get("project_context", "")
        if proj:
            stable_parts.append(proj)

        # Trending topics → volatile (changes weekly)
        topics = briefing.get("trending_topics", [])
        if topics:
            volatile_parts.append("### Trending Topics\n" + ", ".join(topics))

        # Flagged memories → volatile (changes per session)
        flagged = briefing.get("flagged_for_review", 0)
        if flagged:
            volatile_parts.append(f"**{flagged} memories flagged for review** -- use `cairn_memory(action='flagged')` to inspect")

        # Dedup stats → volatile
        dedup = briefing.get("duplicates_prevented", 0)
        if dedup:
            volatile_parts.append(f"*{dedup} duplicates prevented this session*")

        # Advisor suggestions → volatile
        suggestions = briefing.get("advisor_suggestions", "")
        if suggestions:
            volatile_parts.append("### Suggestions\n" + suggestions)

        # Nudge for underused tools — conditional on state
        nudges = []
        try:
            from cairn.bridge import get_profile
            profile = get_profile()
            if not profile or len(profile) <= 1:  # empty or just defaults
                nudges.append("`cairn_profile()` — load user working style preferences")
        except Exception:
            pass

        # Append mandatory next-step to drive protocol compliance
        next_steps = "**Next step**: Call `cairn_protocol()` for your operating rules before starting work."
        if nudges:
            next_steps += "\n**Also recommended**: " + " | ".join(nudges)
        stable_parts.append("---\n" + next_steps)

        # Join with cache breakpoint between stable and volatile
        parts = stable_parts
        if volatile_parts:
            parts = stable_parts + ["<!-- cairn:cache_breakpoint -->"] + volatile_parts

        return mcp_response("\n\n".join(parts))
    except Exception as e:
        logger.error("cairn_welcome failed: %s", e, exc_info=True)
        return mcp_error("Welcome briefing failed")


# ============================================================================
# Handler: cairn_profile
# ============================================================================


async def handle_cairn_profile(arguments: dict) -> dict:
    """Read or update the user profile, or list preferences.

    Actions: 'read' (default), 'update', 'list_preferences'.
    Also supports legacy mode: if 'update' dict provided without action, uses update mode.
    """
    action = arguments.get("action", "read")

    # list_preferences action
    if action == "list_preferences":
        return await handle_cairn_list_preferences(arguments)

    # Support legacy cairn_save_profile param name
    update_data = arguments.get("update") or arguments.get("profile")

    # If action is explicitly 'update' or update_data is provided
    if action == "update" or update_data:
        # Write mode
        try:
            from cairn.bridge import get_profile, save_profile

            existing = get_profile()
            existing.pop("preferences_from_memory", None)
            existing.update(update_data)
            success = save_profile(existing)
            if success:
                return mcp_response(f"Profile updated with {len(update_data)} field(s).")
            else:
                return mcp_error("Failed to save profile to disk.")
        except Exception as e:
            logger.error("cairn_profile (save) failed: %s", e, exc_info=True)
            return mcp_error("Save profile failed")
    else:
        # Read mode
        try:
            from cairn.bridge import get_profile
            from cairn import json_compat as json

            profile = get_profile()
            if not profile:
                return mcp_response("No profile found. Preferences will build your profile over time.")
            return mcp_response(json.dumps(profile, indent=2))
        except Exception as e:
            logger.error("cairn_profile failed: %s", e, exc_info=True)
            return mcp_error("Profile failed")


# ============================================================================
# Handler: cairn_delete_memory
# ============================================================================


async def handle_cairn_delete_memory(arguments: dict) -> dict:
    """Delete a specific memory by its ID."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    caller_session_id = arguments.get("caller_session_id", "").strip()
    force = arguments.get("force", False)

    try:
        from cairn.bridge import delete_memory, _get_store

        # Session ownership check: verify caller owns this memory
        if caller_session_id and not force:
            db = _get_store()
            node = db.get_node(memory_id)
            if node is not None:
                mem_session = (node.metadata or {}).get("session_id", "")
                if mem_session and mem_session != caller_session_id:
                    logger.warning(
                        "Delete blocked: caller %s tried to delete memory owned by session %s",
                        caller_session_id[:12], mem_session[:12],
                    )
                    return mcp_error(
                        f"Ownership check failed: memory belongs to session {mem_session[:12]}. "
                        "Use force=True to override."
                    )

        result = delete_memory(memory_id=memory_id)
        if result.get("success"):
            return mcp_response(f"Deleted memory `{memory_id[:16]}`")
        else:
            return mcp_error(result.get("error", f"Memory {memory_id} not found"))
    except Exception as e:
        logger.error("cairn_delete_memory failed: %s", e, exc_info=True)
        return mcp_error("Delete failed")


# ============================================================================
# Handler: cairn_edit_memory
# ============================================================================


async def handle_cairn_edit_memory(arguments: dict) -> dict:
    """Edit the content of a specific memory."""
    memory_id = arguments.get("memory_id", "").strip()
    new_content = arguments.get("new_content", "").strip()

    if not memory_id:
        return mcp_error("memory_id is required")
    if not new_content:
        return mcp_error("new_content is required")

    try:
        from cairn.bridge import edit_memory

        result = edit_memory(memory_id=memory_id, new_content=new_content)
        if result.get("success"):
            return mcp_response(f"Updated memory `{memory_id[:16]}`\nNew content: {new_content[:200]}")
        else:
            return mcp_error(result.get("error", f"Memory {memory_id} not found"))
    except Exception as e:
        logger.error("cairn_edit_memory failed: %s", e, exc_info=True)
        return mcp_error("Edit failed")


# ============================================================================
# Handler: cairn_list_preferences
# ============================================================================


async def handle_cairn_list_preferences(arguments: dict) -> dict:
    """List all stored user preferences."""
    try:
        from cairn.bridge import list_preferences

        prefs = list_preferences()

        if not prefs:
            return mcp_response("No preferences stored yet.")

        lines = [f"## User Preferences ({len(prefs)} total)\n"]
        for pref in prefs:
            content = pref.get("content", "")[:200]
            created = pref.get("created_at", "")[:16]
            pref_id = pref.get("id", "")[:12]
            lines.append(f"- {content}")
            lines.append(f"  _Created: {created} | id: {pref_id}_")
            lines.append("")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_list_preferences failed: %s", e, exc_info=True)
        return mcp_error("List preferences failed")


# ============================================================================
# Handler: cairn_health (includes former cairn_status stats)
# ============================================================================


async def handle_cairn_health(arguments: dict) -> dict:
    """Detailed health check with memory usage, warnings, and recommendations."""
    try:
        from cairn.bridge import check_health, status

        warn_mb = _clamp_int(arguments.get("warn_mb", 350), default=350, max_val=10000)
        critical_mb = _clamp_int(arguments.get("critical_mb", 800), default=800, max_val=10000)
        max_nodes = _clamp_int(arguments.get("max_nodes", 10000), default=10000, max_val=100000)
        result = check_health(warn_mb=warn_mb, critical_mb=critical_mb, max_nodes=max_nodes)

        # Append basic stats (formerly cairn_status)
        try:
            st = status()
            result += (
                f"Backend: {st.get('backend', 'sqlite')}"
                f" | Store: {st.get('store_path', '~/.cairn')}"
                f" | Vec: {st.get('vec_enabled', False)}\n"
            )
        except Exception as e:
            logger.debug("Health check stats failed: %s", e)

        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_health failed: %s", e, exc_info=True)
        return mcp_error("Health check failed")


# ============================================================================
# Handler: cairn_backup (merged export + import)
# ============================================================================


async def handle_cairn_backup(arguments: dict) -> dict:
    """Export or import memories (backup/restore)."""
    mode = arguments.get("mode", "export").strip()
    filepath = arguments.get("filepath", "").strip()
    if not filepath:
        return mcp_error("filepath is required")

    # Path validation: restrict to ~/.cairn/ to prevent sensitive file access.
    # Use os.path.realpath() for TOCTOU-safe symlink resolution.
    resolved = Path(os.path.realpath(Path(filepath).expanduser())).resolve()
    safe_dir = Path(os.path.realpath(_safe_export_dir())).resolve()
    if not str(resolved).startswith(str(safe_dir) + "/") and resolved.parent != safe_dir:
        return mcp_error(f"Path must be under {_safe_export_dir()}")

    if mode == "import":
        if not resolved.exists():
            return mcp_error("File not found")
        # TOCTOU re-validation: re-resolve right before read to catch symlink changes
        real_at_open = Path(os.path.realpath(resolved))
        if not str(real_at_open).startswith(str(safe_dir) + "/") and real_at_open.parent != safe_dir:
            return mcp_error("Path escapes safe directory after symlink resolution")
        clear_existing = arguments.get("clear_existing", True)
        try:
            from cairn.bridge import import_memories

            return await _run_or_submit_maintain(
                "restore",
                lambda: import_memories(filepath=str(real_at_open), clear_existing=clear_existing),
                arguments,
            )
        except Exception as e:
            logger.error("cairn_backup import failed: %s", e, exc_info=True)
            return mcp_error("Import failed (internal error)")
    else:
        # TOCTOU re-validation: re-resolve parent right before write
        real_parent = Path(os.path.realpath(resolved.parent))
        if not str(real_parent).startswith(str(safe_dir)) and real_parent != safe_dir:
            return mcp_error("Path escapes safe directory after symlink resolution")
        # Reject if target path is itself a symlink (prevent write-through-symlink)
        if resolved.is_symlink():
            return mcp_error("Export target must not be a symlink")
        try:
            from cairn.bridge import export_memories
            from cairn.crypto import is_enabled as crypto_enabled

            def _do_export() -> dict:
                result = export_memories(filepath=str(resolved))
                if crypto_enabled():
                    result["warning"] = (
                        "CAIRN_ENCRYPT is enabled but exports are plaintext. "
                        "The export file contains unencrypted memory content. "
                        "Store it securely or delete after use."
                    )
                return result

            return await _run_or_submit_maintain("backup", _do_export, arguments)
        except Exception as e:
            logger.error("cairn_backup export failed: %s", e, exc_info=True)
            return mcp_error("Export failed (internal error)")


# ============================================================================
# Handler: cairn_lessons (merged with cairn_cross_project_lessons)
# ============================================================================


async def handle_cairn_lessons(arguments: dict) -> dict:
    """Retrieve cross-session or cross-project lessons learned."""
    try:
        cross_project = arguments.get("cross_project", False)
        task = arguments.get("task")
        limit = _clamp_int(arguments.get("limit", 5), default=5, max_val=100)
        agent_type = arguments.get("agent_type")

        if cross_project:
            from cairn.bridge import get_cross_project_lessons

            exclude_project = arguments.get("exclude_project")
            exclude_session = arguments.get("exclude_session")
            lessons = get_cross_project_lessons(
                task=task,
                exclude_project=exclude_project,
                exclude_session=exclude_session,
                limit=limit,
                agent_type=agent_type,
            )
            if not lessons:
                return mcp_response("No cross-project lessons found.")

            output = f"Cross-Project Lessons ({len(lessons)})\n\n"
            for i, lesson in enumerate(lessons, 1):
                proj = lesson.get("source_project", "?")
                projects_seen = lesson.get("projects_seen", 1)
                xp_badge = f" [across {projects_seen} projects]" if projects_seen > 1 else ""
                output += f"{i}. {lesson['content'][:120]}\n"
                output += f"   src={proj}{xp_badge} accessed={lesson.get('access_count', 0)}\n\n"
            return mcp_response(output)
        else:
            from cairn.bridge import get_cross_session_lessons

            project_path = arguments.get("project_path")
            lessons = get_cross_session_lessons(
                task=task,
                project_path=project_path,
                limit=limit,
                agent_type=agent_type,
            )
            if not lessons:
                return mcp_response("No cross-session lessons found yet.")

            output = f"Cross-Session Lessons ({len(lessons)})\n\n"
            for i, lesson in enumerate(lessons, 1):
                verified_count = lesson.get("verified_count", 0)
                if verified_count >= 3:
                    badge = f" [verified x{verified_count}]"
                elif verified_count > 0:
                    badge = f" [seen in {verified_count} sessions]"
                else:
                    badge = ""
                access = lesson.get("access_count", 0)
                output += f"{i}. {lesson.get('content', '')[:200]}{badge}\n"
                output += f"   accessed={access}\n\n"
            return mcp_response(output)
    except Exception as e:
        logger.error("cairn_lessons failed: %s", e, exc_info=True)
        return mcp_error("Lessons failed")




# ============================================================================
# Handler: cairn_feedback
# ============================================================================


async def handle_cairn_feedback(arguments: dict) -> dict:
    """Record feedback on a surfaced memory."""
    memory_id = arguments.get("memory_id", "").strip()
    rating = arguments.get("rating", "").strip()
    reason = arguments.get("reason")

    if not memory_id:
        return mcp_error("memory_id is required")
    if rating not in ("helpful", "unhelpful", "outdated"):
        return mcp_error("rating must be one of: helpful, unhelpful, outdated")

    try:
        from cairn.bridge import record_feedback

        result = record_feedback(memory_id=memory_id, rating=rating, reason=reason)
        if "error" in result:
            return mcp_error(result["error"])
        return mcp_response(
            f"Feedback recorded: {rating} for `{memory_id[:16]}`\n"
            f"New score: {result.get('new_score', 0)} "
            f"({result.get('total_signals', 0)} total signals)"
        )
    except Exception as e:
        logger.error("cairn_feedback failed: %s", e, exc_info=True)
        return mcp_error("Feedback failed")


# ============================================================================
# Handler: cairn_clear_session
# ============================================================================


async def handle_cairn_clear_session(arguments: dict) -> dict:
    """Clear all memories for a session."""
    session_id = arguments.get("session_id", "").strip()
    if not session_id:
        return mcp_error("session_id is required")

    caller_session_id = arguments.get("caller_session_id", "").strip()
    force = arguments.get("force", False)

    # Session ownership check: only allow clearing your own session
    if caller_session_id and not force and session_id != caller_session_id:
        logger.warning(
            "Clear session blocked: caller %s tried to clear session %s",
            caller_session_id[:12], session_id[:12],
        )
        return mcp_error(
            f"Ownership check failed: cannot clear session {session_id[:12]} "
            f"from session {caller_session_id[:12]}. Use force=True to override."
        )

    try:
        from cairn.bridge import clear_session

        result = clear_session(session_id=session_id)
        return mcp_response(f"Cleared session `{session_id[:16]}`: {result.get('removed', 0)} memories removed.")
    except Exception as e:
        logger.error("cairn_clear_session failed: %s", e, exc_info=True)
        return mcp_error("Clear session failed")


# ============================================================================
# Handler: cairn_consolidate
# ============================================================================


async def handle_cairn_consolidate(arguments: dict) -> dict:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges."""
    prune_days = _clamp_int(arguments.get("prune_days", 14), default=14, max_val=365)
    max_summaries = _clamp_int(arguments.get("max_summaries", 50), default=50, max_val=1000)

    try:
        from cairn.bridge import consolidate

        return await _run_or_submit_maintain(
            "consolidate",
            lambda: consolidate(prune_days=prune_days, max_summaries=max_summaries),
            arguments,
        )
    except Exception as e:
        logger.error("cairn_consolidate failed: %s", e, exc_info=True)
        return mcp_error("Consolidation failed")


# ============================================================================
# Handler: cairn_similar
# ============================================================================


async def handle_cairn_get_memory(arguments: dict) -> dict:
    """Fetch one memory by full id or unique prefix — full untruncated content."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")
    if len(memory_id) < 4:
        return mcp_error("memory_id prefix too short — use at least 8 characters")

    include_related = arguments.get("include_related", True)

    try:
        from cairn.bridge import get_memory

        result = get_memory(memory_id=memory_id, include_related=bool(include_related))
        if "ambiguous" in result:
            ids = ", ".join(result["ambiguous"])
            return mcp_error(
                f"Ambiguous prefix '{memory_id}' matches {len(result['ambiguous'])} "
                f"memories: {ids} — use a longer prefix"
            )
        if "error" in result:
            return mcp_error(result["error"])
        return mcp_response(json.dumps(result, indent=2, default=str))
    except Exception as e:
        logger.error("cairn_get failed: %s", e, exc_info=True)
        return mcp_error("Get memory failed")


async def handle_cairn_similar(arguments: dict) -> dict:
    """Find memories similar to a given memory."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    limit = _clamp_int(arguments.get("limit", 5), default=5, max_val=100)

    try:
        from cairn.bridge import find_similar_memories

        result = find_similar_memories(memory_id=memory_id, limit=limit)
        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_similar failed: %s", e, exc_info=True)
        return mcp_error("Similar search failed")


# ============================================================================
# Handler: cairn_timeline
# ============================================================================


async def handle_cairn_timeline(arguments: dict) -> dict:
    """Show memory timeline grouped by day."""
    days = _clamp_int(arguments.get("days", 7), default=7, min_val=0, max_val=365)
    limit_per_day = _clamp_int(arguments.get("limit_per_day", 10), default=10, max_val=100)

    try:
        from cairn.bridge import timeline

        result = timeline(days=days, limit_per_day=limit_per_day)
        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_timeline failed: %s", e, exc_info=True)
        return mcp_error("Timeline failed")


# ============================================================================
# Handler: cairn_traverse
# ============================================================================


async def handle_cairn_traverse(arguments: dict) -> dict:
    """Traverse the memory relationship graph from a starting memory."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return mcp_error("memory_id is required")

    max_hops = arguments.get("max_hops", 2)
    min_weight = arguments.get("min_weight", 0.0)
    edge_types = arguments.get("edge_types")

    try:
        from cairn.bridge import traverse

        result = traverse(
            memory_id=memory_id,
            max_hops=max_hops,
            min_weight=min_weight,
            edge_types=edge_types,
        )
        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_traverse failed: %s", e, exc_info=True)
        return mcp_error("Traverse failed")


# ============================================================================
# Handler: cairn_compact
# ============================================================================


async def handle_cairn_compact(arguments: dict) -> dict:
    """Compact related memories into consolidated knowledge nodes."""
    event_type = arguments.get("event_type", "lesson_learned")
    similarity_threshold = arguments.get("similarity_threshold", 0.6)
    min_cluster_size = _clamp_int(arguments.get("min_cluster_size", 3), default=3, min_val=2, max_val=100)
    dry_run = arguments.get("dry_run", False)

    try:
        from cairn.bridge import compact

        return await _run_or_submit_maintain(
            "compact",
            lambda: compact(
                event_type=event_type,
                similarity_threshold=similarity_threshold,
                min_cluster_size=min_cluster_size,
                dry_run=dry_run,
            ),
            arguments,
        )
    except Exception as e:
        logger.error("cairn_compact failed: %s", e, exc_info=True)
        return mcp_error("Compact failed")




# ============================================================================
# Handler: cairn_forgetting_log
# ============================================================================


async def handle_cairn_forgetting_log(arguments: dict) -> dict:
    """Retrieve the forgetting audit log."""
    limit = _clamp_int(arguments.get("limit", 50), default=50, min_val=1, max_val=500)
    reason = arguments.get("reason")

    try:
        from cairn.bridge import forgetting_log

        result = forgetting_log(limit=limit, reason=reason)
        return mcp_response(result)
    except Exception as e:
        logger.error("cairn_forgetting_log failed: %s", e, exc_info=True)
        return mcp_error("Failed to retrieve forgetting log")


# ============================================================================
# Handler: cairn_type_stats
# ============================================================================


async def handle_cairn_type_stats(arguments: dict) -> dict:
    """Get memory counts grouped by event type."""
    try:
        from cairn.bridge import type_stats

        stats = type_stats()
        if not stats:
            return mcp_response("No memories stored yet.")

        total = sum(stats.values())
        lines = [f"# Memory Type Stats ({total} total)\n"]
        for etype, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total * 100) if total > 0 else 0
            lines.append(f"- **{etype}**: {count} ({pct:.1f}%)")
        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_type_stats failed: %s", e, exc_info=True)
        return mcp_error("Type stats failed")


# ============================================================================
# Handler: cairn_session_stats
# ============================================================================


async def handle_cairn_session_stats(arguments: dict) -> dict:
    """Get memory counts grouped by session ID."""
    try:
        from cairn.bridge import session_stats

        stats = session_stats()
        if not stats:
            return mcp_response("No session data found.")

        # Sort by count descending, show top 20
        sorted_sessions = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:20]
        total = sum(stats.values())
        lines = [f"# Session Stats (top {len(sorted_sessions)} of {len(stats)} sessions, {total} total memories)\n"]
        for sid, count in sorted_sessions:
            truncated = sid[:16] + "..." if len(sid) > 16 else sid
            lines.append(f"- `{truncated}`: {count} memories")
        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_session_stats failed: %s", e, exc_info=True)
        return mcp_error("Session stats failed")


# ============================================================================
# Handler: cairn_weekly_digest
# ============================================================================


async def handle_cairn_weekly_digest(arguments: dict) -> dict:
    """Generate a weekly knowledge digest with stats, trends, and highlights."""
    try:
        from cairn.bridge import get_weekly_digest

        days = arguments.get("days", 7)
        digest = get_weekly_digest(days=days)

        lines = [
            f"Week ({digest['period_days']}d): {digest['period_new']} new"
            f" | {digest['session_count']} sessions"
            f" | {digest['total_memories']} total"
        ]

        # Growth
        if digest["prev_period_count"] > 0:
            direction = "+" if digest["growth_pct"] > 0 else ""
            lines.append(
                f"Growth: {direction}{digest['growth_pct']}%"
                f" ({digest['prev_period_count']}->{digest['period_new']})"
            )

        # Type breakdown
        if digest["type_breakdown"]:
            breakdown = ", ".join(
                f"{etype}: {count}"
                for etype, count in sorted(digest["type_breakdown"].items(), key=lambda x: x[1], reverse=True)
                if count > 0 and etype != "session_summary"
            )
            if breakdown:
                lines.append(f"Types: {breakdown}")

        # Top topics
        if digest["top_topics"]:
            lines.append(f"Topics: {', '.join(digest['top_topics'][:6])}")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_weekly_digest failed: %s", e, exc_info=True)
        return mcp_error("Weekly digest failed")


# ============================================================================
# Handler: cairn_checkpoint
# ============================================================================


async def handle_cairn_checkpoint(arguments: dict) -> dict:
    """Save a task checkpoint for session continuity."""
    task_title = arguments.get("task_title", "").strip()
    progress = arguments.get("progress", "").strip()
    if not task_title or not progress:
        return mcp_error("task_title and progress are required")

    # Build structured checkpoint content
    checkpoint = {
        "version": 1,
        "task_title": task_title,
        "plan": arguments.get("plan", ""),
        "progress": progress,
        "files_touched": arguments.get("files_touched", {}),
        "decisions": arguments.get("decisions", []),
        "key_context": arguments.get("key_context", ""),
        "next_steps": arguments.get("next_steps", ""),
    }

    # Format as searchable text content
    content_lines = [f"## Checkpoint: {task_title}"]
    if checkpoint["plan"]:
        content_lines.append(f"\n### Plan\n{checkpoint['plan']}")
    content_lines.append(f"\n### Progress\n{checkpoint['progress']}")
    if checkpoint["files_touched"]:
        content_lines.append("\n### Files Changed")
        for fp, summary in checkpoint["files_touched"].items():
            content_lines.append(f"- `{fp}`: {summary}")
    if checkpoint["decisions"]:
        content_lines.append("\n### Decisions")
        for d in checkpoint["decisions"]:
            content_lines.append(f"- {d}")
    if checkpoint["key_context"]:
        content_lines.append(f"\n### Key Context\n{checkpoint['key_context']}")
    if checkpoint["next_steps"]:
        content_lines.append(f"\n### Next Steps\n{checkpoint['next_steps']}")

    content = "\n".join(content_lines)

    # Determine checkpoint number for this task
    session_id = _validate_session_id(arguments.get("session_id"))
    project = arguments.get("project")
    checkpoint_num = 1
    try:
        from cairn.bridge import query_structured

        existing = query_structured(
            query_text=f"checkpoint {task_title}",
            limit=10,
            event_type="checkpoint",
        )
        if project:
            existing = [e for e in existing if (e.get("metadata") or {}).get("project") == project]
        checkpoint_num = len(existing) + 1
    except Exception as e:
        logger.debug("Checkpoint numbering failed: %s", e)

    metadata = {
        "checkpoint_number": checkpoint_num,
        "checkpoint_data": checkpoint,
    }

    try:
        from cairn.bridge import auto_capture

        result = auto_capture(
            content=content,
            event_type="checkpoint",
            metadata=metadata,
            session_id=session_id,
            project=project,
        )
        return mcp_response(f"{result}\n\nCheckpoint #{checkpoint_num} saved for: {task_title}")
    except Exception as e:
        logger.error("cairn_checkpoint failed: %s", e, exc_info=True)
        return mcp_error(f"Checkpoint failed: {e}")


# ============================================================================
# Handler: cairn_resume_task
# ============================================================================


async def handle_cairn_resume_task(arguments: dict) -> dict:
    """Resume a checkpointed task with full context."""
    task_title = arguments.get("task_title", "").strip()
    project = arguments.get("project")
    verbosity = arguments.get("verbosity", "full")
    limit = _clamp_int(arguments.get("limit"), 1, 1, 5)

    # Build search query
    query_text = f"checkpoint {task_title}" if task_title else "checkpoint"

    try:
        from cairn.bridge import query_structured

        results = query_structured(
            query_text=query_text,
            limit=limit * 3,  # Over-fetch for filtering
            event_type="checkpoint",
        )

        if not results:
            return mcp_response("No checkpoints found. Start fresh or provide a different task title.")

        # Post-filter by project if specified (metadata match, not query dilution)
        if project:
            filtered = [r for r in results if (r.get("metadata") or {}).get("project") == project]
            if filtered:
                results = filtered

        # Take the most recent checkpoints (by created_at)
        results = sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)[:limit]

        lines = [f"# Task Resume — {len(results)} checkpoint(s) found\n"]

        for r in results:
            meta = r.get("metadata", {})
            checkpoint_data = meta.get("checkpoint_data", {})
            cp_num = meta.get("checkpoint_number", "?")
            created = r.get("created_at", "unknown")[:16]

            if verbosity == "minimal":
                next_steps = checkpoint_data.get("next_steps", "No next steps recorded")
                lines.append(f"## Checkpoint #{cp_num} ({created})")
                lines.append(f"**Task**: {checkpoint_data.get('task_title', 'Unknown')}")
                lines.append(f"**Next Steps**: {next_steps}\n")
            elif verbosity == "summary":
                lines.append(f"## Checkpoint #{cp_num} ({created})")
                lines.append(f"**Task**: {checkpoint_data.get('task_title', 'Unknown')}")
                if checkpoint_data.get("plan"):
                    lines.append(f"**Plan**: {checkpoint_data['plan']}")
                lines.append(f"**Progress**: {checkpoint_data.get('progress', 'Unknown')}")
                lines.append(f"**Next Steps**: {checkpoint_data.get('next_steps', 'None')}\n")
            else:  # full
                lines.append(r.get("content", "No content"))
                if checkpoint_data.get("files_touched") and "Files Changed" not in r.get("content", ""):
                    lines.append("\n### Files Changed")
                    for fp, summary in checkpoint_data["files_touched"].items():
                        lines.append(f"- `{fp}`: {summary}")
                lines.append("")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_resume_task failed: %s", e, exc_info=True)
        return mcp_error(f"Resume failed: {e}")


# ============================================================================
# Handler: cairn_remind
# ============================================================================


async def handle_cairn_remind(arguments: dict) -> dict:
    """Create a time-based reminder."""
    text = arguments.get("text", "").strip()
    duration = arguments.get("duration", "").strip()
    if not text:
        return mcp_error("text is required")
    if not duration:
        return mcp_error("duration is required (e.g. '1h', '30m', '2d')")

    context = arguments.get("context")
    session_id = _validate_session_id(arguments.get("session_id"))
    project = arguments.get("project")

    try:
        from cairn.bridge import create_reminder

        result = create_reminder(
            text=text,
            duration=duration,
            context=context,
            session_id=session_id,
            project=project,
        )
        lines = [
            f"Reminder set: {result['text']}",
            f"Due at: {result['remind_at_local']}",
            f"ID: {result['reminder_id']}",
        ]
        return mcp_response("\n".join(lines))
    except ValueError as e:
        return mcp_error(str(e))
    except Exception as e:
        logger.error("cairn_remind failed: %s", e, exc_info=True)
        return mcp_error(f"Failed to create reminder: {e}")


# ============================================================================
# Handler: cairn_remind_list
# ============================================================================


async def handle_cairn_remind_list(arguments: dict) -> dict:
    """List reminders with status and due times."""
    status = arguments.get("status")
    entity_id = _validate_entity_id(arguments.get("entity_id"))

    try:
        from cairn.bridge import list_reminders

        include_dismissed = status in ("dismissed", "all")
        reminders = list_reminders(status=status, include_dismissed=include_dismissed, entity_id=entity_id)

        if not reminders:
            return mcp_response("No reminders found.")

        lines = [f"**Reminders** ({len(reminders)} found)\n"]
        status_icons = {"pending": "⏳", "fired": "🔔", "dismissed": "✓"}
        for r in reminders:
            icon = status_icons.get(r["status"], "?")
            overdue = " **[OVERDUE]**" if r.get("is_overdue") else ""
            lines.append(f"- {icon} {r['text']}{overdue}")
            lines.append(f"  Due: {r['remind_at_local']} | Status: {r['status']} | Time: {r['time_until']}")
            if r.get("context"):
                lines.append(f"  Context: {r['context'][:120]}")
            lines.append(f"  ID: {r['id']}")

        return mcp_response("\n".join(lines))
    except Exception as e:
        logger.error("cairn_remind_list failed: %s", e, exc_info=True)
        return mcp_error(f"Failed to list reminders: {e}")


# ============================================================================
# Handler: cairn_remind_dismiss
# ============================================================================


async def handle_cairn_remind_dismiss(arguments: dict) -> dict:
    """Dismiss a reminder by ID."""
    reminder_id = arguments.get("reminder_id", "").strip()
    if not reminder_id:
        return mcp_error("reminder_id is required")

    try:
        from cairn.bridge import dismiss_reminder

        result = dismiss_reminder(reminder_id)
        if result.get("success"):
            return mcp_response(f"Dismissed reminder: {result.get('text', reminder_id)}")
        return mcp_error(result.get("error", "Failed to dismiss reminder"))
    except Exception as e:
        logger.error("cairn_remind_dismiss failed: %s", e, exc_info=True)
        return mcp_error(f"Failed to dismiss reminder: {e}")


# ============================================================================
# Handler: cairn_protocol
# ============================================================================


async def handle_cairn_protocol(arguments: dict) -> dict:
    """Serve the coordination playbook dynamically based on context."""
    section = arguments.get("section")
    project = arguments.get("project")

    try:
        from cairn.server.hook_server import mark_protocol_call
        session_id_for_mark = arguments.get("session_id") or os.environ.get("SESSION_ID", "")
        mark_protocol_call(session_id_for_mark, "cairn_protocol")
    except Exception as e:
        logger.debug("mark_protocol_call (protocol) failed: %s", e)

    # Special section: gate_status returns protocol gate diagnostic info
    if section == "gate_status":
        try:
            from cairn.server.hook_server import (
                _gate_call_count,
                _heartbeat_count,
                _protocol_calls,
                _session_peer_count,
                _session_peer_count_time,
            )
            import time as _time

            sid = arguments.get("session_id") or os.environ.get("SESSION_ID", "")
            now = _time.monotonic()
            peer_age = now - _session_peer_count_time.get(sid, 0) if sid in _session_peer_count_time else None
            info = {
                "session_id": sid,
                "gate_call_count": _gate_call_count.get(sid, 0),
                "heartbeat_count": _heartbeat_count.get(sid, 0),
                "protocol_calls": sorted(_protocol_calls.get(sid, set())),
                "cached_peer_count": _session_peer_count.get(sid, "not set"),
                "peer_count_age_s": round(peer_age, 1) if peer_age is not None else "not set",
                "enforcement_window": "closed" if _gate_call_count.get(sid, 0) > 8 else "open",
            }
            lines = ["## Protocol Gate Status"]
            for k, v in info.items():
                lines.append(f"- **{k}**: {v}")
            return mcp_response("\n".join(lines))
        except Exception as e:
            return mcp_error(f"Gate status failed: {e}")

    # Detect peer count for auto-mode selection
    peer_count = 0
    try:
        from cairn_platform.orchestrator.coordination import get_manager

        mgr = get_manager()
        sessions = mgr.list_sessions(auto_clean=True)
        # Exclude self — count only other active peers
        peer_count = max(0, len(sessions) - 1)
    except Exception as e:
        logger.debug("Coordination session list failed: %s", e)

    try:
        from cairn.protocol import get_protocol

        result = get_protocol(
            section=section,
            project=project,
            include_lessons=True,
            peer_count=peer_count,
            session_id=session_id_for_mark or None,
        )

        # Mark protocol as loaded for this session (hooks check this marker)
        try:
            session_id = os.environ.get("SESSION_ID", "")
            if session_id:
                marker = _gate_dir().parent / f"session-{session_id}.protocol"
                marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                marker.write_text("loaded")
        except Exception as e:
            logger.debug("Protocol marker write failed: %s", e)

        return mcp_response(result)
    except ImportError:
        # Protocol engine module not present in this build — return the basic
        # operating rules.
        basic_protocol = (
            "# Cairn Protocol\n\n"
            "## Memory Usage\n"
            "- Call `cairn_store()` after completing tasks to save key decisions\n"
            "- Call `cairn_query()` before non-trivial tasks to check for prior context\n"
            "- Use `cairn_checkpoint` when context window is getting full\n\n"
            "## Session Workflow\n"
            "1. `cairn_welcome()` at session start (done)\n"
            "2. `cairn_query()` before major work\n"
            "3. `cairn_store()` after decisions and task completion\n"
        )

        # Mark protocol as loaded
        try:
            session_id = os.environ.get("SESSION_ID", "")
            if session_id:
                marker = _gate_dir().parent / f"session-{session_id}.protocol"
                marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                marker.write_text("loaded")
        except Exception:
            pass

        return mcp_response(basic_protocol)
    except Exception as e:
        logger.error("cairn_protocol failed: %s", e, exc_info=True)
        return mcp_error(f"Protocol failed: {e}")


async def handle_cairn_briefing(arguments: dict) -> dict:
    """Combined welcome + protocol in a single call (saves round-trips)."""
    project = arguments.get("project")
    session_id = arguments.get("session_id")

    parts = []

    # 1. Welcome briefing
    try:
        from cairn.bridge import welcome
        from cairn import json_compat as json

        briefing = welcome(session_id=session_id, project=project)
        parts.append("# Welcome Briefing\n\n" + json.dumps(briefing, indent=2))
    except Exception as e:
        logger.error("cairn_briefing: welcome failed: %s", e, exc_info=True)
        parts.append(f"# Welcome Briefing\n\n(Failed: {e})")

    # 2. Protocol (solo mode — Desktop is always solo)
    try:
        from cairn.protocol import get_protocol

        result = get_protocol(
            section="solo",
            project=project,
            include_lessons=True,
            peer_count=0,
        )
        parts.append(result)

        # Mark protocol as loaded
        try:
            sid = session_id or os.environ.get("SESSION_ID", "")
            if sid:
                marker = _gate_dir().parent / f"session-{sid}.protocol"
                marker.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                marker.write_text("loaded")
        except Exception as e:
            logger.debug("Briefing protocol marker write failed: %s", e)
    except Exception as e:
        logger.error("cairn_briefing: protocol failed: %s", e, exc_info=True)
        parts.append(f"# Protocol\n\n(Failed: {e})")

    return mcp_response("\n\n---\n\n".join(parts))


# ============================================================================
# Handler: cairn_link — Create edge between two memories
# ============================================================================


async def handle_cairn_link(arguments: dict) -> dict:
    """Manually create a relationship edge between two memories."""
    memory_id = arguments.get("memory_id", "").strip()
    target_id = arguments.get("target_id", "").strip()
    if not memory_id or not target_id:
        return mcp_error("memory_id and target_id are required")

    edge_type = arguments.get("edge_type", "related")
    weight = arguments.get("weight", 1.0)
    try:
        weight = max(0.0, min(1.0, float(weight)))
    except (TypeError, ValueError):
        weight = 1.0

    try:
        from cairn.bridge import _get_store

        db = _get_store()
        # Verify both memories exist
        source = db.get_node(memory_id)
        target = db.get_node(target_id)
        if source is None:
            return mcp_error(f"Source memory `{memory_id}` not found")
        if target is None:
            return mcp_error(f"Target memory `{target_id}` not found")

        success = db.add_edge(memory_id, target_id, edge_type=edge_type, weight=weight)
        if success:
            return mcp_response(
                f"Linked `{memory_id[:12]}` -> `{target_id[:12]}` (type: {edge_type}, weight: {weight:.2f})\n"
                f"Source: {source.content[:80]}\n"
                f"Target: {target.content[:80]}"
            )
        return mcp_error("Failed to create edge")
    except Exception as e:
        logger.error("cairn_link failed: %s", e, exc_info=True)
        return mcp_error(f"Link failed: {e}")


# ============================================================================
# Handler: cairn_flagged — List memories flagged for review
# ============================================================================


async def handle_cairn_flagged(arguments: dict) -> dict:
    """List memories that have been flagged for review (negative feedback score)."""
    limit = _clamp_int(arguments.get("limit", 20), default=20, max_val=200)

    try:
        from cairn.bridge import _get_store

        db = _get_store()
        # Query for flagged memories: feedback_score <= -3
        rows = db._conn.execute(
            """SELECT node_id, content, metadata, created_at,
                      access_count, last_accessed, ttl_seconds
               FROM memories
               WHERE json_extract(metadata, '$.flagged_for_review') = 1
                  OR json_extract(metadata, '$.feedback_score') <= -3
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

        if not rows:
            return mcp_response("# Flagged Memories\n\n*No memories flagged for review.* All clear.")

        results = [db._row_to_result(row) for row in rows]
        output = f"# Flagged Memories ({len(results)} need review)\n\n"
        for i, node in enumerate(results, 1):
            meta = node.metadata or {}
            etype = meta.get("event_type", "memory")
            score = meta.get("feedback_score", 0)
            preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
            output += f"## {i}. [{etype}] `{node.id}` (score: {score})\n"
            output += f"{preview}\n"
            created = node.created_at.isoformat()[:16] if node.created_at else ""
            output += f"*{created}* | Use `cairn_memory action='delete' memory_id='{node.id}'` to remove\n\n"

        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_flagged failed: %s", e, exc_info=True)
        return mcp_error("Flagged query failed")


# ============================================================================
# Handler: cairn_check_contradictions — Preview contradictions before storing
# ============================================================================


async def handle_cairn_check_contradictions(arguments: dict) -> dict:
    """Check new content against existing memories for contradictions."""
    new_content = arguments.get("new_content", "").strip()
    if not new_content:
        return mcp_error("new_content is required")

    limit = _clamp_int(arguments.get("limit", 10), default=10, max_val=50)

    try:
        from cairn.bridge import _get_store

        db = _get_store()

        # Find similar existing memories to check against
        candidates = db.query(new_content, limit=limit)
        if not candidates:
            return mcp_response("# Contradiction Check\n\n*No similar memories found.* Safe to store.")

        candidate_contents = [c.content for c in candidates]
        candidate_ids = [c.id for c in candidates]

        try:
            from cairn.contradictions import detect_contradictions

            results = detect_contradictions(
                new_content=new_content,
                candidates=candidate_contents,
            )
        except ImportError:
            return mcp_response("# Contradiction Check\n\n*Contradiction detection module not available.*")

        if not results:
            return mcp_response(
                f"# Contradiction Check\n\n*No contradictions found* among {len(candidates)} similar memories. Safe to store."
            )

        output = f"# Contradiction Check ({len(results)} potential conflicts)\n\n"
        output += f"**New content:** {new_content[:200]}\n\n"
        for i, cr in enumerate(results, 1):
            idx = cr.candidate_index
            mem_id = candidate_ids[idx] if idx < len(candidate_ids) else "?"
            output += f"## {i}. Conflict with `{mem_id[:12]}` (confidence: {cr.confidence:.0%})\n"
            output += f"**Signals:** {', '.join(cr.signals)}\n"
            output += f"**Existing:** {cr.candidate_text[:200]}\n"
            output += f"**Explanation:** {cr.explanation}\n\n"

        output += "*Review conflicts before storing. Use `cairn_store` to proceed anyway.*\n"
        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_check_contradictions failed: %s", e, exc_info=True)
        return mcp_error(f"Contradiction check failed: {e}")


# ============================================================================
# Handler: cairn_dedup_stats — Deduplication statistics
# ============================================================================


async def handle_cairn_dedup_stats(arguments: dict) -> dict:
    """Show how many duplicate memories Cairn has prevented."""
    try:
        from cairn.bridge import get_dedup_stats

        stats = get_dedup_stats()
        total_prevented = stats.get("content_dedup_skips", 0) + stats.get("embedding_dedup_skips", 0)
        output = "# Deduplication Stats\n\n"
        output += f"- **Duplicates prevented:** {total_prevented}\n"
        output += f"  - Content-level dedup: {stats.get('content_dedup_skips', 0)}\n"
        output += f"  - Embedding-level dedup: {stats.get('embedding_dedup_skips', 0)}\n"
        output += f"- **Memory evolutions:** {stats.get('memory_evolutions', 0)} (updated existing instead of duplicating)\n"
        output += f"- **Total memories:** {stats.get('node_count', 0)}\n"
        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_dedup_stats failed: %s", e, exc_info=True)
        return mcp_error("Dedup stats failed")


# ============================================================================
# Handler: cairn_supersede_memory — Manually mark a memory as superseded
# ============================================================================


async def handle_cairn_supersede_memory(arguments: dict) -> dict:
    """Manually mark a memory as superseded."""
    target_id = arguments.get("target_id", "").strip()
    if not target_id:
        # Fall back to memory_id for convenience
        target_id = arguments.get("memory_id", "").strip()
    if not target_id:
        return mcp_error("target_id is required for action='supersede'")

    reason = arguments.get("reason", "").strip() or "manual supersession"

    try:
        from cairn.bridge import _get_store

        db = _get_store()
        node = db.get_node(target_id)
        if node is None:
            return mcp_error(f"Memory `{target_id}` not found")

        if (node.metadata or {}).get("superseded"):
            superseded_by = (node.metadata or {}).get("superseded_by", "unknown")
            return mcp_response(
                f"Memory `{target_id[:16]}` is already superseded (by `{superseded_by}`)."
            )

        db.mark_superseded(target_id, superseded_by=f"manual: {reason}")
        snippet = (node.content or "")[:80]
        return mcp_response(
            f"Superseded memory `{target_id[:16]}`\n"
            f"Content: {snippet}{'...' if len(node.content or '') > 80 else ''}\n"
            f"Reason: {reason}"
        )
    except Exception as e:
        logger.error("cairn_supersede_memory failed: %s", e, exc_info=True)
        return mcp_error(f"Supersede failed: {e}")


# ============================================================================
# Composite Handler: cairn_memory (edit, delete, feedback, similar, traverse, link, flagged, check_contradictions, supersede)
# ============================================================================


async def handle_cairn_memory(arguments: dict) -> dict:
    """Route cairn_memory actions to existing handlers."""
    action = arguments.get("action", "").strip()

    if action == "get":
        return await handle_cairn_get_memory(arguments)
    elif action == "edit":
        return await handle_cairn_edit_memory(arguments)
    elif action == "delete":
        return await handle_cairn_delete_memory(arguments)
    elif action == "feedback":
        return await handle_cairn_feedback(arguments)
    elif action == "similar":
        return await handle_cairn_similar(arguments)
    elif action == "traverse":
        return await handle_cairn_traverse(arguments)
    elif action == "link":
        return await handle_cairn_link(arguments)
    elif action == "flagged":
        return await handle_cairn_flagged(arguments)
    elif action == "check_contradictions":
        return await handle_cairn_check_contradictions(arguments)
    elif action == "supersede":
        return await handle_cairn_supersede_memory(arguments)
    else:
        return mcp_error(f"Unknown cairn_memory action: {action}. Use: get, edit, delete, feedback, similar, traverse, link, flagged, check_contradictions, supersede")


# ============================================================================
# Composite Handler: cairn_remind (set, list, dismiss)
# ============================================================================


async def handle_cairn_remind_composite(arguments: dict) -> dict:
    """Route cairn_remind actions to existing handlers."""
    action = arguments.get("action", "set").strip()

    if action == "set":
        return await handle_cairn_remind(arguments)
    elif action == "list":
        return await handle_cairn_remind_list(arguments)
    elif action == "dismiss":
        return await handle_cairn_remind_dismiss(arguments)
    else:
        return mcp_error(f"Unknown cairn_remind action: {action}. Use: set, list, dismiss")


# ============================================================================
# Composite Handler: cairn_maintain (health, consolidate, compact, backup, restore, clear_session)
# ============================================================================


def _format_job_payload(job, action: str) -> str:
    """Render a Job dict as readable text for MCP response."""
    lines = [
        f"Job submitted: {job.id}",
        f"Action: {action}",
        f"Status: {job.status}",
        f"Poll with: cairn_maintain action=job_status job_id={job.id}",
    ]
    return "\n".join(lines)


def _format_job_status(job) -> str:
    """Render a Job's current state as readable text."""
    lines = [f"Job {job.id} ({job.name})", f"Status: {job.status}"]
    if job.started_at is not None:
        lines.append(f"Started: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(job.started_at))}")
    if job.finished_at is not None:
        elapsed = round(job.finished_at - (job.started_at or job.submitted_at), 3)
        lines.append(f"Finished: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(job.finished_at))}")
        lines.append(f"Elapsed: {elapsed}s")
    if job.status == "succeeded":
        lines.append("")
        lines.append("Result:")
        lines.append(str(job.result))
    elif job.status == "failed":
        lines.append("")
        lines.append(f"Error: {job.error}")
    return "\n".join(lines)


async def _run_or_submit_maintain(
    action_name: str,
    fn,
    arguments: dict,
) -> dict:
    """Run a synchronous maintenance callable.

    Heavy maintenance ops can exceed the MCP client's RPC timeout (~4 min) and
    cause "Server disconnected" errors. They also block the asyncio event loop
    if awaited directly. Both problems are avoided by routing through the
    shared SQLite executor.

    Modes:
    - wait=False (default): submit as a background Job, return job_id immediately.
      Poll with action=job_status.
    - wait=True: block on the executor and return the full result. Useful for
      tests, CLI bridges, and short ops.
    """
    import asyncio

    wait = bool(arguments.get("wait", False))
    from cairn.server.mcp_server import _SQLITE_EXECUTOR

    if wait:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(_SQLITE_EXECUTOR, fn)
        return mcp_response(result)

    from cairn.server.jobs import get_registry

    job = get_registry().submit(action_name, fn)
    return mcp_response(_format_job_payload(job, action_name))


async def handle_cairn_maintain(arguments: dict) -> dict:
    """Route cairn_maintain actions to existing handlers."""
    action = arguments.get("action", "").strip()

    if action == "health":
        return await handle_cairn_health(arguments)
    elif action == "consolidate":
        return await handle_cairn_consolidate(arguments)
    elif action == "compact":
        return await handle_cairn_compact(arguments)
    elif action == "backup":
        return await handle_cairn_backup({**arguments, "mode": "export"})
    elif action == "restore":
        return await handle_cairn_backup({**arguments, "mode": "import"})
    elif action == "clear_session":
        return await handle_cairn_clear_session(arguments)
    elif action == "discover_connections":
        try:
            from cairn.bridge import discover_connections
            dry_run = arguments.get("dry_run", False)
            lookback_hours = _clamp_int(arguments.get("lookback_hours", 24), default=24, max_val=168)
            raw_threshold = arguments.get("similarity_threshold", 0.70)
            if isinstance(raw_threshold, (int, float)):
                similarity_threshold = max(0.5, min(0.95, float(raw_threshold)))
            else:
                similarity_threshold = 0.70
            return await _run_or_submit_maintain(
                "discover_connections",
                lambda: discover_connections(
                    lookback_hours=lookback_hours,
                    similarity_threshold=similarity_threshold,
                    dry_run=dry_run,
                ),
                arguments,
            )
        except Exception as e:
            logger.error("discover_connections failed: %s", e, exc_info=True)
            return mcp_error("Connection discovery failed")
    elif action == "synthesize_insights":
        try:
            from cairn.bridge import synthesize_system_insights
            dry_run = arguments.get("dry_run", True)
            return await _run_or_submit_maintain(
                "synthesize_insights",
                lambda: synthesize_system_insights(dry_run=dry_run),
                arguments,
            )
        except Exception as e:
            logger.error("synthesize_insights failed: %s", e, exc_info=True)
            return mcp_error("Synthesize insights failed")
    elif action == "backfill_embeddings":
        try:
            from cairn.bridge import backfill_embeddings
            batch_size = _clamp_int(arguments.get("batch_size", 50), default=50, max_val=200)
            return await _run_or_submit_maintain(
                "backfill_embeddings",
                lambda: backfill_embeddings(batch_size=batch_size),
                arguments,
            )
        except Exception as e:
            logger.error("backfill_embeddings failed: %s", e, exc_info=True)
            return mcp_error("Backfill embeddings failed")
    elif action == "job_status":
        job_id = arguments.get("job_id", "").strip()
        if not job_id:
            return mcp_error("job_id is required for job_status")
        try:
            from cairn.server.jobs import get_registry

            job = get_registry().get(job_id)
            if job is None:
                return mcp_error(f"Job {job_id} not found (expired or unknown)")
            return mcp_response(_format_job_status(job))
        except Exception as e:
            logger.error("job_status failed: %s", e, exc_info=True)
            return mcp_error("Job status failed")
    elif action == "list_constraints":
        try:
            from cairn.bridge import list_constraints
            result = list_constraints(arguments.get("project"))
            return mcp_response(result)
        except Exception as e:
            logger.error("list_constraints failed: %s", e, exc_info=True)
            return mcp_error("List constraints failed")
    elif action == "check_constraint":
        try:
            from cairn.bridge import check_constraints
            file_path = arguments.get("file_path", "").strip()
            if not file_path:
                return mcp_error("file_path is required for check_constraint")
            violations = check_constraints(file_path, arguments.get("project"))
            return mcp_response({"file_path": file_path, "violations": violations, "count": len(violations)})
        except Exception as e:
            logger.error("check_constraint failed: %s", e, exc_info=True)
            return mcp_error("Check constraint failed")
    elif action == "save_constraints":
        try:
            from cairn.bridge import save_constraints
            rules = arguments.get("rules")
            if not rules or not isinstance(rules, list):
                return mcp_error("rules (list) is required for save_constraints")
            result = save_constraints(rules, arguments.get("project"))
            return mcp_response(result)
        except Exception as e:
            logger.error("save_constraints failed: %s", e, exc_info=True)
            return mcp_error("Save constraints failed")
    else:
        return mcp_error(f"Unknown cairn_maintain action: {action}. Use: health, consolidate, compact, discover_connections, backup, restore, clear_session, synthesize_insights, backfill_embeddings, job_status, list_constraints, check_constraint, save_constraints")


# ============================================================================
# Composite Handler: cairn_stats (types, sessions, digest, forgetting_log)
# ============================================================================


async def handle_cairn_stats(arguments: dict) -> dict:
    """Route cairn_stats actions to existing handlers."""
    action = arguments.get("action", "").strip()

    if action == "types":
        return await handle_cairn_type_stats(arguments)
    elif action == "sessions":
        return await handle_cairn_session_stats(arguments)
    elif action == "digest":
        return await handle_cairn_weekly_digest(arguments)
    elif action == "forgetting_log":
        return await handle_cairn_forgetting_log(arguments)
    elif action == "dedup":
        return await handle_cairn_dedup_stats(arguments)
    elif action == "milestones":
        return await handle_cairn_milestones(arguments)
    elif action == "access_rate":
        return await handle_cairn_access_rate(arguments)
    elif action == "retrieval_context":
        return await handle_cairn_retrieval_context(arguments)
    elif action == "diagnostic":
        return await handle_cairn_diagnostic(arguments)
    elif action == "graph_stats":
        try:
            from cairn.bridge import _get_store
            store = _get_store()
            conn = store._conn
            # Total edges
            total = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
            # Edge type distribution
            type_dist = conn.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM edges GROUP BY edge_type ORDER BY cnt DESC"
            ).fetchall()
            # Avg edges per memory
            node_count = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            avg_edges = round(total / max(node_count, 1), 2)
            output = "# Graph Stats\n\n"
            output += f"- **Total edges:** {total}\n"
            output += f"- **Total nodes:** {node_count}\n"
            output += f"- **Avg edges/node:** {avg_edges}\n\n"
            if type_dist:
                output += "## Edge Type Distribution\n"
                output += "| Type | Count |\n|------|-------|\n"
                for row in type_dist:
                    output += f"| {row[0]} | {row[1]} |\n"
            return mcp_response(output)
        except Exception as e:
            logger.error("graph_stats failed: %s", e, exc_info=True)
            return mcp_error("Graph stats failed")
    elif action == "utilization":
        try:
            from cairn.usage_tracker import UsageTracker

            # Collect all defined tool names from schemas
            from cairn.server.tool_schemas import TOOL_SCHEMAS

            all_tools = {t["name"] for t in TOOL_SCHEMAS}
            try:
                from cairn.server.coord_schemas import COORD_TOOL_SCHEMAS

                all_tools |= {t["name"] for t in COORD_TOOL_SCHEMAS}
            except ImportError:
                pass

            tracker = UsageTracker()
            try:
                # Last 30 days usage
                top_tools_30d = tracker.get_top_tools(days=30, limit=200)
                used_tools = {t["tool_name"] for t in top_tools_30d}
                never_used = sorted(all_tools - used_tools)

                total_defined = len(all_tools)
                total_used = len(used_tools & all_tools)
                pct_used = round(100 * total_used / max(total_defined, 1), 1)

                output = "# Tool Utilization Report (30 days)\n\n"
                output += f"- **Tools defined:** {total_defined}\n"
                output += f"- **Tools with usage:** {total_used} ({pct_used}%)\n"
                output += f"- **Never called:** {len(never_used)}\n\n"

                # Top 10 most-called
                top10 = sorted(top_tools_30d, key=lambda t: t["call_count"], reverse=True)[:10]
                if top10:
                    output += "## Top 10 Most-Called Tools\n"
                    output += "| Tool | Calls | Tokens | Cost (USD) |\n"
                    output += "|------|-------|--------|------------|\n"
                    for t in top10:
                        cost = f"${t['total_cost_usd']:.4f}" if t["total_cost_usd"] else "$0"
                        output += f"| {t['tool_name']} | {t['call_count']} | {t['total_tokens']:,} | {cost} |\n"

                # Never-called tools
                if never_used:
                    output += "\n## Never-Called Tools\n"
                    for name in never_used:
                        output += f"- {name}\n"

                # Utilization trend: last 7 days vs previous 7 days
                recent_7d = tracker.get_top_tools(days=7, limit=200)
                recent_calls = sum(t["call_count"] for t in recent_7d)
                recent_tools = len({t["tool_name"] for t in recent_7d})

                prev_14d = tracker.get_top_tools(days=14, limit=200)
                prev_calls_14d = sum(t["call_count"] for t in prev_14d)
                prev_tools_14d = {t["tool_name"] for t in prev_14d}
                # Previous 7 days = 14-day totals minus last 7 days
                prev_calls = prev_calls_14d - recent_calls
                prev_tools = len(prev_tools_14d - {t["tool_name"] for t in recent_7d})

                output += "\n## Utilization Trend (7-day comparison)\n"
                output += f"- **Last 7 days:** {recent_calls} calls across {recent_tools} tools\n"
                output += f"- **Previous 7 days:** {prev_calls} calls\n"
                if prev_calls > 0:
                    change = round(100 * (recent_calls - prev_calls) / prev_calls, 1)
                    direction = "up" if change > 0 else "down"
                    output += f"- **Change:** {direction} {abs(change)}%\n"

            finally:
                tracker.close()

            return mcp_response(output)
        except Exception as e:
            logger.error("utilization check failed: %s", e, exc_info=True)
            return mcp_error(f"Utilization check failed: {e}")
    # Behavioral habits actions (removed in the self-hosted fork)
    elif action in ("habits_list", "habits_confirm", "habits_deny", "habits_analyze", "habits_profile", "habits_recommendations"):
        return await handle_cairn_habits(arguments)
    else:
        return mcp_error(f"Unknown cairn_stats action: {action}. Use: types, sessions, digest, forgetting_log, dedup, milestones, access_rate, retrieval_context, diagnostic, graph_stats, utilization")


async def handle_cairn_access_rate(arguments: dict) -> dict:
    """Return access rate breakdown for memories."""
    try:
        from cairn.bridge import access_rate_stats

        stats = access_rate_stats()

        output = "# Memory Access Rate\n\n"
        output += f"- **Total memories:** {stats['total_memories']}\n"
        output += f"- **Never accessed:** {stats['zero_access_count']} ({stats['never_accessed_pct']}%)\n"
        output += f"- **Average access count:** {stats['avg_access_count']}\n\n"

        output += "## By Event Type\n"
        output += "| Type | Count | Avg Access | Never Accessed |\n"
        output += "|------|-------|------------|----------------|\n"
        for t in stats["by_type"]:
            output += f"| {t['event_type']} | {t['count']} | {t['avg_access_count']} | {t['zero_access_count']} ({t['zero_access_pct']}%) |\n"

        if stats["top_accessed"]:
            output += "\n## Top 10 Most Accessed\n"
            for m in stats["top_accessed"]:
                output += f"- **{m['access_count']}x** [{m['event_type']}] {m['content']}\n"

        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_access_rate failed: %s", e, exc_info=True)
        return mcp_error(f"Access rate query failed: {e}")


async def handle_cairn_diagnostic(arguments: dict) -> dict:
    """Unified Cairn health and value diagnostic."""
    try:
        from cairn import json_compat as json
        from cairn.bridge import diagnostic_report

        days = arguments.get("days", 30)
        report = diagnostic_report(days=days)
        return {"content": [{"type": "text", "text": json.dumps(report, indent=2)}]}
    except Exception as e:
        logger.error("cairn_diagnostic failed: %s", e, exc_info=True)
        return mcp_error(f"Diagnostic failed: {e}")


async def handle_cairn_retrieval_context(arguments: dict) -> dict:
    """Return recent retrieval context (query/score/vec_sim per retrieved memory)."""
    try:
        from cairn.bridge import retrieval_context

        entries = retrieval_context()
        if not entries:
            return mcp_response("No retrieval context available (no recent queries).")

        output = "# Recent Retrieval Context\n\n"
        output += "| Node ID | Query | Score | Vec Sim | Timestamp |\n"
        output += "|---------|-------|-------|---------|-----------|\n"
        for e in entries:
            nid = e.get("node_id", "?")[:12]
            query = (e.get("query_text") or "")[:40]
            score = e.get("score", 0.0)
            vec_sim = e.get("vec_sim", 0.0)
            ts = (e.get("timestamp") or "")[:19]
            output += f"| {nid} | {query} | {score:.4f} | {vec_sim:.4f} | {ts} |\n"

        output += f"\n**Total entries:** {len(entries)}"
        return mcp_response(output)
    except Exception as e:
        logger.error("cairn_retrieval_context failed: %s", e, exc_info=True)
        return mcp_error(f"Retrieval context query failed: {e}")


async def handle_cairn_milestones(arguments: dict) -> dict:
    """Return achieved milestones and current streak."""
    try:
        from cairn.milestones import list_milestones, get_streak
        from cairn.bridge import _get_store

        milestones = list_milestones()
        store = _get_store()
        streak = get_streak(store)

        output = "# Milestones & Streaks\n\n"

        # Streak section
        output += "## Streak\n"
        output += f"- **Current:** {streak['current']} day{'s' if streak['current'] != 1 else ''}\n"
        output += f"- **Longest:** {streak['longest']} day{'s' if streak['longest'] != 1 else ''}\n"
        output += f"- **Active today:** {'Yes' if streak['today_active'] else 'No'}\n\n"

        # Milestones section
        output += "## Milestones\n"
        if milestones:
            for m in milestones:
                achieved = m.get("achieved_at", "unknown")[:16]
                output += f"- **{m['name']}** ({achieved})\n"
        else:
            output += "*No milestones achieved yet.*\n"

        return mcp_response(output)
    except ImportError:
        return mcp_error("Milestones module not available in this build")
    except Exception as e:
        logger.error("cairn_milestones failed: %s", e, exc_info=True)
        return mcp_error(f"Milestones query failed: {e}")


# ============================================================================
# Composite Handler: cairn_habits (removed)
# ============================================================================


async def handle_cairn_habits(arguments: dict) -> dict:
    """Behavioral pattern analysis was removed in the self-hosted fork."""
    return mcp_error(
        "Behavioral pattern analysis (cairn_habits) is not available in this "
        "build — it was removed in the self-hosted fork."
    )


# ============================================================================
# Composite Handler: cairn_reflect (contradictions, evolution, stale)
# ============================================================================


async def handle_cairn_reflect(arguments: dict) -> dict:
    """Route cairn_reflect actions to analysis functions."""
    action = arguments.get("action", "").strip()

    if action == "contradictions":
        return await _handle_reflect_contradictions(arguments)
    elif action == "evolution":
        return await _handle_reflect_evolution(arguments)
    elif action == "stale":
        return await _handle_reflect_stale(arguments)
    else:
        return mcp_error(
            f"Unknown cairn_reflect action: {action}. Use: contradictions, evolution, stale"
        )


async def _handle_reflect_contradictions(arguments: dict) -> dict:
    """Find contradicting memories on a topic."""
    topic = (arguments.get("topic") or "").strip()
    if not topic:
        return mcp_error("'topic' is required for action='contradictions'")

    try:
        from cairn.bridge import _get_store
        from cairn.reflect import find_contradictions

        store = _get_store()
        limit = _clamp_int(arguments.get("limit", 20), default=20, max_val=50)
        entity_id = _validate_entity_id(arguments.get("entity_id"))

        result = find_contradictions(store, topic, limit=limit, entity_id=entity_id)

        output = f"# Contradiction Audit: {topic}\n\n"
        output += f"**Memories analyzed:** {result['memories_analyzed']}\n"
        output += f"**Contradictions found:** {len(result['contradictions'])}\n\n"

        if not result["contradictions"]:
            output += "No contradictions detected."
        else:
            for i, c in enumerate(result["contradictions"], 1):
                output += f"## {i}. Confidence: {c['confidence']:.0%}\n"
                output += f"**Memory A** (`{c['memory_a_id'][:12]}`): {c['memory_a_content']}\n\n"
                output += f"**Memory B** (`{c['memory_b_id'][:12]}`): {c['memory_b_content']}\n\n"
                output += f"**Signals:** {', '.join(c['signals'])} | **Reason:** {c['reason']}\n\n"
                output += "---\n\n"

        return mcp_response(output)
    except ImportError:
        return mcp_error("Contradiction analysis module not available in this build")
    except Exception as e:
        logger.error("cairn_reflect contradictions failed: %s", e, exc_info=True)
        return mcp_error(f"Contradiction audit failed: {e}")


async def _handle_reflect_evolution(arguments: dict) -> dict:
    """Trace how understanding of a topic evolved."""
    topic = (arguments.get("topic") or "").strip()
    if not topic:
        return mcp_error("'topic' is required for action='evolution'")

    try:
        from cairn.bridge import _get_store
        from cairn.reflect import trace_evolution

        store = _get_store()
        limit = _clamp_int(arguments.get("limit", 20), default=20, max_val=50)
        entity_id = _validate_entity_id(arguments.get("entity_id"))

        result = trace_evolution(store, topic, limit=limit, entity_id=entity_id)

        output = f"# Knowledge Evolution: {topic}\n\n"
        output += f"**Total memories:** {result['total_memories']}\n"
        output += f"**Evolution chains:** {len(result['chains'])}\n\n"

        if not result["chains"]:
            output += "No evolution chains found (memories may exist but lack evolution/supersedes edges)."
        else:
            for i, chain in enumerate(result["chains"], 1):
                output += f"## Chain {i} ({chain['length']} memories)\n\n"
                for j, mem in enumerate(chain["memories"]):
                    marker = "  " if j > 0 else ""
                    ts = mem["created_at"][:19] if mem["created_at"] else "?"
                    etype = f" [{mem['event_type']}]" if mem["event_type"] else ""
                    output += f"{marker}{j + 1}. `{mem['node_id'][:12]}` ({ts}){etype}\n"
                    output += f"{marker}   {mem['content']}\n\n"

                if chain["edges"]:
                    output += "**Edges:** "
                    edge_descs = [
                        f"`{e['from'][:8]}`-[{e['edge_type']}]->`{e['to'][:8]}`"
                        for e in chain["edges"]
                    ]
                    output += ", ".join(edge_descs) + "\n\n"

                output += "---\n\n"

        return mcp_response(output)
    except ImportError:
        return mcp_error("Evolution tracing module not available in this build")
    except Exception as e:
        logger.error("cairn_reflect evolution failed: %s", e, exc_info=True)
        return mcp_error(f"Evolution trace failed: {e}")


async def _handle_reflect_stale(arguments: dict) -> dict:
    """Surface stale memories for human review."""
    try:
        from cairn.bridge import _get_store
        from cairn.reflect import find_stale

        store = _get_store()
        days = _clamp_int(arguments.get("days", 30), default=30, max_val=365)
        min_age_days = _clamp_int(arguments.get("min_age_days", 14), default=14, max_val=365)
        limit = _clamp_int(arguments.get("limit", 30), default=30, max_val=100)
        entity_id = _validate_entity_id(arguments.get("entity_id"))

        result = find_stale(store, days=days, min_age_days=min_age_days, limit=limit, entity_id=entity_id)

        output = "# Stale Memory Audit\n\n"
        output += f"**Total candidates:** {result['total_candidates']}\n"
        output += f"**Showing:** {len(result['stale_memories'])} (sorted by staleness)\n\n"

        if not result["stale_memories"]:
            output += "No stale memories found. Your memory store is well-maintained!"
        else:
            output += "| # | ID | Score | Age | Type | Reasons | Preview |\n"
            output += "|---|-----|-------|-----|------|---------|---------|\n"
            for i, m in enumerate(result["stale_memories"], 1):
                mid = m["id"][:12]
                score = f"{m['staleness_score']:.0%}"
                # Calculate age from created_at
                age = ""
                if m["created_at"]:
                    try:
                        from datetime import datetime, timezone
                        created = datetime.fromisoformat(m["created_at"])
                        age_days = (datetime.now(timezone.utc) - created).days
                        age = f"{age_days}d"
                    except Exception as e:
                        logger.debug("Date parse failed for stale audit: %s", e)
                        age = "?"
                etype = m["event_type"]
                reasons = ", ".join(m["reasons"])
                preview = m["content_preview"][:60].replace("|", "/").replace("\n", " ")
                output += f"| {i} | `{mid}` | {score} | {age} | {etype} | {reasons} | {preview} |\n"

            output += "\n**Actions:** Use `cairn_memory(action='delete', memory_id='...')` to remove, or `cairn_memory(action='feedback', memory_id='...', rating='helpful')` to mark as worth keeping."

        return mcp_response(output)
    except ImportError:
        return mcp_error("Stale memory analysis module not available in this build")
    except Exception as e:
        logger.error("cairn_reflect stale failed: %s", e, exc_info=True)
        return mcp_error(f"Stale audit failed: {e}")


# ============================================================================
# GPT Consultation
# ============================================================================


async def handle_cairn_consult_gpt(arguments: dict) -> dict:
    """Consult GPT for a second opinion on hard problems."""
    prompt = (arguments.get("prompt") or "").strip()
    if not prompt:
        return mcp_error("'prompt' is required for cairn_consult_gpt")

    context = (arguments.get("context") or "").strip()
    if context:
        full_prompt = f"{prompt}\n\n--- Context ---\n{context}"
    else:
        full_prompt = prompt

    kwargs: dict = {}
    if "system" in arguments and arguments["system"]:
        kwargs["system"] = arguments["system"]
    if "temperature" in arguments and arguments["temperature"] is not None:
        kwargs["temperature"] = float(arguments["temperature"])
    if "max_tokens" in arguments and arguments["max_tokens"] is not None:
        kwargs["max_tokens"] = _clamp_int(arguments["max_tokens"], default=4096, min_val=1, max_val=16384)

    try:
        from cairn.llm import gpt_complete
    except ImportError:
        return mcp_error(
            "GPT consultation requires the 'openai' package. "
            "Install with: pip install openai"
        )

    model = os.environ.get("CAIRN_GPT_MODEL", "gpt-4o")
    response = gpt_complete(full_prompt, **kwargs)

    if not response:
        return mcp_error(
            "GPT consultation returned empty response. "
            "Check: OPENAI_API_KEY is set, model is accessible, prompt is valid."
        )

    return mcp_response(f"## GPT Consultation ({model})\n\n{response}")


async def handle_cairn_consult_claude(arguments: dict) -> dict:
    """Consult Claude for a second opinion on hard problems (for non-Anthropic agents)."""
    prompt = (arguments.get("prompt") or "").strip()
    if not prompt:
        return mcp_error("'prompt' is required for cairn_consult_claude")

    context = (arguments.get("context") or "").strip()
    if context:
        full_prompt = f"{prompt}\n\n--- Context ---\n{context}"
    else:
        full_prompt = prompt

    kwargs: dict = {}
    if "system" in arguments and arguments["system"]:
        kwargs["system"] = arguments["system"]
    if "temperature" in arguments and arguments["temperature"] is not None:
        kwargs["temperature"] = float(arguments["temperature"])
    if "max_tokens" in arguments and arguments["max_tokens"] is not None:
        kwargs["max_tokens"] = _clamp_int(arguments["max_tokens"], default=4096, min_val=1, max_val=16384)

    try:
        from cairn.llm import claude_complete
    except ImportError:
        return mcp_error(
            "Claude consultation requires the 'anthropic' package. "
            "Install with: pip install anthropic"
        )

    model = os.environ.get("CAIRN_CLAUDE_MODEL", "claude-sonnet-4-6")
    response = claude_complete(full_prompt, **kwargs)

    if not response:
        return mcp_error(
            "Claude consultation returned empty response. "
            "Check: ANTHROPIC_API_KEY is set, model is accessible, prompt is valid."
        )

    return mcp_response(f"## Claude Consultation ({model})\n\n{response}")


# ============================================================================
# Handler: cairn_review
# ============================================================================


async def handle_cairn_review(arguments: dict) -> dict:
    """Review a code diff with multi-agent specialist panel powered by Cairn memory."""
    diff_text = arguments.get("diff", "").strip()
    if not diff_text:
        return mcp_error("diff is required")

    repo = arguments.get("repo", "unknown")
    mode = arguments.get("mode", "normal")
    if mode not in ("strict", "normal", "verbose"):
        mode = "normal"
    agents = arguments.get("agents")
    summarize_only = arguments.get("summarize_only", False)
    session_id = _validate_session_id(arguments.get("session_id"))
    entity_id = _validate_entity_id(arguments.get("entity_id"))

    try:
        from cairn.review import run_review
        result = run_review(
            diff_text=diff_text,
            repo=repo,
            mode=mode,
            agent_types=agents,
            summarize_only=summarize_only,
            session_id=session_id,
            entity_id=entity_id,
        )
        return mcp_response(result)
    except ImportError:
        return mcp_error("Code review requires the revue package. Install: pip install revue")
    except Exception as e:
        logger.error("cairn_review failed: %s", e, exc_info=True)
        return mcp_error(f"Review failed: {e}")


# ============================================================================
# Condensed Mode Meta-Tool Handlers
# ============================================================================

# Populated by mcp_server.py after all schemas (core + pro + plugins) are merged.
_ALL_SCHEMAS: list = []
# Reference to the full HANDLERS dict, set after dict creation below.
_ALL_HANDLERS: dict = {}


async def handle_cairn_tools(args: Dict[str, Any]) -> dict:
    """List available tools or get the full schema for a specific tool."""
    import json

    tool_name = args.get("tool")

    if tool_name:
        # Return full schema for a specific tool
        for schema in _ALL_SCHEMAS:
            if schema["name"] == tool_name:
                return mcp_response(json.dumps(schema["inputSchema"], indent=2))
        return mcp_error(f"Unknown tool: {tool_name}")

    # List all tools that have a registered handler (or are meta-tools)
    meta_tools = {"cairn_tools", "cairn_call"}
    lines = []
    for schema in _ALL_SCHEMAS:
        name = schema["name"]
        if name not in _ALL_HANDLERS and name not in meta_tools:
            continue
        lines.append(f"- **{name}**: {schema['description']}")

    if not lines:
        return mcp_response("No tools available.")

    header = f"Available Cairn tools ({len(lines)}):\n\n"
    footer = "\n\nUse cairn_tools(tool='name') to get the full input schema for any tool."
    return mcp_response(header + "\n".join(lines) + footer)


async def handle_cairn_call(args: Dict[str, Any]) -> dict:
    """Execute any Cairn tool by name with arguments."""
    tool_name = args.get("tool")
    tool_args = args.get("args") or {}

    if not tool_name:
        return mcp_error("Required parameter 'tool' is missing.")

    if tool_name in ("cairn_call", "cairn_tools"):
        return mcp_error("Cannot call meta-tools through cairn_call. Use them directly.")

    handler = _ALL_HANDLERS.get(tool_name)
    if not handler:
        return mcp_error(f"Unknown tool: {tool_name}. Use cairn_tools() to list available tools.")

    return await handler(tool_args)


# ============================================================================
# Handler Registry
# ============================================================================

# The name -> handler map is now DERIVED from the single ToolSpec table in
# cairn.server.registry (which imports this module for the handler callables).
# To avoid an import cycle, this module never imports registry at load time;
# `HANDLERS` is resolved lazily on first attribute access via __getattr__.

# Harness-agnostic structured-context wire contract -- see context_handlers.py.
# Registered into the cairn_call dispatch table here; the composite/alias
# handlers are registered by registry.py when it builds HANDLERS.
from cairn.server.context_handlers import CONTEXT_HANDLERS  # noqa: E402
_ALL_HANDLERS.update(CONTEXT_HANDLERS)

_HANDLERS_CACHE: dict | None = None


def __getattr__(name: str):
    """Lazily expose the derived HANDLERS map (registry core + context tools).

    Deferring the registry import to attribute-access time keeps handlers.py
    free of any registry reference during module load, so the two modules load
    in either order without a cycle.
    """
    global _HANDLERS_CACHE
    if name == "HANDLERS":
        if _HANDLERS_CACHE is None:
            from cairn.server.registry import HANDLERS as _core
            _HANDLERS_CACHE = {**_core, **CONTEXT_HANDLERS}
        return _HANDLERS_CACHE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
