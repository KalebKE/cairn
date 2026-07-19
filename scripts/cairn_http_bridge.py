#!/usr/bin/env python3
"""
Cairn HTTP Bridge — Exposes Cairn's bridge API over HTTP.

Thin FastAPI wrapper that accepts JSON-RPC 2.0 requests and delegates to
CairnBridge. Project scoping is configured via environment variables.

Usage:
    CAIRN_BRIDGE_PROJECT=myproject CAIRN_BRIDGE_PROJECT_PATH=/path/to/project \\
        uvicorn cairn.scripts.cairn_http_bridge:app --port 9092
    # or directly:
    python scripts/cairn_http_bridge.py
"""

import logging
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from cairn.bridge import (
    auto_capture,
    check_health,
    get_cross_session_lessons,
    query,
    timeline,
    type_stats,
)

logger = logging.getLogger("cairn.http_bridge")

# ---------------------------------------------------------------------------
# Project scoping — configure via env vars
# ---------------------------------------------------------------------------
PROJECT = os.getenv("CAIRN_BRIDGE_PROJECT", "default")
PROJECT_PATH = os.getenv("CAIRN_BRIDGE_PROJECT_PATH", "")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Cairn HTTP Bridge", version="1.0.0")


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str = 1
    method: str
    params: Dict[str, Any] = {}


class JsonRpcError(BaseModel):
    code: int
    message: str


class JsonRpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str = 1
    result: Optional[Dict[str, Any]] = None
    error: Optional[JsonRpcError] = None


def _text_result(text: str) -> Dict[str, Any]:
    """Wrap a string into MCP-compatible content format."""
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _handle_cairn_store(args: Dict[str, Any]) -> str:
    content = args.get("content", "")
    event_type = args.get("event_type", "decision")
    metadata = args.get("metadata")
    return auto_capture(
        content=content,
        event_type=event_type,
        metadata=metadata,
        project=PROJECT,
    )


def _handle_cairn_query(args: Dict[str, Any]) -> str:
    query_text = args.get("query", "")
    limit = args.get("limit", 10)
    event_type = args.get("event_type")
    return query(
        query_text=query_text,
        limit=limit,
        project=PROJECT,
        event_type=event_type,
    )


def _handle_cairn_lessons(args: Dict[str, Any]) -> str:
    task = args.get("task")
    limit = args.get("limit", 5)
    lessons = get_cross_session_lessons(
        task=task,
        project_path=PROJECT_PATH,
        limit=limit,
    )
    if not lessons:
        return "No lessons found."
    lines = []
    for i, lesson in enumerate(lessons, 1):
        lines.append(f"{i}. [{lesson.get('event_type', 'lesson')}] {lesson.get('content', '')[:200]}")
    return "\n".join(lines)


def _handle_cairn_timeline(args: Dict[str, Any]) -> str:
    days = args.get("days", 7)
    limit_per_day = args.get("limit_per_day", 10)
    return timeline(days=days, limit_per_day=limit_per_day)


def _handle_cairn_type_stats(args: Dict[str, Any]) -> str:
    stats = type_stats()
    import json
    return json.dumps(stats)


def _handle_cairn_health(args: Dict[str, Any]) -> str:
    return check_health()


def _handle_cairn_route_prompt(args: Dict[str, Any]) -> str:
    try:
        from cairn.router.engine import route_prompt
    except ImportError:
        return '{"error": "Router not installed. pip install cairn[router]"}'

    prompt = args.get("prompt", "")
    priority = args.get("priority", "balanced")
    result = route_prompt(prompt=prompt, priority=priority)
    import json
    return json.dumps(result)


TOOL_HANDLERS = {
    "cairn_store": _handle_cairn_store,
    "cairn_query": _handle_cairn_query,
    "cairn_lessons": _handle_cairn_lessons,
    "cairn_timeline": _handle_cairn_timeline,
    "cairn_type_stats": _handle_cairn_type_stats,
    "cairn_health": _handle_cairn_health,
    "cairn_route_prompt": _handle_cairn_route_prompt,
}


# ---------------------------------------------------------------------------
# JSON-RPC endpoint
# ---------------------------------------------------------------------------

@app.post("/mcp")
async def mcp_endpoint(req: JsonRpcRequest) -> JsonRpcResponse:
    """Handle JSON-RPC 2.0 requests in MCP tools/call format."""
    if req.method != "tools/call":
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(code=-32601, message=f"Unknown method: {req.method}"),
        )

    tool_name = req.params.get("name", "")
    tool_args = req.params.get("arguments", {})

    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(
                code=-32602,
                message=f"Unknown tool: {tool_name}. Available: {', '.join(TOOL_HANDLERS)}",
            ),
        )

    try:
        result_text = handler(tool_args)
        return JsonRpcResponse(id=req.id, result=_text_result(result_text))
    except Exception as e:
        logger.error("Tool %s failed: %s\n%s", tool_name, e, traceback.format_exc())
        return JsonRpcResponse(
            id=req.id,
            error=JsonRpcError(code=-32000, message=str(e)),
        )


@app.get("/health")
async def health_check():
    """Quick health endpoint for monitoring."""
    return {"status": "ok", "project": PROJECT}


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9092, log_level="info")
