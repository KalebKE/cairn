"""Cairn MCP Tool Schemas -- 16 tools for memory management.

Backward-compatible view over the registry. The tool set is now defined once
in :mod:`cairn.server.registry` as a ``ToolSpec`` table; this module is a thin
compatibility shim that re-exports the derived ``TOOL_SCHEMAS`` /
``STANDALONE_TOOLS`` / ``CONDENSED_TOOL_SCHEMAS`` / ``get_condensed_schemas``
and layers the harness-agnostic context tools on top (env-gated wire
contract). Existing ``from cairn.server.tool_schemas import TOOL_SCHEMAS``
call sites keep working unchanged.

Condensed Mode (CodeMode-inspired):
  When CAIRN_CONDENSED=1, only five are exposed directly: 3 standalone
  essentials (cairn_welcome, cairn_protocol, cairn_store) + 2 meta-tools
  (cairn_tools, cairn_call). Everything else is reachable via
  cairn_call(tool=..., args=...).
"""
from cairn.server.registry import (  # noqa: F401
    TOOL_SCHEMAS as _CORE_TOOL_SCHEMAS,
    STANDALONE_TOOLS,
    CONDENSED_TOOL_SCHEMAS,
    get_condensed_schemas,
)

# Fresh mutable copy so the context extend below does not mutate the registry's
# canonical list.
TOOL_SCHEMAS = list(_CORE_TOOL_SCHEMAS)

# Harness-agnostic context wire contract. Gate the lower-level generic
# context_assemble alias behind an env var, but expose context_packet by
# default: build_context_packet is the single rendering path for both this
# MCP tool and the daemon's PostToolUse surfacing hook.
import os as _os
from cairn.server.context_tool_schemas import CONTEXT_TOOL_SCHEMAS  # noqa: E402

TOOL_SCHEMAS.extend(
    schema for schema in CONTEXT_TOOL_SCHEMAS
    if schema.get("name") == "context_packet"
)
if _os.environ.get("CAIRN_CONTEXT_WIRE", "").lower() in ("1", "true", "yes"):
    TOOL_SCHEMAS.extend(
        schema for schema in CONTEXT_TOOL_SCHEMAS
        if schema.get("name") != "context_packet"
    )
