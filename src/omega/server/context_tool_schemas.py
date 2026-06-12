"""Harness-agnostic structured-context wire contract.

Exposes OMEGA's "active context digest" as structured sections that any
harness (Claude Code today, iii / OpenAI Agents SDK / LangGraph tomorrow)
can splice into its own system prompt -- without depending on the Claude
Code CLAUDE.md SessionStart hook.

Coexists with omega_welcome (which keeps its rich Claude-Code-shaped
Dict return). Same SQLite source, different face.

The section list is an honest mapping to OMEGA's actual event_type model:
no invented "identity" or "hard_rules" sections -- those belong to the
caller's harness layer, not OMEGA.
"""

_SCOPE_SCHEMA = {
    "type": "object",
    "description": "Scope this assembly. All fields optional; combine as needed.",
    "properties": {
        "entity_id": {"type": "string"},
        "project": {"type": "string"},
        "session_id": {"type": "string"},
    },
    "additionalProperties": False,
}


CONTEXT_TOOL_SCHEMAS = [
    {
        "name": "context_assemble",
        "description": (
            "Assemble OMEGA's active context as structured sections plus a "
            "pre-rendered markdown blob. Harness-agnostic; clients splice into "
            "their own system prompt. For Claude Code, omega_welcome remains the "
            "richer briefing surface."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": _SCOPE_SCHEMA,
                "sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional filter. Default: all 6 sections "
                        "(active_reminders, active_constraints, recent_decisions, "
                        "top_lessons, user_preferences, user_facts). Unknown ids ignored."
                    ),
                },
                "since": {
                    "type": "string",
                    "description": (
                        "ISO-8601 lower bound for recent_decisions and top_lessons. "
                        "Default: 7 days ago."
                    ),
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "context_packet",
        "description": (
            "Build a compact task-aware memory packet for the current work. "
            "Combines retrieval seeds, graph chains, safety filtering, warning "
            "receipts, and token-budgeted markdown."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Current task or user intent to retrieve context for.",
                },
                "files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant file paths. Used as retrieval and packet-title hints.",
                },
                "scope": _SCOPE_SCHEMA,
                "mode": {
                    "type": "string",
                    "enum": ["before_edit", "planning", "debug", "review", "command"],
                    "description": "Retrieval surface hint. Default: before_edit.",
                },
                "budget_tokens": {
                    "type": "integer",
                    "minimum": 120,
                    "maximum": 4000,
                    "description": "Approximate packet budget. Default: 800.",
                },
                "max_sensitivity": {
                    "type": "string",
                    "enum": ["public", "internal", "confidential", "restricted"],
                    "description": "Maximum sensitivity allowed in the packet. Default: restricted.",
                },
                "include_receipt": {
                    "type": "boolean",
                    "description": "Include packet usage/token receipt. Default: true.",
                },
            },
            "additionalProperties": False,
        },
    },
]
