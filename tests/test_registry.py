"""Tests for the derived MCP tool registry (cairn.server.registry).

Closes the coverage gap the C2 plan flagged: the ToolSpec table is the single
source of truth, and every view (TOOL_SCHEMAS, HANDLERS, STANDALONE, condensed
set) is derived from it — plus the cairn_call / cairn_tools meta-tools and the
two arg-transform aliases had no direct coverage.
"""
import pytest

from cairn.server import registry
from cairn.server.registry import (
    TOOLS,
    TOOL_SCHEMAS,
    HANDLERS,
    STANDALONE_TOOLS,
    CONDENSED_TOOL_SCHEMAS,
    get_condensed_schemas,
    _remember_transform,
    _phrase_transform,
)
from cairn.server import handlers as _handlers_module
from cairn.server.tool_schemas import TOOL_SCHEMAS as WIRED_TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Derivation invariants — the whole point of the single-source table
# ---------------------------------------------------------------------------

def test_tool_names_are_unique():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)), "duplicate tool name in TOOLS"


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_every_spec_is_in_handlers(spec):
    assert spec.name in HANDLERS
    assert callable(HANDLERS[spec.name])


@pytest.mark.parametrize("spec", TOOLS, ids=[t.name for t in TOOLS])
def test_exposed_specs_generate_schemas(spec):
    """A non-meta spec with a schema appears in TOOL_SCHEMAS; meta ones do not."""
    if spec.schema is not None and not spec.meta:
        assert spec.schema in TOOL_SCHEMAS
    if spec.meta:
        assert spec.schema not in TOOL_SCHEMAS
        assert spec.schema in CONDENSED_TOOL_SCHEMAS


def test_registry_generates_all_views():
    # Every exposed schema has a handler; handlers are a superset of schemas.
    schema_names = {s["name"] for s in TOOL_SCHEMAS}
    assert schema_names <= set(HANDLERS)
    assert len(HANDLERS) >= len(TOOL_SCHEMAS) >= 12
    # Standalone tools are all exposed (must have a schema to be listed directly).
    for name in STANDALONE_TOOLS:
        assert name in schema_names
    # Meta tools are exactly the condensed extras.
    assert {s["name"] for s in CONDENSED_TOOL_SCHEMAS} == {t.name for t in TOOLS if t.meta}


def test_condensed_set_is_standalone_plus_meta():
    condensed = get_condensed_schemas(WIRED_TOOL_SCHEMAS)
    names = [s["name"] for s in condensed]
    assert set(STANDALONE_TOOLS) <= set(names)
    assert {"cairn_tools", "cairn_call"} <= set(names)
    # Nothing outside standalone + meta leaks into the condensed set.
    assert set(names) == set(STANDALONE_TOOLS) | {"cairn_tools", "cairn_call"}


# ---------------------------------------------------------------------------
# Arg-transform aliases
# ---------------------------------------------------------------------------

def test_remember_transform_defaults_event_type():
    assert _remember_transform({"text": "x"})["event_type"] == "user_preference"


def test_remember_transform_preserves_explicit_event_type():
    assert _remember_transform({"text": "x", "event_type": "decision"})["event_type"] == "decision"


def test_phrase_transform_maps_phrase_and_forces_mode():
    out = _phrase_transform({"phrase": "exact words", "limit": 3})
    assert out["query"] == "exact words"
    assert out["mode"] == "phrase"
    assert out["limit"] == 3


def test_phrase_transform_falls_back_to_query_key():
    out = _phrase_transform({"query": "already-query"})
    assert out["query"] == "already-query"
    assert out["mode"] == "phrase"


# ---------------------------------------------------------------------------
# Meta-tool dispatch: cairn_call / cairn_tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_bridge")
async def test_cairn_call_dispatches_to_granular_handler():
    result = await HANDLERS["cairn_call"](
        {"tool": "cairn_store", "args": {"content": "registry dispatch canary", "event_type": "decision"}}
    )
    assert not result.get("isError"), result
    text = result["content"][0]["text"]
    assert "Stored" in text or "Deduped" in text or "Evolved" in text


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_bridge")
async def test_cairn_call_rejects_meta_tools():
    for meta in ("cairn_call", "cairn_tools"):
        result = await HANDLERS["cairn_call"]({"tool": meta, "args": {}})
        assert result.get("isError")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_bridge")
async def test_cairn_call_unknown_tool_errors():
    result = await HANDLERS["cairn_call"]({"tool": "cairn_does_not_exist", "args": {}})
    assert result.get("isError")
    assert "Unknown tool" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.usefixtures("_reset_bridge")
async def test_cairn_call_missing_tool_param_errors():
    result = await HANDLERS["cairn_call"]({"args": {}})
    assert result.get("isError")


@pytest.mark.asyncio
async def test_cairn_tools_lists_handler_backed_tools(monkeypatch):
    # cairn_tools reads handlers._ALL_SCHEMAS (wired by mcp_server in prod).
    monkeypatch.setattr(_handlers_module, "_ALL_SCHEMAS", WIRED_TOOL_SCHEMAS, raising=False)
    result = await HANDLERS["cairn_tools"]({})
    text = result["content"][0]["text"]
    # Lists the exposed, handler-backed composites.
    assert "cairn_store" in text
    assert "cairn_query" in text
    assert "cairn_maintain" in text


@pytest.mark.asyncio
async def test_cairn_tools_returns_specific_schema(monkeypatch):
    monkeypatch.setattr(_handlers_module, "_ALL_SCHEMAS", WIRED_TOOL_SCHEMAS, raising=False)
    result = await HANDLERS["cairn_tools"]({"tool": "cairn_query"})
    text = result["content"][0]["text"]
    assert '"mode"' in text and "semantic" in text


@pytest.mark.asyncio
async def test_cairn_tools_unknown_tool_errors(monkeypatch):
    monkeypatch.setattr(_handlers_module, "_ALL_SCHEMAS", WIRED_TOOL_SCHEMAS, raising=False)
    result = await HANDLERS["cairn_tools"]({"tool": "cairn_nope"})
    assert result.get("isError")
