"""Tests for constraint management via cairn_maintain handler."""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_maintain_list_constraints():
    """cairn_maintain(action='list_constraints') returns rules."""
    from cairn.server.handlers import handle_cairn_maintain

    mock_result = {"count": 1, "rules": [{"pattern": "*.env", "constraint": "no edit", "severity": "block"}], "constraints_dir": "/tmp"}
    with patch("cairn.bridge.list_constraints", return_value=mock_result):
        result = await handle_cairn_maintain({"action": "list_constraints"})
    assert result["content"][0]["text"]  # mcp_response wraps in content


@pytest.mark.asyncio
async def test_maintain_list_constraints_with_project():
    """list_constraints passes project through."""
    from cairn.server.handlers import handle_cairn_maintain

    with patch("cairn.bridge.list_constraints", return_value={"count": 0, "rules": [], "constraints_dir": "/tmp"}) as mock_lc:
        await handle_cairn_maintain({"action": "list_constraints", "project": "/my/project"})
    mock_lc.assert_called_once_with("/my/project")


@pytest.mark.asyncio
async def test_maintain_check_constraint():
    """cairn_maintain(action='check_constraint') checks file path."""
    from cairn.server.handlers import handle_cairn_maintain

    mock_violations = [{"pattern": ".env*", "constraint": "secrets file", "severity": "block", "source": "global"}]
    with patch("cairn.bridge.check_constraints", return_value=mock_violations):
        result = await handle_cairn_maintain({"action": "check_constraint", "file_path": ".env"})
    text = result["content"][0]["text"]
    assert "violations" in text or ".env" in text


@pytest.mark.asyncio
async def test_maintain_check_constraint_missing_path():
    """check_constraint requires file_path."""
    from cairn.server.handlers import handle_cairn_maintain

    result = await handle_cairn_maintain({"action": "check_constraint"})
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_maintain_save_constraints():
    """cairn_maintain(action='save_constraints') round-trips rules."""
    from cairn.server.handlers import handle_cairn_maintain

    rules = [{"pattern": "*.secret", "constraint": "no touch", "severity": "warn"}]
    mock_result = {"saved": 1, "file": "/tmp/global.json"}
    with patch("cairn.bridge.save_constraints", return_value=mock_result):
        result = await handle_cairn_maintain({"action": "save_constraints", "rules": rules})
    assert not result.get("isError")


@pytest.mark.asyncio
async def test_maintain_save_constraints_missing_rules():
    """save_constraints requires rules list."""
    from cairn.server.handlers import handle_cairn_maintain

    result = await handle_cairn_maintain({"action": "save_constraints"})
    assert result.get("isError") is True


@pytest.mark.asyncio
async def test_maintain_synthesize_insights():
    """synthesize_insights action doesn't error."""
    from cairn.server.handlers import handle_cairn_maintain

    with patch("cairn.bridge.synthesize_system_insights", return_value={"insights": [], "count": 0}):
        result = await handle_cairn_maintain({"action": "synthesize_insights"})
    assert not result.get("isError")


@pytest.mark.asyncio
async def test_maintain_backfill_embeddings():
    """backfill_embeddings action doesn't error."""
    from cairn.server.handlers import handle_cairn_maintain

    with patch("cairn.bridge.backfill_embeddings", return_value={"processed": 0, "skipped": 0}):
        result = await handle_cairn_maintain({"action": "backfill_embeddings"})
    assert not result.get("isError")
