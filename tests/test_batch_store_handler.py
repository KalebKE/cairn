"""Tests for batch store via cairn_store handler."""

import pytest
from unittest.mock import patch


@pytest.mark.asyncio
async def test_batch_store_multiple_items():
    """cairn_store(items=[...]) stores all items."""
    from cairn.server.handlers import handle_cairn_store

    items = [
        {"content": "memory one", "event_type": "decision"},
        {"content": "memory two", "event_type": "lesson_learned"},
    ]
    mock_result = {"ids": ["id1", "id2"], "count": 2}
    with patch("cairn.bridge.batch_store", return_value=mock_result):
        result = await handle_cairn_store({"items": items})
    text = result["content"][0]["text"]
    assert "id1" in text or "2" in text


@pytest.mark.asyncio
async def test_batch_store_empty_list():
    """cairn_store(items=[]) returns empty result, not error."""
    from cairn.server.handlers import handle_cairn_store

    result = await handle_cairn_store({"items": []})
    assert not result.get("isError")
    text = result["content"][0]["text"]
    assert "0" in text


@pytest.mark.asyncio
async def test_batch_store_invalid_type():
    """cairn_store(items="not a list") returns error."""
    from cairn.server.handlers import handle_cairn_store

    result = await handle_cairn_store({"items": "not a list"})
    assert result.get("isError") is True
