"""Tests for cairn_consult_gpt tool: handler + gpt_complete function."""

import os
from unittest.mock import patch

import pytest

from cairn.server.tool_schemas import TOOL_SCHEMAS
from cairn.server.handlers import HANDLERS


# ---------------------------------------------------------------------------
# Schema & handler registration
# ---------------------------------------------------------------------------


def test_schema_registered():
    names = [s["name"] for s in TOOL_SCHEMAS]
    assert "cairn_consult_gpt" in names


def test_handler_registered():
    assert "cairn_consult_gpt" in HANDLERS


# ---------------------------------------------------------------------------
# Handler tests (mock gpt_complete)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_prompt():
    handler = HANDLERS["cairn_consult_gpt"]
    result = await handler({})
    assert result["isError"] is True
    assert "prompt" in result["content"][0]["text"].lower()


@pytest.mark.asyncio
async def test_empty_prompt():
    handler = HANDLERS["cairn_consult_gpt"]
    result = await handler({"prompt": "   "})
    assert result["isError"] is True


@pytest.mark.asyncio
async def test_successful_consultation():
    handler = HANDLERS["cairn_consult_gpt"]
    with patch("cairn.llm.gpt_complete", return_value="Use connection pooling."):
        result = await handler({"prompt": "How to optimize DB connections?"})
    assert "isError" not in result
    text = result["content"][0]["text"]
    assert "## GPT Consultation" in text
    assert "Use connection pooling." in text


@pytest.mark.asyncio
async def test_context_concatenation():
    handler = HANDLERS["cairn_consult_gpt"]
    with patch("cairn.llm.gpt_complete", return_value="Fix the index.") as mock:
        await handler({"prompt": "Why is this slow?", "context": "SELECT * FROM big_table"})
    call_args = mock.call_args
    full_prompt = call_args[0][0]
    assert "Why is this slow?" in full_prompt
    assert "--- Context ---" in full_prompt
    assert "SELECT * FROM big_table" in full_prompt


@pytest.mark.asyncio
async def test_context_not_added_when_empty():
    handler = HANDLERS["cairn_consult_gpt"]
    with patch("cairn.llm.gpt_complete", return_value="Answer.") as mock:
        await handler({"prompt": "Simple question"})
    full_prompt = mock.call_args[0][0]
    assert "--- Context ---" not in full_prompt


@pytest.mark.asyncio
async def test_empty_response_is_error():
    handler = HANDLERS["cairn_consult_gpt"]
    with patch("cairn.llm.gpt_complete", return_value=""):
        result = await handler({"prompt": "Test prompt"})
    assert result["isError"] is True
    assert "OPENAI_API_KEY" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_optional_params_forwarded():
    handler = HANDLERS["cairn_consult_gpt"]
    with patch("cairn.llm.gpt_complete", return_value="Response.") as mock:
        await handler({
            "prompt": "Test",
            "system": "You are a crypto expert.",
            "temperature": 0.2,
            "max_tokens": 8192,
        })
    kwargs = mock.call_args[1]
    assert kwargs["system"] == "You are a crypto expert."
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# gpt_complete unit tests (mock _complete_openai)
# ---------------------------------------------------------------------------


def test_gpt_complete_defaults():
    from cairn.llm import gpt_complete

    with patch("cairn.llm._complete_openai", return_value="GPT says hi") as mock:
        result = gpt_complete("Hello")
    assert result == "GPT says hi"
    mock.assert_called_once()
    _, kwargs = mock.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.7
    assert kwargs["timeout"] == 120.0


def test_gpt_complete_env_var_model():
    from cairn.llm import gpt_complete

    with patch.dict(os.environ, {"CAIRN_GPT_MODEL": "gpt-5.3"}):
        with patch("cairn.llm._complete_openai", return_value="ok") as mock:
            gpt_complete("Test")
    assert mock.call_args[1]["model"] == "gpt-5.3"


def test_gpt_complete_explicit_model_overrides_env():
    from cairn.llm import gpt_complete

    with patch.dict(os.environ, {"CAIRN_GPT_MODEL": "gpt-5.3"}):
        with patch("cairn.llm._complete_openai", return_value="ok") as mock:
            gpt_complete("Test", model="gpt-4-turbo")
    assert mock.call_args[1]["model"] == "gpt-4-turbo"


def test_gpt_complete_no_api_key():
    from cairn.llm import gpt_complete

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=False):
        with patch("cairn.llm._complete_openai", return_value="") as mock:
            result = gpt_complete("Test")
    assert result == ""


def test_gpt_complete_exception_returns_empty():
    from cairn.llm import gpt_complete

    with patch("cairn.llm._complete_openai", side_effect=Exception("API down")):
        result = gpt_complete("Test")
    assert result == ""


def test_gpt_complete_custom_system():
    from cairn.llm import gpt_complete

    with patch("cairn.llm._complete_openai", return_value="ok") as mock:
        gpt_complete("Test", system="You are a math tutor.")
    assert mock.call_args[0][1] == "You are a math tutor."
