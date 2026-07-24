"""Tests for cairn.llm provider abstraction."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_llm_clients():
    """Reset singleton LLM clients between tests."""
    from cairn.llm import reset_clients
    reset_clients()
    yield
    reset_clients()


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch):
    """Make tests hermetic against the developer's ambient CAIRN_LLM_* config
    (e.g. a local hookup exported in ~/.zshenv). Tests set what they need."""
    for v in ("CAIRN_LLM_PROVIDER", "CAIRN_LLM_MODEL_FAST", "CAIRN_LLM_MODEL_STANDARD",
              "CAIRN_LLM_BASE_URL", "CAIRN_LLM_API_KEY"):
        monkeypatch.delenv(v, raising=False)


class TestLlmComplete:
    """Test llm_complete() with mocked providers."""

    def test_anthropic_default_provider(self, monkeypatch):
        """Default provider is anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CAIRN_LLM_PROVIDER", raising=False)

        mock_content = MagicMock()
        mock_content.text = "extracted summary"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from cairn.llm import llm_complete
            result = llm_complete("hello", "system prompt", max_tokens=100)

        assert result == "extracted summary"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs.kwargs["max_tokens"] == 100

    def test_openai_provider(self, monkeypatch):
        """OpenAI provider uses openai SDK."""
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "openai response"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            result = llm_complete("hello", "system prompt", max_tokens=100)

        assert result == "openai response"

    def test_openai_compat_provider(self, monkeypatch):
        """openai_compat provider uses openai SDK with custom base_url."""
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "openai_compat")
        monkeypatch.setenv("CAIRN_LLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("CAIRN_LLM_API_KEY", "local-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "vllm response"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            result = llm_complete("hello", "system prompt")

        assert result == "vllm response"
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:8000/v1"

    def test_returns_empty_on_missing_api_key(self, monkeypatch):
        """Returns empty string when API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CAIRN_LLM_PROVIDER", raising=False)

        from cairn.llm import llm_complete
        result = llm_complete("hello", "system prompt")
        assert result == ""

    def test_returns_empty_on_api_error(self, monkeypatch):
        """Returns empty string on API error."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CAIRN_LLM_PROVIDER", raising=False)

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception("timeout")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from cairn.llm import llm_complete
            result = llm_complete("hello", "system prompt")

        assert result == ""

    def test_model_tier_standard(self, monkeypatch):
        """model_tier='standard' maps to Sonnet."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("CAIRN_LLM_PROVIDER", raising=False)

        mock_content = MagicMock()
        mock_content.text = "sonnet response"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from cairn.llm import llm_complete
            llm_complete("hello", "system prompt", model_tier="standard")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"

    def test_unknown_provider_returns_empty(self, monkeypatch):
        """Unknown provider returns empty string."""
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "unknown_provider")

        from cairn.llm import llm_complete
        result = llm_complete("hello", "system prompt")
        assert result == ""


class TestGetApiKey:
    """Test API key resolution."""

    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-123")
        from cairn.llm import _get_api_key
        assert _get_api_key("anthropic") == "ak-123"

    def test_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        from cairn.llm import _get_api_key
        assert _get_api_key("openai") == "test-openai-key"

    def test_compat_key(self, monkeypatch):
        monkeypatch.setenv("CAIRN_LLM_API_KEY", "local-key")
        from cairn.llm import _get_api_key
        assert _get_api_key("openai_compat") == "local-key"

    def test_compat_defaults_to_none_string(self, monkeypatch):
        monkeypatch.delenv("CAIRN_LLM_API_KEY", raising=False)
        from cairn.llm import _get_api_key
        assert _get_api_key("openai_compat") == "none"

    def test_missing_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from cairn.llm import _get_api_key
        assert _get_api_key("anthropic") == ""


def _mock_openai_capture():
    """Return (patched sys.modules dict, the OpenAI mock) capturing kwargs."""
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    return mock_openai, mock_client


class TestRegistryProviders:
    """Named OpenAI-compatible providers route with the right base_url + key."""

    def test_gemini_base_url_and_key(self, monkeypatch):
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-1")
        monkeypatch.delenv("CAIRN_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("CAIRN_LLM_MODEL_FAST", raising=False)
        mock_openai, _ = _mock_openai_capture()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            assert llm_complete("hi", "sys") == "ok"
        kw = mock_openai.OpenAI.call_args.kwargs
        assert kw["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert kw["api_key"] == "gk-1"
        model = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs["model"]
        assert model == "gemini-2.5-flash-lite"

    def test_generic_key_fallback_for_named_provider(self, monkeypatch):
        """A named provider works with only the generic CAIRN_LLM_API_KEY set."""
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setenv("CAIRN_LLM_API_KEY", "generic-key")
        mock_openai, _ = _mock_openai_capture()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            assert llm_complete("hi", "sys") == "ok"
        kw = mock_openai.OpenAI.call_args.kwargs
        assert kw["base_url"] == "https://api.groq.com/openai/v1"
        assert kw["api_key"] == "generic-key"

    def test_model_override_on_named_provider(self, monkeypatch):
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "deepinfra")
        monkeypatch.setenv("DEEPINFRA_API_KEY", "dk-1")
        monkeypatch.setenv("CAIRN_LLM_MODEL_FAST", "Qwen/Qwen2.5-7B-Instruct")
        mock_openai, _ = _mock_openai_capture()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            llm_complete("hi", "sys")
        model = mock_openai.OpenAI.return_value.chat.completions.create.call_args.kwargs["model"]
        assert model == "Qwen/Qwen2.5-7B-Instruct"

    def test_base_url_override_wins(self, monkeypatch):
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "gk-1")
        monkeypatch.setenv("CAIRN_LLM_BASE_URL", "http://proxy.internal/v1")
        mock_openai, _ = _mock_openai_capture()
        with patch.dict("sys.modules", {"openai": mock_openai}):
            from cairn.llm import llm_complete
            llm_complete("hi", "sys")
        assert mock_openai.OpenAI.call_args.kwargs["base_url"] == "http://proxy.internal/v1"

    def test_ollama_keyless(self, monkeypatch):
        monkeypatch.setenv("CAIRN_LLM_PROVIDER", "ollama")
        monkeypatch.delenv("CAIRN_LLM_API_KEY", raising=False)
        from cairn.llm import _get_api_key
        assert _get_api_key("ollama") == "ollama"  # harmless placeholder

    def test_list_providers_covers_majors(self):
        from cairn.llm import list_providers
        names = set(list_providers())
        assert {"anthropic", "openai", "gemini", "mistral", "deepinfra",
                "groq", "openrouter", "openai_compat"} <= names


def test_usage_capture_openai(monkeypatch):
    """get_last_usage() reflects the last completion's token counts."""
    monkeypatch.setenv("CAIRN_LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    mock_choice = MagicMock()
    mock_choice.message.content = "2"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 123
    mock_response.usage.completion_tokens = 4
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai = MagicMock()
    mock_openai.OpenAI.return_value = mock_client
    with patch.dict("sys.modules", {"openai": mock_openai}):
        from cairn.llm import get_last_usage, llm_complete, reset_usage
        reset_usage()
        llm_complete("hi", "sys")
        u = get_last_usage()
    assert u["input_tokens"] == 123 and u["output_tokens"] == 4


def test_gemini_key_alias_google(monkeypatch):
    """gemini honors GOOGLE_API_KEY when GEMINI_API_KEY is unset."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("CAIRN_LLM_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog-1")
    from cairn.llm import _get_api_key
    assert _get_api_key("gemini") == "goog-1"
