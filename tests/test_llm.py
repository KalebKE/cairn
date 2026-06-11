"""Tests for omega.llm provider abstraction."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_llm_clients():
    """Reset singleton LLM clients between tests."""
    from omega.llm import reset_clients
    reset_clients()
    yield
    reset_clients()


class TestLlmComplete:
    """Test llm_complete() with mocked providers."""

    def test_anthropic_default_provider(self, monkeypatch):
        """Default provider is anthropic."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OMEGA_LLM_PROVIDER", raising=False)

        mock_content = MagicMock()
        mock_content.text = "extracted summary"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from omega.llm import llm_complete
            result = llm_complete("hello", "system prompt", max_tokens=100)

        assert result == "extracted summary"
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-haiku-4-5-20251001"
        assert call_kwargs.kwargs["max_tokens"] == 100

    def test_openai_provider(self, monkeypatch):
        """OpenAI provider uses openai SDK."""
        monkeypatch.setenv("OMEGA_LLM_PROVIDER", "openai")
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
            from omega.llm import llm_complete
            result = llm_complete("hello", "system prompt", max_tokens=100)

        assert result == "openai response"

    def test_openai_compat_provider(self, monkeypatch):
        """openai_compat provider uses openai SDK with custom base_url."""
        monkeypatch.setenv("OMEGA_LLM_PROVIDER", "openai_compat")
        monkeypatch.setenv("OMEGA_LLM_BASE_URL", "http://localhost:8000/v1")
        monkeypatch.setenv("OMEGA_LLM_API_KEY", "local-key")

        mock_choice = MagicMock()
        mock_choice.message.content = "vllm response"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = MagicMock()
        mock_openai.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai}):
            from omega.llm import llm_complete
            result = llm_complete("hello", "system prompt")

        assert result == "vllm response"
        mock_openai.OpenAI.assert_called_once()
        call_kwargs = mock_openai.OpenAI.call_args
        assert call_kwargs.kwargs["base_url"] == "http://localhost:8000/v1"

    def test_returns_empty_on_missing_api_key(self, monkeypatch):
        """Returns empty string when API key is missing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OMEGA_LLM_PROVIDER", raising=False)

        from omega.llm import llm_complete
        result = llm_complete("hello", "system prompt")
        assert result == ""

    def test_returns_empty_on_api_error(self, monkeypatch):
        """Returns empty string on API error."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OMEGA_LLM_PROVIDER", raising=False)

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value.messages.create.side_effect = Exception("timeout")

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from omega.llm import llm_complete
            result = llm_complete("hello", "system prompt")

        assert result == ""

    def test_model_tier_standard(self, monkeypatch):
        """model_tier='standard' maps to Sonnet."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        monkeypatch.delenv("OMEGA_LLM_PROVIDER", raising=False)

        mock_content = MagicMock()
        mock_content.text = "sonnet response"

        mock_response = MagicMock()
        mock_response.content = [mock_content]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            from omega.llm import llm_complete
            llm_complete("hello", "system prompt", model_tier="standard")

        call_kwargs = mock_client.messages.create.call_args
        assert call_kwargs.kwargs["model"] == "claude-sonnet-4-6"

    def test_unknown_provider_returns_empty(self, monkeypatch):
        """Unknown provider returns empty string."""
        monkeypatch.setenv("OMEGA_LLM_PROVIDER", "unknown_provider")

        from omega.llm import llm_complete
        result = llm_complete("hello", "system prompt")
        assert result == ""


class TestGetApiKey:
    """Test API key resolution."""

    def test_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-123")
        from omega.llm import _get_api_key
        assert _get_api_key("anthropic") == "ak-123"

    def test_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        from omega.llm import _get_api_key
        assert _get_api_key("openai") == "test-openai-key"

    def test_compat_key(self, monkeypatch):
        monkeypatch.setenv("OMEGA_LLM_API_KEY", "local-key")
        from omega.llm import _get_api_key
        assert _get_api_key("openai_compat") == "local-key"

    def test_compat_defaults_to_none_string(self, monkeypatch):
        monkeypatch.delenv("OMEGA_LLM_API_KEY", raising=False)
        from omega.llm import _get_api_key
        assert _get_api_key("openai_compat") == "none"

    def test_missing_key_returns_empty(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from omega.llm import _get_api_key
        assert _get_api_key("anthropic") == ""
