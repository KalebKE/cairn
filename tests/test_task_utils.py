"""Tests for cairn.task_utils text cleaning and summarization."""

import pytest
from unittest.mock import patch

from cairn.task_utils import _basic_clean, clean_task_text, summarize_task_text


class TestCleanTaskText:
    def test_empty_string(self):
        assert clean_task_text("") == ""

    def test_short_string(self):
        assert clean_task_text("hi") == ""

    def test_xml_tags_stripped(self):
        result = clean_task_text("<system-reminder>ignored</system-reminder>Fix the auth bug in login flow")
        assert "<" not in result
        assert "Fix" in result

    def test_skip_prefix_memory_handoff(self):
        assert clean_task_text("MEMORY HANDOFF data here") == ""

    def test_skip_prefix_implement_plan(self):
        assert clean_task_text("Implement the following plan: step 1, step 2") == ""

    def test_skip_exact_proceed(self):
        assert clean_task_text("proceed") == ""

    def test_skip_exact_lgtm(self):
        assert clean_task_text("lgtm") == ""

    def test_markdown_header_stripped(self):
        result = clean_task_text("## Add user authentication to the API")
        assert result.startswith("Add")
        assert "#" not in result

    def test_resume_prefix_stripped(self):
        result = clean_task_text("Resume: Fix the broken deployment pipeline")
        assert result.startswith("Fix")
        assert "Resume" not in result

    def test_first_line_only(self):
        result = clean_task_text("Add dark mode toggle\nThis should support both light and dark themes")
        assert "Add dark mode" in result
        assert "themes" not in result

    def test_sentence_split(self):
        result = clean_task_text("Fix the auth bug. Then update the tests for coverage")
        assert "auth bug" in result
        assert "tests" not in result

    def test_cap_at_60_chars(self):
        long_text = "Refactor the entire authentication subsystem to use the new OAuth provider configuration"
        result = clean_task_text(long_text)
        assert len(result) <= 60

    def test_normal_task(self):
        result = clean_task_text("Add a logout button to the navigation bar")
        assert "logout" in result.lower()


class TestBasicClean:
    def test_preserves_full_text(self):
        text = "Fix the auth bug in login flow and update tests"
        result = _basic_clean(text)
        assert "update tests" in result

    def test_strips_tags(self):
        result = _basic_clean("<tag>content</tag>Real task here")
        assert "<" not in result
        assert "Real task" in result

    def test_skip_prefixes_apply(self):
        assert _basic_clean("MEMORY HANDOFF data") == ""

    def test_empty(self):
        assert _basic_clean("") == ""


class TestSummarizeTaskText:
    def test_fallback_no_api_key(self):
        """Without ANTHROPIC_API_KEY, should fall back to clean_task_text."""
        result = summarize_task_text("Add a logout button to the navigation bar")
        assert result != ""
        assert "logout" in result.lower()

    def test_short_text_uses_clean(self):
        result = summarize_task_text("Fix the auth bug")
        assert result == clean_task_text("Fix the auth bug")

    def test_empty_returns_empty(self):
        assert summarize_task_text("") == ""

    def test_skip_prefix_returns_empty(self):
        assert summarize_task_text("MEMORY HANDOFF data here with enough chars") == ""

    def test_uses_llm_complete(self, monkeypatch):
        """summarize_task_text uses cairn.llm.llm_complete under the hood."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

        with patch("cairn.task_utils.llm_complete", return_value="fix auth token refresh") as mock_llm:
            result = summarize_task_text(
                "We need to fix the authentication token refresh bug that causes users to be logged out"
            )

        mock_llm.assert_called_once()
        assert result == "fix auth token refresh"
