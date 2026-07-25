"""Tests for cairn.task_utils text cleaning.

summarize_task_text (LLM-backed) and _basic_clean were removed 2026-07-25 —
no callers, and their removal keeps this module LLM-import-free. This test
module also asserts that property so a future re-introduction of a remote
call here fails loudly.
"""

from cairn.task_utils import clean_task_text


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


class TestNoLLMImports:
    def test_module_has_no_llm_dependency(self):
        """task_utils must stay free of remote-LLM imports (status-bar path)."""
        import cairn.task_utils as tu

        assert not hasattr(tu, "llm_complete")
        assert not hasattr(tu, "summarize_task_text")
