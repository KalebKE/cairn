"""Tests for trajectory-to-skill distillation."""

import json
from unittest.mock import patch, MagicMock

import pytest


def test_quality_gate_skips_short_sessions():
    """Sessions with fewer than 3 memories are not distilled."""
    from cairn.bridge import distill_trajectory

    with patch("cairn.bridge._get_store") as mock_store:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "error found", "event_type": "error_pattern", "metadata": {}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-short")
        assert result is None


def test_quality_gate_skips_no_completion():
    """Sessions without task_completion or commit are not distilled."""
    from cairn.bridge import distill_trajectory

    with patch("cairn.bridge._get_store") as mock_store:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "explored codebase", "event_type": "decision", "metadata": {}},
            {"content": "read some files", "event_type": "decision", "metadata": {}},
            {"content": "interesting finding", "event_type": "lesson_learned", "metadata": {}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-no-completion")
        assert result is None


def test_quality_gate_passes_with_completion():
    """Sessions with task_completion and 3+ memories pass the gate."""
    from cairn.bridge import distill_trajectory

    mock_llm_response = json.dumps({
        "skill_type": "debugging",
        "summary": "Debug null check bug in auth module",
        "steps": ["detect_error", "read_context", "apply_fix", "verify", "commit"],
        "key_insight": "Always validate optional fields",
        "tools_used": ["Grep", "Read", "Edit", "Bash"],
        "files_involved": ["auth.py"],
        "outcome": "success",
    })

    with patch("cairn.bridge._get_store") as mock_store, \
         patch("cairn.bridge.llm_complete", return_value=mock_llm_response) as mock_llm, \
         patch("cairn.bridge.auto_capture", return_value="node-123") as mock_capture:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "TypeError in auth.py", "event_type": "error_pattern", "metadata": {}},
            {"content": "Root cause: missing null check", "event_type": "decision", "metadata": {}},
            {"content": "Committed fix abc123", "event_type": "task_completion", "metadata": {"commit": "abc123"}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-ok")
        assert result is not None
        mock_llm.assert_called_once()
        mock_capture.assert_called_once()
        # Verify stored as skill_template
        call_kwargs = mock_capture.call_args
        assert call_kwargs[1]["event_type"] == "skill_template"


def test_quality_gate_passes_with_commit_in_metadata():
    """Sessions with a commit in metadata (no explicit task_completion type) pass."""
    from cairn.bridge import distill_trajectory

    mock_llm_response = json.dumps({
        "skill_type": "feature",
        "summary": "Add endpoint",
        "steps": ["scaffold", "implement", "test", "commit"],
        "key_insight": "Wire middleware first",
        "tools_used": ["Write", "Bash"],
        "files_involved": ["routes.py"],
        "outcome": "success",
    })

    with patch("cairn.bridge._get_store") as mock_store, \
         patch("cairn.bridge.llm_complete", return_value=mock_llm_response), \
         patch("cairn.bridge.auto_capture", return_value="node-456"):
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "Design decision", "event_type": "decision", "metadata": {}},
            {"content": "Wrote handler", "event_type": "decision", "metadata": {}},
            {"content": "Committed abc", "event_type": "decision", "metadata": {"commit": "abc"}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-commit")
        assert result is not None


def test_llm_failure_returns_none():
    """LLM failure is fail-open — returns None, no skill stored."""
    from cairn.bridge import distill_trajectory

    with patch("cairn.bridge._get_store") as mock_store, \
         patch("cairn.bridge.llm_complete", return_value="") as mock_llm, \
         patch("cairn.bridge.auto_capture") as mock_capture:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "error", "event_type": "error_pattern", "metadata": {}},
            {"content": "fix", "event_type": "decision", "metadata": {}},
            {"content": "done", "event_type": "task_completion", "metadata": {}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-llm-fail")
        assert result is None
        mock_capture.assert_not_called()


def test_llm_skip_response_returns_none():
    """LLM returning skip:true means session is too routine."""
    from cairn.bridge import distill_trajectory

    with patch("cairn.bridge._get_store") as mock_store, \
         patch("cairn.bridge.llm_complete", return_value='{"skip": true}'), \
         patch("cairn.bridge.auto_capture") as mock_capture:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "small fix", "event_type": "decision", "metadata": {}},
            {"content": "committed", "event_type": "task_completion", "metadata": {}},
            {"content": "done", "event_type": "task_completion", "metadata": {}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-skip")
        assert result is None
        mock_capture.assert_not_called()


def test_malformed_json_returns_none():
    """Malformed LLM JSON is fail-open."""
    from cairn.bridge import distill_trajectory

    with patch("cairn.bridge._get_store") as mock_store, \
         patch("cairn.bridge.llm_complete", return_value="not json at all"), \
         patch("cairn.bridge.auto_capture") as mock_capture:
        mock_db = MagicMock()
        mock_db.get_by_session.return_value = [
            {"content": "e", "event_type": "error_pattern", "metadata": {}},
            {"content": "d", "event_type": "decision", "metadata": {}},
            {"content": "t", "event_type": "task_completion", "metadata": {}},
        ]
        mock_store.return_value = mock_db

        result = distill_trajectory("test-session-bad-json")
        assert result is None
        mock_capture.assert_not_called()
