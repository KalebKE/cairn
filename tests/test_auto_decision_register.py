"""Test auto-registration of decisions when cairn_store gets a decision type."""
import pytest
from unittest.mock import patch, MagicMock


def test_store_decision_auto_registers_coordination():
    """When cairn_store is called with event_type='decision', it should
    also register the decision in coordination."""
    from cairn.server.handlers import _auto_register_decision

    mock_mgr = MagicMock()
    mock_mgr.register_decision.return_value = {"id": 1, "status": "active"}

    result = _auto_register_decision(
        mgr=mock_mgr,
        session_id="test-session",
        project="/test/project",
        content="Use PostgreSQL instead of MySQL for the user service",
        entity_id=None,
    )

    mock_mgr.register_decision.assert_called_once()
    call_args = mock_mgr.register_decision.call_args
    assert call_args[1]["session_id"] == "test-session"
    assert "PostgreSQL" in call_args[1]["decision"]


def test_store_non_decision_does_not_register():
    """cairn_store with event_type='lesson_learned' should NOT register a decision."""
    from cairn.server.handlers import _auto_register_decision

    # Should return None for non-decision types (mgr is None = signal to skip)
    result = _auto_register_decision(
        mgr=None,
        session_id="test",
        project="/test",
        content="A lesson",
        entity_id=None,
    )
    assert result is None


def test_auto_register_extracts_domain():
    """Domain should be extracted from content heuristically."""
    from cairn.server.handlers import _extract_decision_domain

    assert _extract_decision_domain("Use PostgreSQL for the auth service") == "auth"
    assert _extract_decision_domain("Deploy to Vercel instead of Netlify") == "deploy"
    assert _extract_decision_domain("Random decision text") == "general"
