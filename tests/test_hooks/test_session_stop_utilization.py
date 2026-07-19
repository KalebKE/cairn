"""Test session stop utilization report."""
import pytest


def test_utilization_report_flags_missing_tools():
    """When agent never called cairn_reflect or cairn_decision_query,
    the report should flag them as unused."""
    from hooks.session_stop import _build_utilization_report

    # Simulate a session that called some tools but skipped critical ones
    tool_calls = [
        "cairn_welcome", "cairn_protocol", "cairn_query", "cairn_store",
        "Read", "Edit", "Bash", "cairn_query", "cairn_store",
    ]
    report = _build_utilization_report(tool_calls)

    assert "cairn_reflect" in report["missed"]
    assert "cairn_decision_query" in report["missed"]
    assert report["score"] < 100  # Not a perfect score


def test_utilization_report_perfect_score():
    """When all critical tools were called, score is 100."""
    from hooks.session_stop import _build_utilization_report

    tool_calls = [
        "cairn_welcome", "cairn_protocol", "cairn_query", "cairn_store",
        "cairn_reflect", "cairn_decision_query", "cairn_file_check",
        "cairn_checkpoint", "cairn_coord_status",
    ]
    report = _build_utilization_report(tool_calls)

    assert len(report["missed"]) == 0
    assert report["score"] == 100


def test_utilization_report_empty_session():
    """An empty session should flag all critical tools."""
    from hooks.session_stop import _build_utilization_report

    report = _build_utilization_report([])
    assert report["score"] == 0
    assert len(report["missed"]) > 0
