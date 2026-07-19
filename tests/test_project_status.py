"""Tests for the project_status event type and auto-generation."""
from unittest.mock import patch

from cairn.types import AutoCaptureEventType, EVENT_TYPE_TTL
from cairn.bridge import DEDUP_THRESHOLDS, EVOLUTION_TYPES


class TestProjectStatusType:
    """Verify project_status is registered in all constant tables."""

    def test_project_status_ttl_is_permanent(self):
        assert EVENT_TYPE_TTL[AutoCaptureEventType.PROJECT_STATUS] is None  # PERMANENT = None

    def test_project_status_in_dedup(self):
        assert AutoCaptureEventType.PROJECT_STATUS in DEDUP_THRESHOLDS
        assert DEDUP_THRESHOLDS[AutoCaptureEventType.PROJECT_STATUS] == 0.85

    def test_project_status_in_evolution(self):
        assert AutoCaptureEventType.PROJECT_STATUS in EVOLUTION_TYPES


class TestProjectStatusGeneration:
    """Verify _build_project_status() produces correct output."""

    def test_build_project_status_with_activity(self):
        from cairn.hooks.session_stop import _build_project_status

        mock_decisions = [{"content": "Use SQLite for storage"}]
        mock_tasks = [{"content": "Implemented auth module"}]

        with patch("cairn.bridge.query_structured") as mock_qs:
            def side_effect(query_text, **kwargs):
                if kwargs.get("event_type") == "decision":
                    return mock_decisions
                elif kwargs.get("event_type") == "task_completion":
                    return mock_tasks
                return []
            mock_qs.side_effect = side_effect

            result = _build_project_status("s1", "/proj/cairn")

        assert result is not None
        assert "Project: cairn" in result
        assert "Use SQLite" in result
        assert "auth module" in result

    def test_build_project_status_no_activity(self):
        from cairn.hooks.session_stop import _build_project_status

        with patch("cairn.bridge.query_structured", return_value=[]):
            result = _build_project_status("s1", "/proj/cairn")

        assert result is None

    def test_build_project_status_no_project(self):
        from cairn.hooks.session_stop import _build_project_status

        result = _build_project_status("s1", "")
        assert result is None

    def test_project_status_stored_at_session_stop(self, tmp_cairn_dir):
        """End session with activity, verify project_status is stored in memories."""
        from cairn.bridge import _get_store, reset_memory, auto_capture
        reset_memory()
        store = _get_store()

        mock_decisions = [{"content": "Use PostgreSQL for production"}]
        mock_tasks = [{"content": "Implemented auth module"}]

        def qs_side_effect(query_text, **kwargs):
            if kwargs.get("event_type") == "decision":
                return mock_decisions
            elif kwargs.get("event_type") == "task_completion":
                return mock_tasks
            return []

        from cairn.hooks.session_stop import _build_project_status
        with patch("cairn.bridge.query_structured", side_effect=qs_side_effect):
            status_text = _build_project_status("s1", "/proj/myapp")

        assert status_text is not None

        auto_capture(
            content=status_text,
            event_type="project_status",
            session_id="s1",
            project="/proj/myapp",
            metadata={"source": "session_stop_auto", "project": "/proj/myapp"},
        )

        nodes = store.get_by_type("project_status", limit=10)
        project_nodes = [
            n for n in nodes
            if (n.metadata or {}).get("project") == "/proj/myapp"
        ]
        assert len(project_nodes) >= 1
        assert "myapp" in project_nodes[0].content
        reset_memory()


class TestProjectStatusSurfacing:
    """Verify welcome() surfaces project_status."""

    def test_welcome_shows_project_status(self, tmp_cairn_dir):
        from cairn.bridge import welcome, auto_capture, reset_memory
        reset_memory()

        auto_capture(
            content="Project: cairn | Key decisions: Use SQLite; Add caching",
            event_type="project_status",
            session_id="s1",
            project="/proj/cairn",
            metadata={"source": "session_stop_auto", "project": "/proj/cairn"},
        )

        result = welcome(project="/proj/cairn")
        obs = result.get("observation_prefix", "")
        ctx = result.get("project_context", "")
        combined = obs + ctx

        assert "Project Status" in combined
        reset_memory()
