"""Tests for cairn.milestones module."""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from cairn import json_compat as json


@pytest.fixture(autouse=True)
def _patch_milestones_dir(tmp_cairn_dir, monkeypatch):
    """Point milestones module at the temp CAIRN_HOME."""
    import cairn.milestones as mod
    monkeypatch.setattr(mod, "CAIRN_HOME", tmp_cairn_dir)
    monkeypatch.setattr(mod, "MILESTONES_DIR", tmp_cairn_dir / "milestones")


class TestCheckMilestone:
    def test_first_time_returns_true(self):
        from cairn.milestones import _check_milestone

        assert _check_milestone("test-milestone") is True

    def test_already_achieved_returns_false(self):
        from cairn.milestones import _check_milestone

        _check_milestone("test-milestone")
        assert _check_milestone("test-milestone") is False

    def test_creates_marker_with_json_metadata(self, tmp_cairn_dir):
        from cairn.milestones import _check_milestone

        _check_milestone("json-test")
        marker = tmp_cairn_dir / "milestones" / "json-test"
        assert marker.exists()
        data = json.loads(marker.read_text())
        assert data["name"] == "json-test"
        assert "achieved_at" in data

    def test_different_milestones_independent(self):
        from cairn.milestones import _check_milestone

        assert _check_milestone("a") is True
        assert _check_milestone("b") is True
        assert _check_milestone("a") is False


class TestCaptureThresholds:
    def test_threshold_1(self):
        from cairn.milestones import check_capture_milestones

        msg = check_capture_milestones(1)
        assert msg is not None
        assert "First memory" in msg

    def test_threshold_10(self):
        from cairn.milestones import check_capture_milestones

        msg = check_capture_milestones(10)
        assert msg is not None
        assert "10" in msg

    def test_threshold_50(self):
        from cairn.milestones import check_capture_milestones

        msg = check_capture_milestones(50)
        assert msg is not None
        assert "50" in msg

    def test_threshold_100(self):
        from cairn.milestones import check_capture_milestones

        msg = check_capture_milestones(100)
        assert msg is not None
        assert "100" in msg

    @pytest.mark.parametrize("count", [500, 1000, 1500, 2000])
    def test_high_thresholds_are_neutral_messages(self, count):
        """High-count milestones celebrate without upgrade CTAs (fork: no Pro)."""
        from cairn.milestones import check_capture_milestones

        msg = check_capture_milestones(count)
        assert msg is not None
        assert f"{count:,}" in msg
        assert "tracqi.com" not in msg
        assert "Upgrade" not in msg

    def test_no_milestone_below_first_threshold(self):
        from cairn.milestones import check_capture_milestones

        assert check_capture_milestones(0) is None

    def test_no_milestone_after_already_achieved(self):
        from cairn.milestones import check_capture_milestones

        # First call at 7 triggers capture-1 (highest crossed threshold)
        assert check_capture_milestones(7) is not None
        # Subsequent calls below next threshold: already achieved
        assert check_capture_milestones(7) is None
        assert check_capture_milestones(9) is None

    def test_milestone_only_triggers_once(self):
        from cairn.milestones import check_capture_milestones

        assert check_capture_milestones(1) is not None
        assert check_capture_milestones(1) is None  # Already achieved


class TestStreak:
    def _make_mock_store(self, date_strings):
        """Create a mock store with given dates as memory days."""
        store = MagicMock()
        rows = [(d,) for d in date_strings]
        store._conn.execute.return_value.fetchall.return_value = rows
        return store

    def test_no_activity(self):
        from cairn.milestones import get_streak

        store = self._make_mock_store([])
        result = get_streak(store)
        assert result == {"current": 0, "longest": 0, "today_active": False}

    def test_today_only(self):
        from cairn.milestones import get_streak

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        store = self._make_mock_store([today])
        result = get_streak(store)
        assert result["current"] == 1
        assert result["longest"] == 1
        assert result["today_active"] is True

    def test_consecutive_days(self):
        from cairn.milestones import get_streak

        today = datetime.now(timezone.utc)
        days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(5)]
        store = self._make_mock_store(days)
        result = get_streak(store)
        assert result["current"] == 5
        assert result["longest"] == 5
        assert result["today_active"] is True

    def test_gap_resets_current(self):
        from cairn.milestones import get_streak

        today = datetime.now(timezone.utc)
        # Today, yesterday, then gap, then 3 more days
        days = [
            today.strftime("%Y-%m-%d"),
            (today - timedelta(days=1)).strftime("%Y-%m-%d"),
            # gap at day 2
            (today - timedelta(days=3)).strftime("%Y-%m-%d"),
            (today - timedelta(days=4)).strftime("%Y-%m-%d"),
            (today - timedelta(days=5)).strftime("%Y-%m-%d"),
        ]
        store = self._make_mock_store(days)
        result = get_streak(store)
        assert result["current"] == 2  # today + yesterday
        assert result["longest"] == 3  # the 3-day block

    def test_old_streak_no_current(self):
        from cairn.milestones import get_streak

        today = datetime.now(timezone.utc)
        # Last activity was 5 days ago, 3 consecutive days
        days = [
            (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            (today - timedelta(days=6)).strftime("%Y-%m-%d"),
            (today - timedelta(days=7)).strftime("%Y-%m-%d"),
        ]
        store = self._make_mock_store(days)
        result = get_streak(store)
        assert result["current"] == 0  # Too old
        assert result["longest"] == 3
        assert result["today_active"] is False


class TestListMilestones:
    def test_empty_dir(self, tmp_cairn_dir):
        from cairn.milestones import list_milestones

        assert list_milestones() == []

    def test_lists_achieved_milestones(self):
        from cairn.milestones import _check_milestone, list_milestones

        _check_milestone("first-capture")
        _check_milestone("first-recall")

        milestones = list_milestones()
        assert len(milestones) == 2
        names = {m["name"] for m in milestones}
        assert "first-capture" in names
        assert "first-recall" in names
        for m in milestones:
            assert m["achieved_at"] != ""

    def test_handles_legacy_empty_markers(self, tmp_cairn_dir):
        from cairn.milestones import list_milestones

        markers_dir = tmp_cairn_dir / "milestones"
        markers_dir.mkdir(parents=True, exist_ok=True)
        (markers_dir / "legacy-marker").touch()

        milestones = list_milestones()
        assert len(milestones) == 1
        assert milestones[0]["name"] == "legacy-marker"
        assert milestones[0]["achieved_at"] != ""


class TestMilestonesHandler:
    @pytest.mark.asyncio
    async def test_handler_returns_formatted_response(self):
        from cairn.milestones import _check_milestone

        _check_milestone("test-handler")

        from cairn.server.handlers import handle_cairn_milestones

        # Mock the store for streak
        import cairn.bridge as bridge
        original_get_store = bridge._get_store

        mock_store = MagicMock()
        mock_store._conn.execute.return_value.fetchall.return_value = []
        bridge._get_store = lambda: mock_store

        try:
            result = await handle_cairn_milestones({})
            assert "isError" not in result
            text = result["content"][0]["text"]
            assert "Milestones" in text
            assert "Streak" in text
            assert "test-handler" in text
        finally:
            bridge._get_store = original_get_store

    @pytest.mark.asyncio
    async def test_stats_routing(self):
        from cairn.server.handlers import handle_cairn_stats

        import cairn.bridge as bridge
        original_get_store = bridge._get_store

        mock_store = MagicMock()
        mock_store._conn.execute.return_value.fetchall.return_value = []
        bridge._get_store = lambda: mock_store

        try:
            result = await handle_cairn_stats({"action": "milestones"})
            assert "isError" not in result
            text = result["content"][0]["text"]
            assert "Milestones" in text
        finally:
            bridge._get_store = original_get_store
