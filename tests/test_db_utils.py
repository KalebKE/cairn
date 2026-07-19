"""Tests for cairn.db_utils retry logic."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from cairn.db_utils import DB_RETRY_ATTEMPTS, retry_on_locked, retry_write_on_locked


class TestRetryOnLocked:
    def test_succeeds_first_try(self):
        fn = MagicMock(return_value=42)
        assert retry_on_locked(fn, "a", b="c") == 42
        fn.assert_called_once_with("a", b="c")

    @patch("cairn.db_utils.time.sleep")
    def test_retries_on_locked_error(self, mock_sleep):
        fn = MagicMock(
            side_effect=[
                sqlite3.OperationalError("database is locked"),
                "ok",
            ]
        )
        assert retry_on_locked(fn) == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once()

    def test_raises_non_locked_error(self):
        fn = MagicMock(side_effect=sqlite3.OperationalError("no such table"))
        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            retry_on_locked(fn)
        fn.assert_called_once()

    @patch("cairn.db_utils.time.sleep")
    def test_exhausts_attempts(self, mock_sleep):
        fn = MagicMock(
            side_effect=sqlite3.OperationalError("database is locked")
        )
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            retry_on_locked(fn)
        assert fn.call_count == DB_RETRY_ATTEMPTS
        assert mock_sleep.call_count == DB_RETRY_ATTEMPTS - 1


class TestRetryWriteOnLocked:
    def test_succeeds_first_try(self):
        conn = MagicMock()
        fn = MagicMock(return_value="result")
        assert retry_write_on_locked(conn, fn, "x") == "result"
        conn.execute.assert_called_once_with("BEGIN IMMEDIATE")
        conn.commit.assert_called_once()
        fn.assert_called_once_with("x")

    @patch("cairn.db_utils.time.sleep")
    def test_retries_on_locked(self, mock_sleep):
        conn = MagicMock()
        conn.execute.side_effect = [
            sqlite3.OperationalError("database is locked"),
            None,  # BEGIN IMMEDIATE succeeds on retry
        ]
        fn = MagicMock(return_value="ok")
        assert retry_write_on_locked(conn, fn) == "ok"
        assert conn.execute.call_count == 2
        conn.rollback.assert_called_once()
        mock_sleep.assert_called_once()

    @patch("cairn.db_utils.time.sleep")
    def test_rollback_on_final_failure(self, mock_sleep):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        fn = MagicMock()
        with pytest.raises(sqlite3.OperationalError):
            retry_write_on_locked(conn, fn)
        # rollback called on every attempt
        assert conn.rollback.call_count == DB_RETRY_ATTEMPTS

    @patch("cairn.db_utils.time.sleep")
    def test_rollback_itself_fails(self, mock_sleep):
        conn = MagicMock()
        conn.execute.side_effect = sqlite3.OperationalError("database is locked")
        conn.rollback.side_effect = Exception("rollback broken")
        fn = MagicMock()
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            retry_write_on_locked(conn, fn)
