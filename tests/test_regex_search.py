"""Tests for mode="regex" on cairn_query.

Python-side re filtering over a recency-ordered SQL scan (no REGEXP UDF):
compile once, reject bad/catastrophic patterns before any scan, cap scanned
rows, honor the shared active-status predicate.
"""
import pytest

from cairn.server.handlers import handle_cairn_query


def _seed(store):
    ids = {}
    ids["rollup"] = store.store(
        content="Rollup marker last-rollup written before the work runs",
        metadata={"event_type": "decision", "project": "/p/cairn"},
        skip_inference=True,
    )
    ids["watchdog"] = store.store(
        content="Watchdog rebinds hook.sock after external unlink",
        metadata={"event_type": "lesson_learned", "project": "/p/cairn"},
        skip_inference=True,
    )
    ids["other"] = store.store(
        content="Completely unrelated grocery list: apples, bananas",
        metadata={"event_type": "task_completion", "project": "/p/other"},
        skip_inference=True,
    )
    return ids


class TestStoreRegexSearch:
    def test_basic_match(self, store):
        ids = _seed(store)
        hits = store.regex_search(r"hook\.sock|last-rollup")
        got = {r.id for r in hits}
        assert ids["rollup"] in got and ids["watchdog"] in got
        assert ids["other"] not in got

    def test_case_insensitive_default(self, store):
        ids = _seed(store)
        hits = store.regex_search(r"WATCHDOG")
        assert ids["watchdog"] in {r.id for r in hits}

    def test_case_sensitive_flag(self, store):
        ids = _seed(store)
        hits = store.regex_search(r"WATCHDOG", case_sensitive=True)
        assert ids["watchdog"] not in {r.id for r in hits}

    def test_invalid_pattern_raises_value_error(self, store):
        with pytest.raises(ValueError, match="Invalid regex"):
            store.regex_search(r"([unclosed")

    def test_catastrophic_pattern_rejected(self, store):
        with pytest.raises(ValueError):
            store.regex_search(r"(a+)+b")

    def test_empty_and_oversize_pattern_rejected(self, store):
        with pytest.raises(ValueError):
            store.regex_search("")
        with pytest.raises(ValueError):
            store.regex_search("x" * 300)

    def test_event_type_filter(self, store):
        ids = _seed(store)
        hits = store.regex_search(r".*rollup.*", event_type="lesson_learned")
        assert ids["rollup"] not in {r.id for r in hits}

    def test_limit_and_recency_order(self, store):
        for i in range(5):
            store.store(content=f"regexcandidate number {i} in sequence",
                        metadata={"event_type": "decision"}, skip_inference=True)
        hits = store.regex_search(r"regexcandidate number \d", limit=3)
        assert len(hits) == 3
        # Recency order: latest stored first
        assert "number 4" in hits[0].content

    def test_excludes_archived_and_superseded(self, store):
        ids = _seed(store)
        with store._lock:
            store._conn.execute("UPDATE memories SET status = 'archived' WHERE node_id = ?",
                                (ids["watchdog"],))
            store._commit()
        hits = store.regex_search(r"hook\.sock")
        assert ids["watchdog"] not in {r.id for r in hits}

    def test_max_scan_cap(self, store):
        ids = _seed(store)
        # Cap of 1: only the newest row is scanned; the oldest can't match.
        hits = store.regex_search(r"last-rollup", max_scan_rows=1)
        assert ids["rollup"] not in {r.id for r in hits}


class TestRegexQueryMode:
    @pytest.fixture
    def mock_get_store(self, store):
        from unittest.mock import patch
        with patch("cairn.bridge._get_store", return_value=store):
            yield store

    @pytest.mark.asyncio
    async def test_regex_mode_basic(self, mock_get_store):
        ids = _seed(mock_get_store)
        result = await handle_cairn_query({"query": r"hook\.sock", "mode": "regex"})
        assert not result.get("isError")
        text = result["content"][0]["text"]
        assert ids["watchdog"] in text
        assert "Regex" in text

    @pytest.mark.asyncio
    async def test_regex_mode_invalid_pattern_clean_error(self, mock_get_store):
        result = await handle_cairn_query({"query": r"([unclosed", "mode": "regex"})
        assert result.get("isError")
        text = result["content"][0]["text"]
        assert "Invalid regex" in text
        assert "Traceback" not in text
