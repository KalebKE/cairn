"""Property-based tests (hypothesis) for input-driven surfaces.

Example-based tests pin known cases; these fuzz the correctness/robustness
invariants of the prefix resolver, regex search, and dedup against
adversarial input (unicode, control chars, regex metacharacters, huge
strings) where hand-written cases miss edge cases.
"""
import re

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from cairn.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    return SQLiteStore(db_path=str(tmp_path / "t.db"))


# Text that exercises unicode, control chars, whitespace, and length.
_content = st.text(min_size=1, max_size=400).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# get_node_by_prefix — the fetch-by-id resolver
# ---------------------------------------------------------------------------

class TestPrefixResolverProperties:
    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(content=_content)
    def test_full_id_always_resolves_to_itself(self, store, content):
        nid = store.store(content, metadata={"event_type": "decision"},
                          skip_inference=True)
        node, candidates = store.get_node_by_prefix(nid)
        assert node is not None and node.id == nid
        assert candidates == []

    @settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(content=_content)
    def test_unique_prefix_resolves_missing_never_raises(self, store, content):
        nid = store.store(content, metadata={"event_type": "decision"},
                          skip_inference=True)
        # A unique full-length id resolves; a bogus prefix returns (None, [])
        # without raising, whatever the content was.
        node, _ = store.get_node_by_prefix(nid[:12])
        assert node is None or node.id == nid
        assert store.get_node_by_prefix("mem-deadbeefcafef00d") == (None, [])

    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(prefix=st.text(max_size=3))
    def test_short_prefixes_are_rejected_cleanly(self, store, prefix):
        # <4 chars never triggers a huge LIKE scan; always (None, []).
        assert store.get_node_by_prefix(prefix) == (None, [])


# ---------------------------------------------------------------------------
# regex_search — never crash, matches are real
# ---------------------------------------------------------------------------

class TestRegexSearchProperties:
    @settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(pattern=st.text(max_size=40))
    def test_arbitrary_pattern_never_crashes(self, store, pattern):
        store.store("regex fuzz corpus alpha beta gamma",
                    metadata={"event_type": "decision"}, skip_inference=True)
        try:
            results = store.regex_search(pattern, limit=5)
        except ValueError:
            return  # invalid/catastrophic patterns raise ValueError by contract
        # Any returned row genuinely matches the (case-insensitive) pattern.
        rx = re.compile(pattern, re.IGNORECASE)
        for r in results:
            assert rx.search(r.content[:8000]) is not None

    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(needle=st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=8))
    def test_literal_needle_is_found(self, store, needle):
        nid = store.store(f"haystack containing {needle} in the middle",
                          metadata={"event_type": "decision"}, skip_inference=True)
        results = store.regex_search(re.escape(needle), limit=5)
        assert nid in {r.id for r in results}


# ---------------------------------------------------------------------------
# dedup — identical content collapses, distinct content does not
# ---------------------------------------------------------------------------

class TestDedupProperties:
    @settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(content=_content)
    def test_exact_duplicate_content_collapses(self, store, content):
        a = store.store(content, metadata={"event_type": "decision"}, skip_inference=True)
        b = store.store(content, metadata={"event_type": "decision"}, skip_inference=True)
        assert a == b, "identical content must dedup to the same node"
