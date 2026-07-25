"""Tests for LLM-based query expansion module."""

import json
import os
import time
import pytest
from unittest.mock import patch, MagicMock

from cairn.query_expansion import (
    expand_query,
    is_expansion_enabled,
    clear_cache,
    _CACHE_TTL_S,
)


class TestExpansionEnabled:
    """Test the gating env var."""

    def test_disabled_by_default(self):
        """Opt-in since the 2026-07 A/B: expansion is a synchronous cloud LLM
        call inside query() (temperature 0.3), so with a provider key present
        it made retrieval nondeterministic — while its LongMemEval delta was
        inside noise (paired McNemar over 500 questions: p=0.29 overall,
        p=0.63 on single-session-preference). Off unless explicitly enabled."""
        os.environ.pop("CAIRN_QUERY_EXPANSION", None)
        assert not is_expansion_enabled()

    def test_enabled_when_set(self):
        os.environ["CAIRN_QUERY_EXPANSION"] = "1"
        try:
            assert is_expansion_enabled()
        finally:
            os.environ.pop("CAIRN_QUERY_EXPANSION", None)

    def test_disabled_when_zero(self):
        os.environ["CAIRN_QUERY_EXPANSION"] = "0"
        try:
            assert not is_expansion_enabled()
        finally:
            os.environ.pop("CAIRN_QUERY_EXPANSION", None)


class TestExpandQuery:
    """Test expand_query function."""

    def test_empty_query_returns_empty(self):
        result = expand_query("")
        assert result == {"lex": [], "vec": [], "hyde": ""}

    def test_short_query_returns_empty(self):
        result = expand_query("ab")
        assert result == {"lex": [], "vec": [], "hyde": ""}

    @patch("cairn.llm.llm_complete")
    def test_successful_expansion(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["deployment config", "deploy settings"],
            "vec": ["how to configure deployment", "deployment configuration options"],
            "hyde": "",
        })
        clear_cache()
        result = expand_query("that deployment thing")
        assert len(result["lex"]) == 2
        assert len(result["vec"]) == 2
        assert result["hyde"] == ""
        mock_llm.assert_called_once()

    @patch("cairn.llm.llm_complete")
    def test_hyde_generation(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["auth bug fix"],
            "vec": ["authentication issue resolution"],
            "hyde": "We fixed the authentication bug by updating the JWT validation logic.",
        })
        clear_cache()
        result = expand_query("the auth issue", include_hyde=True)
        assert result["hyde"] != ""
        assert "authentication" in result["hyde"].lower() or "JWT" in result["hyde"]

    @patch("cairn.llm.llm_complete")
    def test_hyde_omitted_when_not_requested(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["auth bug"],
            "vec": ["authentication problem"],
            "hyde": "Some passage that should be ignored",
        })
        clear_cache()
        result = expand_query("the auth issue", include_hyde=False)
        assert result["hyde"] == ""

    @patch("cairn.llm.llm_complete")
    def test_max_variants_respected(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["a", "b", "c", "d", "e"],
            "vec": ["x", "y", "z", "w"],
            "hyde": "",
        })
        clear_cache()
        result = expand_query("test query", max_variants=2)
        assert len(result["lex"]) <= 2
        assert len(result["vec"]) <= 2

    @patch("cairn.llm.llm_complete")
    def test_llm_returns_empty(self, mock_llm):
        mock_llm.return_value = ""
        clear_cache()
        result = expand_query("test query")
        assert result == {"lex": [], "vec": [], "hyde": ""}

    @patch("cairn.llm.llm_complete")
    def test_llm_returns_invalid_json(self, mock_llm):
        mock_llm.return_value = "not json at all"
        clear_cache()
        result = expand_query("test query")
        assert result == {"lex": [], "vec": [], "hyde": ""}

    @patch("cairn.llm.llm_complete")
    def test_llm_returns_markdown_wrapped_json(self, mock_llm):
        mock_llm.return_value = '```json\n{"lex": ["foo"], "vec": ["bar"], "hyde": ""}\n```'
        clear_cache()
        result = expand_query("test query")
        assert result["lex"] == ["foo"]
        assert result["vec"] == ["bar"]

    @patch("cairn.llm.llm_complete")
    def test_llm_exception_returns_empty(self, mock_llm):
        mock_llm.side_effect = RuntimeError("API down")
        clear_cache()
        result = expand_query("test query")
        assert result == {"lex": [], "vec": [], "hyde": ""}


class TestExpansionCache:
    """Test the LRU cache behavior."""

    @patch("cairn.llm.llm_complete")
    def test_cache_hit_avoids_llm_call(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["cached variant"],
            "vec": ["cached rephrase"],
            "hyde": "",
        })
        clear_cache()

        # First call — hits LLM
        result1 = expand_query("cache test query")
        assert mock_llm.call_count == 1

        # Second call — cached, no LLM
        result2 = expand_query("cache test query")
        assert mock_llm.call_count == 1
        assert result1 == result2

    @patch("cairn.llm.llm_complete")
    def test_different_queries_different_cache_keys(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["v"], "vec": ["v"], "hyde": "",
        })
        clear_cache()

        expand_query("query one")
        expand_query("query two")
        assert mock_llm.call_count == 2

    def test_clear_cache(self):
        clear_cache()
        assert clear_cache() == 0  # Already empty

    @patch("cairn.llm.llm_complete")
    def test_cache_ttl_expiry(self, mock_llm):
        mock_llm.return_value = json.dumps({
            "lex": ["v"], "vec": ["v"], "hyde": "",
        })
        clear_cache()

        # Manually insert expired entry
        from cairn.query_expansion import _cache, _cache_lock
        cache_key = ("ttl test query", False, 3)
        with _cache_lock:
            _cache[cache_key] = (time.monotonic() - _CACHE_TTL_S - 10, {"lex": ["old"], "vec": [], "hyde": ""})

        # Should re-query LLM since cache entry is expired
        result = expand_query("ttl test query")
        assert mock_llm.call_count == 1
        assert result["lex"] == ["v"]  # New result, not cached "old"
