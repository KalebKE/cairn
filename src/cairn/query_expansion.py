"""LLM-based query expansion for improved retrieval recall.

Generates semantic variants of user queries to improve recall for vague
or underspecified queries. Auto-enabled for conceptual/vague queries.
Set CAIRN_QUERY_EXPANSION=0 to disable entirely.

Variant types:
  - lex: 2-5 word keyword-focused variants (for FTS5)
  - vec: Natural language rephrasings (for vector search)
  - hyde: Hypothetical memory passage answering the query (HyDE technique)
"""

import json
import logging
import os
import threading
import time as _time
from collections import OrderedDict
from typing import Dict

logger = logging.getLogger("cairn.query_expansion")

# Cache config
_CACHE_MAX = 64
_CACHE_TTL_S = 300  # 5 minutes

_cache: OrderedDict = OrderedDict()
_cache_lock = threading.Lock()

_EXPANSION_SYSTEM = """\
You generate search query variants to improve memory retrieval.
Output JSON only, no explanation. Schema:
{"lex": ["keyword variant 1", ...], "vec": ["natural language rephrase 1", ...], "hyde": "hypothetical memory passage or empty string"}

Rules:
- lex: 2-3 variants, each 2-5 words, keyword-focused (for full-text search)
- vec: 2-3 variants, natural language rephrasings (for embedding similarity)
- hyde: A 1-2 sentence passage that a stored memory answering the query might contain. Only generate if the query is conceptual/vague. Empty string otherwise.
- Do NOT repeat the original query verbatim in any variant.
- Keep variants diverse — different vocabulary, not just rewordings.\
"""


def is_expansion_enabled() -> bool:
    """Check if query expansion is enabled.

    Default is False (opt-in): set CAIRN_QUERY_EXPANSION=1 to enable.

    Expansion is a synchronous cloud-LLM call (temperature 0.3, 3s timeout)
    inside store.query(), so enabling it with a provider key present makes
    retrieval nondeterministic, latency-coupled to the provider, and sends
    query text off-machine. The 2026-07 paired A/B on LongMemEval (500
    questions) measured its benefit as inside noise: R@5 0.966 vs 0.958
    overall (McNemar p=0.29). The vocabulary-bridging idea lives on in the
    async enrichment path (doc2query-style, write-time) where it is
    deterministic at query time.
    """
    return os.environ.get("CAIRN_QUERY_EXPANSION", "0") == "1"


def expand_query(
    query_text: str,
    include_hyde: bool = False,
    max_variants: int = 3,
) -> Dict[str, object]:
    """Generate semantic variants of a query for improved retrieval.

    Args:
        query_text: The original user query.
        include_hyde: Whether to generate a HyDE passage.
        max_variants: Max number of lex/vec variants each.

    Returns:
        {"lex": [...], "vec": [...], "hyde": "..."} — empty lists/string on failure.
    """
    empty = {"lex": [], "vec": [], "hyde": ""}

    if not query_text or len(query_text.strip()) < 3:
        return empty

    # Check cache
    cache_key = (query_text, include_hyde, max_variants)
    with _cache_lock:
        if cache_key in _cache:
            ts, result = _cache[cache_key]
            if (_time.monotonic() - ts) < _CACHE_TTL_S:
                _cache.move_to_end(cache_key)
                return result
            else:
                del _cache[cache_key]

    try:
        from cairn.llm import llm_complete

        hyde_instruction = (
            "Generate a hyde passage." if include_hyde
            else "Set hyde to empty string."
        )
        prompt = (
            f"Query: {query_text}\n"
            f"Max variants per type: {max_variants}\n"
            f"{hyde_instruction}"
        )
        raw = llm_complete(
            prompt,
            _EXPANSION_SYSTEM,
            max_tokens=300,
            temperature=0.3,
            timeout=3.0,
            model_tier="fast",
        )
        if not raw:
            return empty

        # Parse JSON — handle LLM wrapping it in markdown code blocks
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            # Strip ```json ... ``` wrapper
            lines = cleaned.split("\n")
            cleaned = "\n".join(
                l for l in lines if not l.strip().startswith("```")
            )

        parsed = json.loads(cleaned)
        result: Dict[str, object] = {
            "lex": list(parsed.get("lex", []))[:max_variants],
            "vec": list(parsed.get("vec", []))[:max_variants],
            "hyde": str(parsed.get("hyde", "")) if include_hyde else "",
        }

        # Cache the result
        with _cache_lock:
            _cache[cache_key] = (_time.monotonic(), result)
            while len(_cache) > _CACHE_MAX:
                _cache.popitem(last=False)

        return result

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.debug("Query expansion parse failed: %s", e)
        return empty
    except Exception as e:
        logger.debug("Query expansion failed: %s", e)
        return empty


def clear_cache() -> int:
    """Clear the expansion cache. Returns number of evicted entries."""
    with _cache_lock:
        n = len(_cache)
        _cache.clear()
        return n
