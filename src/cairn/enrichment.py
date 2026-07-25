"""doc2query-style write-time enrichment — async vocabulary bridging.

Problem: indirect queries share almost no wording with the memory that
answers them ("what should I serve for dinner with my homegrown
ingredients?" vs a memory about the user's vegetable garden). The 2026-07
LongMemEval A/B showed query-side LLM expansion attacking this from the
read path buys nothing significant (R@5 0.966 vs 0.958, McNemar p=0.29)
while making retrieval nondeterministic and latency-coupled to a cloud
provider.

This module is the write-side version of the same idea (doc2query /
docT5query): during async maintenance, an LLM generates the questions a
user might ask that this memory answers, phrased in DIFFERENT vocabulary.
The generated queries are stored with the memory:

- appended to ``extracted_keywords``, which the FTS5 triggers already
  index (BM25 can now match the anticipated phrasing), and
- folded into the embedding (content + queries), pulling the doc vector
  toward query space.

Retrieval stays 100% local and deterministic: the enrichment artifacts are
durable rows computed once per memory, not per-query samples. Keyless
installs no-op cleanly — ``llm_complete`` returns "" and nothing changes.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

from cairn import json_compat as json
from cairn.llm import llm_complete

logger = logging.getLogger("cairn.enrichment")

# Knowledge types worth an LLM call. Episodic exhaust (task_completion,
# session_summary) is rollup's job; infrastructure types are never queried
# semantically.
ENRICH_ELIGIBLE_TYPES = ("user_preference", "decision", "lesson_learned", "memory")

# Below this the content is its own best index — an anticipated-query set
# for "prefers ruff" adds nothing BM25 doesn't already have.
MIN_CONTENT_CHARS = 80

# Only enrich rows this old (lets same-session dedup/supersession settle
# first — enriching a row that supersession is about to replace wastes the
# call and leaves stale FTS terms).
MIN_AGE_HOURS = 1

MAX_QUERIES = 5
_CONTENT_EXCERPT_CHARS = 2000

_SYSTEM_PROMPT = (
    "You index a personal memory store. Given one stored memory, write the "
    "search queries a user might type LATER whose answer is this memory — "
    "including indirect ones that share almost no wording with it (the "
    "memory says 'grows tomatoes and basil in the backyard'; a later query "
    "is 'what should I cook with my homegrown ingredients'). Use different "
    "vocabulary from the memory wherever possible; synonyms, consequences, "
    "and use-cases beat restatements. Return ONLY a JSON array of 3-5 "
    "short query strings. No markdown, no commentary."
)


def generate_anticipated_queries(content: str, event_type: str = "") -> List[str]:
    """Ask the configured LLM for queries this memory should answer.

    Returns [] on any failure — no key, timeout, unparseable output. The
    call is temperature 0 so re-running enrichment on the same content is
    reproducible (per provider snapshot).
    """
    prompt = f"Memory ({event_type or 'memory'}):\n{content[:_CONTENT_EXCERPT_CHARS]}"
    raw = llm_complete(
        prompt,
        _SYSTEM_PROMPT,
        max_tokens=300,
        temperature=0.0,
        timeout=15.0,
        model_tier="fast",
    )
    if not raw:
        return []

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(
            line for line in cleaned.split("\n") if not line.strip().startswith("```")
        )
    try:
        parsed = json.loads(cleaned)
    except Exception:
        logger.debug("enrichment: unparseable LLM output: %.120s", raw)
        return []
    if not isinstance(parsed, list):
        return []
    queries = [q.strip() for q in parsed if isinstance(q, str) and 0 < len(q.strip()) <= 200]
    return queries[:MAX_QUERIES]


def _fts_resync(store, row_id: int, content: str, old_kw: str, new_kw: str) -> None:
    """Mirror the trigger-maintained FTS row after an extracted_keywords change.

    The memories_au trigger fires only on UPDATE OF content, so keyword-only
    updates must maintain the external-content FTS index by hand: 'delete'
    with the OLD indexed text, insert with the NEW (exactly what the
    triggers do — see schema.py).
    """
    store._conn.execute(
        "INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', ?, ?)",
        (row_id, f"{content} {old_kw}".rstrip()),
    )
    store._conn.execute(
        "INSERT INTO memories_fts(rowid, content) VALUES (?, ?)",
        (row_id, f"{content} {new_kw}".rstrip()),
    )


def enrich_pending(store, limit: int = 50, min_age_hours: float = MIN_AGE_HOURS) -> Dict[str, Any]:
    """Enrich up to ``limit`` un-enriched knowledge memories.

    Idempotent: enriched rows carry ``metadata.doc2query`` and are never
    reprocessed. Fail-open per row (one bad LLM response skips that row),
    fail-closed overall (no key → early ``no_llm`` bail after the first
    few empty responses, nothing written).
    """
    from cairn.embedding import generate_embeddings_batch
    from cairn.sqlite_store._types import _serialize_f32

    placeholders = ",".join("?" * len(ENRICH_ELIGIBLE_TYPES))
    rows = store._conn.execute(
        f"""SELECT id, node_id, content, metadata, extracted_keywords, created_at
            FROM memories
            WHERE event_type IN ({placeholders})
              AND length(content) >= ?
              AND (metadata IS NULL OR metadata NOT LIKE '%"doc2query"%')
            ORDER BY id DESC LIMIT ?""",
        (*ENRICH_ELIGIBLE_TYPES, MIN_CONTENT_CHARS, limit),
    ).fetchall()

    now = datetime.now(timezone.utc)
    updated = skipped_empty = skipped_young = 0

    for row_id, node_id, content, metadata_raw, old_kw, created_at in rows:
        # Age gate (parsed in Python — created_at formats vary across versions)
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if (now - created).total_seconds() < min_age_hours * 3600:
                skipped_young += 1
                continue
        except Exception:
            pass  # unparseable timestamp: treat as old enough

        try:
            meta = json.loads(metadata_raw) if metadata_raw else {}
        except Exception:
            meta = {}
        if "doc2query" in meta:  # LIKE-filter false positive (nested string)
            continue

        queries = generate_anticipated_queries(content, meta.get("event_type", ""))
        if not queries:
            skipped_empty += 1
            # No key / provider down: every call returns "" instantly.
            # Bail before burning the whole batch on nothing.
            if updated == 0 and skipped_empty >= 3:
                return {
                    "status": "no_llm",
                    "updated": updated,
                    "skipped_empty": skipped_empty,
                    "skipped_young": skipped_young,
                    "candidates": len(rows),
                }
            continue

        meta["doc2query"] = {
            "queries": queries,
            "at": now.isoformat(),
            "provider": os.environ.get("CAIRN_LLM_PROVIDER", "anthropic"),
        }
        query_text = " ".join(queries)
        new_kw = f"{old_kw} {query_text}".strip() if old_kw else query_text

        # Re-embed content + anticipated queries (doc2query's vector half).
        embedding = None
        if getattr(store, "_vec_available", False):
            try:
                embed_text = content + "\n" + "\n".join(queries)
                result = generate_embeddings_batch([embed_text], mode="document")
                embedding = result[0] if result and result[0] else None
            except Exception as e:
                logger.debug("enrichment: re-embed failed for %s: %s", node_id, e)

        with store._lock:
            if getattr(store, "_fts_available", False):
                try:
                    _fts_resync(store, row_id, content, old_kw or "", new_kw)
                except Exception as e:
                    logger.debug("enrichment: FTS resync failed for %s: %s", node_id, e)
            store._conn.execute(
                "UPDATE memories SET metadata = ?, extracted_keywords = ? WHERE id = ?",
                (json.dumps(meta), new_kw, row_id),
            )
            if embedding:
                store._conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (row_id,))
                store._conn.execute(
                    "INSERT INTO memories_vec (rowid, embedding) VALUES (?, ?)",
                    (row_id, _serialize_f32(embedding)),
                )
            store._commit()
        updated += 1

    if updated:
        store._invalidate_query_cache()

    return {
        "status": "ok",
        "updated": updated,
        "skipped_empty": skipped_empty,
        "skipped_young": skipped_young,
        "candidates": len(rows),
    }
