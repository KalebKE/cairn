"""
Cairn Bridge -- High-level API for Cairn memory system.

Provides the public interface used by the MCP server handlers.
All functions are thin wrappers that delegate to the SQLiteStore singleton.

Public API (36 functions, see __all__ for full list):
    Core:       auto_capture, store, remember, delete_memory, edit_memory
    Query:      query, query_structured, phrase_search, find_similar_memories
    Session:    welcome, clear_session, batch_store
    Health:     check_health, status, get_dedup_stats
    Profile:    get_profile, save_profile, extract_preferences, list_preferences
    Lessons:    get_cross_session_lessons, get_cross_project_lessons
    Maintenance: consolidate, compact, deduplicate, timeline, traverse
    Export:     export_memories, import_memories, reingest
    Stats:      type_stats, session_stats
    Constraints: check_constraints, list_constraints, save_constraints
    Feedback:   record_feedback
    Testing:    reset_memory
"""

import atexit
import logging
import os
import re
import threading
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cairn import json_compat as json
from cairn.exceptions import ValidationError
from cairn.llm import llm_complete  # noqa: F401 — used in distill_trajectory, module-level for test patchability
from cairn.types import TTLCategory, AutoCaptureEventType

logger = logging.getLogger("cairn.bridge")


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------

CAIRN_HOME = Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))

# --- Ingestion vocabulary + helpers (peeled to cairn.bridge.ingest) ---
from cairn.bridge.ingest import (  # noqa: E402,F401
    _normalize_for_dedup,
    _extract_tags,
    _infer_temporal_range,
    _extract_facts,
    _compress_to_observation,
    _auto_relate,
    _schedule_auto_relate,
    _detect_and_supersede,
    _split_atomic_facts,
    _schedule_entity_extraction,
    DEDUP_THRESHOLDS,
    EVOLUTION_TYPES,
    EVOLUTION_THRESHOLD,
    _BLOCKLIST_STARTSWITH,
    _BLOCKLIST_CONTAINS,
    _MIN_CONTENT_LENGTH,
    _INFRASTRUCTURE_EVENT_TYPES,
    _TAG_LANGUAGES,
    _TAG_TOOLS,
    _TAG_ALIASES,
    _GO_CONTEXT_WORDS,
    _TAG_CONCEPTS,
    _CONCEPT_CANONICAL,
    _CROSS_TYPE_SUPERSEDE,
)


# Content blocklist — reject system noise at ingestion time.

# Minimum content length for auto-capture (reject very short noise).



def _check_milestone(name: str) -> bool:
    """Return True if milestone not yet achieved (first time). Creates marker.

    DEPRECATED: Use cairn.milestones._check_milestone instead.
    Kept as thin redirect for any callers that import from bridge.
    """
    from cairn.milestones import _check_milestone as _cm
    return _cm(name)


# ---------------------------------------------------------------------------
# Lazy singleton -- SQLiteStore replaces CairnMemory
# ---------------------------------------------------------------------------

_store_instance = None
_store_lock = threading.Lock()

# Welcome-briefing cache. Lives here (not in bridge/welcome.py) because
# reset_memory() clears it; welcome.py reads/writes it via _bridge._welcome_cache.
_welcome_cache: Dict[str, tuple] = {}  # key -> (monotonic_ts, result_dict)
_WELCOME_CACHE_TTL = 30.0  # seconds


# ---------------------------------------------------------------------------
# Bridge initialization options
# ---------------------------------------------------------------------------

_bridge_enable_vector_search: bool = False
_bridge_onnx_model_path: Optional[str] = None


def initialize_bridge(
    enable_vector_search: bool = False,
    onnx_model_path: Optional[str] = None,
) -> None:
    """Configure bridge options before first store access.

    Must be called before any store operation if non-default options are needed.

    Args:
        enable_vector_search: If True, pass the ONNX embed model to SQLiteStore
            to enable offline semantic (vector) search via sqlite-vec.
        onnx_model_path: Path to the ONNX embedding model file.  When *None*
            the store will use its own default path (if any).
    """
    global _bridge_enable_vector_search, _bridge_onnx_model_path
    _bridge_enable_vector_search = enable_vector_search
    _bridge_onnx_model_path = onnx_model_path
    logger.debug(
        "Bridge configured: enable_vector_search=%s, onnx_model_path=%s",
        enable_vector_search,
        onnx_model_path,
    )


def _get_store():
    """Get or create the SQLiteStore singleton (thread-safe).

    Uses local variable for init to ensure _store_instance is only set
    after full initialization (migration + cleanup + atexit registration).
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is not None:
            return _store_instance
        # Auto-migrate from JSON graphs if needed (first run after upgrade)
        from cairn.migrate_to_sqlite import auto_migrate_if_needed

        auto_migrate_if_needed()

        from cairn.sqlite_store import SQLiteStore

        # Build keyword arguments based on bridge configuration.
        store_kwargs: Dict[str, Any] = {}
        if _bridge_enable_vector_search:
            store_kwargs["onnx_model_path"] = _bridge_onnx_model_path
            logger.info(
                "Vector search enabled; ONNX model path: %s",
                _bridge_onnx_model_path or "<store default>",
            )

        # Init into local var first; only publish to global after full setup
        store = SQLiteStore(**store_kwargs)
        # Purge expired nodes on startup
        expired = store.cleanup_expired()
        if expired > 0:
            logger.info(f"Startup: purged {expired} expired nodes")
        atexit.register(_close_store)
        _store_instance = store  # Publish only after full init
    return _store_instance


def _close_store():
    """Close SQLiteStore on process exit."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during refresh: %s", e)


def reset_memory():
    """Reset the singleton (useful for testing)."""
    global _store_instance
    if _store_instance is not None:
        try:
            _store_instance.close()
        except Exception as e:
            logger.debug("Store close failed during reset: %s", e)
    _store_instance = None
    _welcome_cache.clear()
# --- Query (peeled to cairn.bridge.query) ---
from cairn.bridge.query import (  # noqa: E402,F401
    semantic_search, query, query_structured,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _human_ttl(ttl: Optional[int]) -> str:
    """Format TTL seconds as human-readable string."""
    if not ttl:
        return "permanent"
    if ttl < 3600:
        return f"{ttl // 60}m"
    if ttl < 86400:
        return f"{ttl // 3600}h"
    return f"{ttl // 86400}d"







def _relative_time(created_at) -> str:
    """Format a datetime as a human-readable relative time string."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if isinstance(created_at, str):
        try:
            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            created_at = datetime.fromisoformat(created_at)
        except (ValueError, TypeError):
            return ""
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = delta.total_seconds()
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        m = int(seconds // 60)
        return f"{m}m ago"
    if seconds < 86400:
        h = int(seconds // 3600)
        return f"{h}h ago"
    days = int(seconds // 86400)
    if days == 1:
        return "yesterday"
    if days < 30:
        return f"{days}d ago"
    months = days // 30
    if months == 1:
        return "1 month ago"
    return f"{months} months ago"


def _jaccard(text_a: str, text_b: str, min_word_len: int = 4) -> float:
    """Jaccard similarity on word sets (fast, no embeddings)."""
    words_a = {w for w in text_a.split() if len(w) >= min_word_len}
    words_b = {w for w in text_b.split() if len(w) >= min_word_len}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)




# ---------------------------------------------------------------------------
# Public API -- Core CRUD
# ---------------------------------------------------------------------------


def auto_capture(
    content: str,
    event_type: str,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    ttl_override: Optional[int] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Store a memory with auto-classification, dedup, and evolution.

    This is the primary ingestion function. It:
    1. Checks for near-duplicate content (Jaccard) and reuses if found.
    2. Tries to *evolve* an existing memory with new insights (Zettelkasten).
    3. Falls back to creating a new memory node.

    Returns:
        Markdown confirmation string.
    """
    content = unicodedata.normalize("NFC", content)

    # Auto-resolve entity_id from project if not explicitly provided
    if not entity_id and project:
        try:
            from cairn_platform.entity.engine import resolve_project_entity
            entity_id = resolve_project_entity(project)
        except Exception as e:
            logger.debug("Entity resolution failed: %s", e)

    # Determine source early — hooks vs direct API calls have different filtering rules.
    _source = (metadata or {}).get("source", "")
    _is_hook = _source.startswith("auto_") or _source.endswith("_hook")

    # Block system noise early — startswith patterns are position-specific, safe for all sources.
    for pattern in _BLOCKLIST_STARTSWITH:
        if content.startswith(pattern):
            return "**Memory Blocked** (system noise)"
    # Contains patterns only apply to hook-sourced content to avoid false positives
    # on direct API calls (e.g. storing a lesson that mentions "error":).
    # Also skip for preference/fact types — users legitimately store prefs containing "error".
    _BLOCKLIST_EXEMPT_TYPES = {AutoCaptureEventType.USER_PREFERENCE, AutoCaptureEventType.USER_FACT}
    if _is_hook and event_type not in _BLOCKLIST_EXEMPT_TYPES:
        for pattern in _BLOCKLIST_CONTAINS:
            if pattern in content:
                return "**Memory Blocked** (system noise)"

    # Min-length gate — only for auto-captured content from hooks, not direct API calls.
    if _is_hook and len(content) < _MIN_CONTENT_LENGTH and event_type != AutoCaptureEventType.USER_PREFERENCE:
        return "**Memory Blocked** (too short)"

    # Block infrastructure event types that generate noise and inflate never-accessed count
    if _is_hook and event_type in _INFRASTRUCTURE_EVENT_TYPES:
        return "**Memory Blocked** (infrastructure noise)"

    # Block zero-value outcome records (tokens=0 partial sessions)
    if _is_hook and event_type == "task_completion" and "tokens=0" in content:
        return "**Memory Blocked** (zero-token outcome)"

    # Block JSON-blob decisions — raw tool output stored as "decisions"
    # Exempt coord_dual_write: its [domain] prefix is not JSON
    if event_type == "decision" and _source != "coord_dual_write":
        # Strip known prefixes to check the actual body
        _body = content
        for _pfx in ("Decision: ", "Plan/decision captured: ", "Fact: "):
            if _body.startswith(_pfx):
                _body = _body[len(_pfx):]
        _body_stripped = _body.lstrip()
        if _body_stripped.startswith(("{", "[", '"filePath', '"type"')):
            return "**Memory Blocked** (JSON blob, not a decision)"

    store = _get_store()
    meta = dict(metadata or {})
    meta["event_type"] = event_type
    if project:
        meta["project"] = project
    meta["captured_at"] = datetime.now(timezone.utc).isoformat()

    # Set capture confidence (if not already set by caller)
    if not meta.get("capture_confidence"):
        source = meta.get("source", "")
        if source == "user_remember":
            meta["capture_confidence"] = "high"
        elif event_type == "user_preference":
            meta["capture_confidence"] = "high"
        elif event_type in ("lesson_learned", "error_pattern") and not source.startswith("auto_"):
            # Direct API calls for lessons/errors = validated by agent
            meta["capture_confidence"] = "high"
        elif source in ("auto_plan_capture",):
            # Auto-captured plans are speculative
            meta["capture_confidence"] = "low"
        elif source.startswith("auto_") or source.endswith("_hook"):
            meta["capture_confidence"] = "medium"
        else:
            # Direct API calls (agent-initiated store) = higher trust
            meta["capture_confidence"] = "high"

    # Auto-tag extraction
    auto_tags = _extract_tags(content, project)
    if auto_tags:
        existing_tags = meta.get("tags", [])
        meta["tags"] = sorted(set(existing_tags + auto_tags))[:15]

    # Fact extraction for high-value types — merge fact terms into tags
    # (boosted in Phase 2.5 word/tag overlap for retrieval).
    _FACT_EXTRACTION_TYPES = {"decision", "lesson_learned", "session_summary", "error_pattern", "advisor_insight"}
    if event_type in _FACT_EXTRACTION_TYPES:
        try:
            facts = _extract_facts(content)
            if facts:
                # Merge fact terms into tags (boosted in Phase 2.5 word/tag overlap)
                existing_tags = meta.get("tags", [])
                # Take the shortest/most specific facts as tags (avoid long phrases)
                fact_tags = [f for f in facts if len(f) <= 25 and " " not in f]
                meta["tags"] = sorted(set(existing_tags + fact_tags))[:20]
        except Exception as e:
            logger.debug(f"Fact extraction failed: {e}")

    ttl = ttl_override if ttl_override is not None else TTLCategory.for_event_type(event_type)

    # System insights are architectural knowledge — make them permanent
    if ttl is not None and meta.get("category") == "system_insight":
        ttl = None  # None = permanent (no expiry)

    # ------------------------------------------------------------------
    # Phase 1 + 2: Content dedup, error burst, and evolution
    # ------------------------------------------------------------------
    # Single query for both dedup and evolution (same search text).
    # This avoids duplicate embedding generation + DB round-trips.
    dedup_threshold = DEDUP_THRESHOLDS.get(event_type)
    _similar_results = None  # Lazy-loaded, shared between dedup and evolution

    # Pre-compute embedding once — reused for dedup query and final store
    _precomputed_embedding = None
    if dedup_threshold is not None or event_type in EVOLUTION_TYPES:
        try:
            from cairn.embedding import generate_embedding
            _precomputed_embedding = generate_embedding(content)
        except Exception as e:
            logger.debug(f"Pre-computed embedding generation failed: {e}")

    if dedup_threshold is not None or event_type in EVOLUTION_TYPES:
        try:
            _similar_results = store.query(
                content[:200], limit=8,
                query_embedding=_precomputed_embedding,
            )
        except Exception as e:
            logger.debug(f"Similar-content query failed: {e}")

    # Phase 1: Content-level dedup
    if dedup_threshold is not None and _similar_results:
        try:
            for existing in _similar_results:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                # Session filter for dedup: only dedup within same session
                # Exception: decisions, lessons, and task completions dedup cross-session
                # (same architectural choice, lesson, or completion restated across sessions)
                _CROSS_SESSION_DEDUP_TYPES = {
                    AutoCaptureEventType.DECISION,
                    AutoCaptureEventType.LESSON_LEARNED,
                    AutoCaptureEventType.TASK_COMPLETION,
                    AutoCaptureEventType.ADVISOR_INSIGHT,
                }
                if session_id and event_type not in _CROSS_SESSION_DEDUP_TYPES:
                    existing_session = (existing.metadata or {}).get("session_id", "")
                    if existing_session and existing_session != session_id:
                        continue
                if event_type == AutoCaptureEventType.ERROR_PATTERN:
                    sim = _jaccard(_normalize_for_dedup(content), _normalize_for_dedup(existing.content))
                else:
                    sim = _jaccard(content.lower(), existing.content.lower())
                if sim > dedup_threshold:
                    store.update_node(existing.id, access_count=(existing.access_count or 0) + 1)
                    store.stats.setdefault("content_dedup_skips", 0)
                    store.stats["content_dedup_skips"] += 1
                    _schedule_auto_relate(store, existing.id)
                    logger.debug(f"Content dedup: skipped {event_type} (jaccard={sim:.2f}), reusing {existing.id[:12]}")
                    return f"Deduped → {existing.id}"
        except Exception as e:
            logger.debug(f"Content dedup check skipped: {e}")

    # Phase 1.5: Error burst detection
    if event_type == AutoCaptureEventType.ERROR_PATTERN and session_id:
        try:
            # Use similar results if available, otherwise minimal query
            burst_candidates = _similar_results or []
            session_errors = [
                r
                for r in burst_candidates
                if (r.metadata or {}).get("event_type") == AutoCaptureEventType.ERROR_PATTERN
                and (r.metadata or {}).get("session_id") == session_id
            ]
            if len(session_errors) >= 3:
                # Only capture if truly novel (Jaccard < 0.40 with all recent errors)
                is_novel = all(_jaccard(content.lower(), e.content.lower()) < 0.40 for e in session_errors)
                if not is_novel:
                    store.stats.setdefault("error_burst_skips", 0)
                    store.stats["error_burst_skips"] += 1
                    return "Blocked (error burst — duplicate)"
        except Exception as e:
            logger.debug(f"Error burst check skipped: {e}")

    # Phase 2: Memory evolution (Zettelkasten-inspired)
    if event_type in EVOLUTION_TYPES and _similar_results:
        try:
            for existing in _similar_results[:3]:
                if (existing.metadata or {}).get("event_type", "") != event_type:
                    continue
                sim = _jaccard(content.lower(), existing.content.lower())
                if EVOLUTION_THRESHOLD <= sim < (dedup_threshold or 0.95):
                    old_words = {w.lower() for w in existing.content.split() if len(w) > 3}
                    new_info = {w.lower() for w in content.split() if len(w) > 3} - old_words
                    if len(new_info) == 0:
                        # Near-exact reconfirmation — bump access count to strengthen memory
                        store.update_node(
                            existing.id,
                            access_count=(existing.access_count or 0) + 1,
                        )
                        store.stats.setdefault("reconfirmation_bumps", 0)
                        store.stats["reconfirmation_bumps"] += 1
                        return f"Reconfirmed {existing.id} (access bumped)"
                    # Allow evolution with even 1 new word (was 3 — caused dead zone)
                    # The sentence-level filter below still requires >= 2 new words per sentence

                    evolved = existing.content.rstrip()
                    if not evolved.endswith("."):
                        evolved += "."

                    new_sentences = []
                    # Split on sentence boundaries, preserving abbreviations
                    # like "Dr.", "e.g.", "i.e.", version numbers "v2.0"
                    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", content):
                        sentence = sentence.strip()
                        if not sentence or len(sentence) < 10:
                            continue
                        s_words = {w.lower() for w in sentence.split() if len(w) > 3}
                        if s_words and len(s_words - old_words) >= 2:
                            new_sentences.append(sentence)

                    if new_sentences:
                        addition = " ".join(new_sentences[:2])
                        new_content = f"{evolved} [Updated] {addition}"
                        emeta = dict(existing.metadata or {})
                        evo_count = emeta.get("evolution_count", 0) + 1
                        emeta["evolution_count"] = evo_count
                        emeta["last_evolved"] = datetime.now(timezone.utc).isoformat()
                        emeta["evolved_from_sessions"] = list(
                            set(emeta.get("evolved_from_sessions", []) + ([session_id] if session_id else []))
                        )[:10]

                        store.update_node(
                            existing.id,
                            content=new_content,
                            metadata=emeta,
                            access_count=(existing.access_count or 0) + 1,
                        )

                        store.stats.setdefault("memory_evolutions", 0)
                        store.stats["memory_evolutions"] += 1
                        _schedule_auto_relate(store, existing.id)
                        logger.info(f"Memory evolved: {existing.id[:12]} (evolution #{evo_count}, jaccard={sim:.2f})")
                        return f"Evolved {existing.id} (#{evo_count})"
                    break  # Only try the top match
        except Exception as e:
            logger.debug(f"Memory evolution check skipped: {e}")

    # ------------------------------------------------------------------
    # Phase 2.5: Conflict detection — find contradictions with existing
    # ------------------------------------------------------------------
    _conflict_results = []
    if _similar_results and event_type in (
        "user_preference", "decision", "lesson_learned", "error_pattern"
    ):
        try:
            from cairn.conflicts import detect_conflicts

            _conflict_results = detect_conflicts(
                content, event_type, _similar_results[:5],
            )

            # Auto-resolve: mark old memory as outdated
            for conflict in _conflict_results:
                if conflict["auto_resolve"] and conflict.get("existing_id"):
                    try:
                        store.record_feedback(
                            conflict["existing_id"], "outdated",
                            reason=f"Conflict auto-resolved: {conflict['reason']}",
                        )
                    except Exception as e:
                        logger.debug(f"Auto-resolve feedback failed: {e}")

                # Flag-only: add conflict_flags to metadata for visibility
                if not conflict["auto_resolve"] and conflict.get("existing_id"):
                    try:
                        existing_node = store.get_node(conflict["existing_id"])
                        if existing_node:
                            emeta = dict(existing_node.metadata or {})
                            flags = emeta.get("conflict_flags", [])
                            flags.append({
                                "reason": conflict["reason"],
                                "confidence": conflict["confidence"],
                            })
                            emeta["conflict_flags"] = flags[-5:]  # Keep last 5
                            store.update_node(conflict["existing_id"], metadata=emeta)
                    except Exception as e:
                        logger.debug(f"Conflict flagging failed: {e}")
        except Exception as e:
            logger.debug(f"Conflict detection failed: {e}")

    # ------------------------------------------------------------------
    # Phase 3: Store new node
    # ------------------------------------------------------------------
    # Wire entity_id into metadata for tag-based discovery
    if entity_id:
        meta["entity_id"] = entity_id

    # Wire agent_type into metadata for discovery
    if agent_type:
        meta["agent_type"] = agent_type

    node_id = store.store(
        content=content,
        session_id=session_id,
        metadata=meta,
        embedding=_precomputed_embedding,
        ttl_seconds=ttl,
        entity_id=entity_id,
        agent_type=agent_type,
    )

    ttl_str = _human_ttl(ttl)
    output = f"Stored {node_id} ({event_type}, {ttl_str})"

    # Surface deep contradiction detection results
    try:
        contradiction_results = store.get_last_contradiction_results()
        if contradiction_results:
            cr_lines = []
            for cr in contradiction_results:
                cr_lines.append(
                    f"  - `{cr['node_id'][:16]}` ({cr['confidence']:.0%}): {cr['reason']}"
                )
            output += "\n\n[CONTRADICTION] New memory may contradict:\n" + "\n".join(cr_lines)
    except Exception as e:
        logger.debug("Contradiction surfacing failed: %s", e)

    # Surface capacity warning if near limit
    if hasattr(store, '_capacity_warning') and store._capacity_warning:
        output += f"\n\n**Warning:** {store._capacity_warning}"

    # Append conflict information compactly
    if _conflict_results:
        resolved = sum(1 for c in _conflict_results if c["auto_resolve"])
        flagged = len(_conflict_results) - resolved
        parts = []
        if resolved:
            parts.append(f"{resolved} auto-resolved")
        if flagged:
            parts.append(f"{flagged} flagged")
        output += f" | conflicts: {', '.join(parts)}"

    # Milestone check (cheap: one node_count query + file existence check).
    try:
        from cairn.milestones import check_capture_milestones
        count = store.node_count()
        milestone_msg = check_capture_milestones(count)
        if milestone_msg:
            output += f" | {milestone_msg}"
    except Exception as e:
        logger.debug("Milestone check failed: %s", e)

    # ------------------------------------------------------------------
    # Phase 3.1: Async entity extraction (non-blocking)
    # ------------------------------------------------------------------
    _schedule_entity_extraction(store, node_id, content, event_type)

    # ------------------------------------------------------------------
    # Phase 3.5: Observation compression for high-value types
    # ------------------------------------------------------------------
    _HIGH_VALUE_OBSERVATION_TYPES = {"decision", "lesson_learned", "error_pattern", "user_preference", "constraint", "advisor_insight"}
    if event_type in _HIGH_VALUE_OBSERVATION_TYPES:
        try:
            observation = _compress_to_observation(content, event_type)
            if observation:
                meta["observation"] = observation
                store.update_node(node_id, metadata=meta)
        except Exception as e:
            logger.debug(f"Observation compression failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4: Auto-relate — link to similar existing memories (background)
    # ------------------------------------------------------------------
    _schedule_auto_relate(store, node_id)

    # ------------------------------------------------------------------
    # Phase 4.1: Contradiction detection — supersede old conflicting memories
    # ------------------------------------------------------------------
    try:
        supersede_count = _detect_and_supersede(
            store, node_id, content, event_type, entity_id,
        )
        if supersede_count:
            output += f" | {supersede_count} superseded"
    except Exception as e:
        logger.debug(f"Contradiction detection failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4.2: Atomic fact splitting — create sub-nodes for recall
    # ------------------------------------------------------------------
    try:
        atomic_facts = _split_atomic_facts(content, event_type)
        fact_count = 0
        for fact_text in atomic_facts:
            fact_meta = {
                "event_type": "user_fact",
                "source_node": node_id,
                "auto_extracted": True,
            }
            if session_id:
                fact_meta["session_id"] = session_id
            if project:
                fact_meta["project"] = project
            if entity_id:
                fact_meta["entity_id"] = entity_id
            fact_id = store.store(
                content=fact_text,
                session_id=session_id,
                metadata=fact_meta,
                entity_id=entity_id,
            )
            store.add_edge(node_id, fact_id, "contains_fact", 1.0)
            fact_count += 1
        if fact_count:
            output += f" | {fact_count} facts extracted"
    except Exception as e:
        logger.debug(f"Atomic fact splitting failed for {node_id[:12]}: {e}")

    # ------------------------------------------------------------------
    # Phase 4.5: Auto-supersede stale reminders
    # ------------------------------------------------------------------
    _COMPLETION_TYPES = {"decision", "task_completion"}
    if event_type in _COMPLETION_TYPES:
        try:
            superseded_count = 0
            superseded_ids: set = set()
            content_words = {w.lower() for w in content.split() if len(w) > 3}

            # --- Pass 1: Embedding similarity (threshold lowered to 0.40) ---
            embedding = store.get_embedding(node_id)
            if embedding:
                similar = store.find_similar(embedding, limit=10)
                for r in similar:
                    if r.id == node_id:
                        continue
                    r_type = (r.metadata or {}).get("event_type")
                    if r_type not in ("reminder", "checkpoint"):
                        continue
                    if (r.metadata or {}).get("superseded"):
                        continue
                    if r.relevance < 0.40:
                        continue
                    superseded_ids.add(r.id)

            # --- Pass 2: Keyword matching (3+ word overlap, like task auto-resolve) ---
            with store._lock:
                pending_rows = store._conn.execute(
                    "SELECT node_id, content FROM memories "
                    "WHERE event_type = 'reminder' "
                    "AND json_extract(metadata, '$.reminder_status') = 'pending'"
                ).fetchall()
            for r_id, r_content in pending_rows:
                if r_id in superseded_ids:
                    continue
                r_words = {w.lower() for w in (r_content or "").split() if len(w) > 3}
                matches = sum(1 for w in r_words if w in content_words)
                if matches >= 3:
                    superseded_ids.add(r_id)

            # --- Apply: mark superseded AND set reminder_status = dismissed ---
            for s_id in superseded_ids:
                r_row = store.get(s_id)
                if not r_row:
                    continue
                r_meta = dict(r_row.metadata or {})
                r_meta["superseded"] = True
                r_meta["superseded_by"] = node_id
                r_meta["reminder_status"] = "dismissed"
                r_meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
                r_meta["dismissed_reason"] = "auto_superseded"
                store.update_node(s_id, metadata=r_meta)
                r_type = r_meta.get("event_type", "reminder")
                store._log_forgetting_external(
                    s_id, r_row.content, r_type,
                    "auto_superseded", {"superseded_by": node_id},
                )
                superseded_count += 1
            if superseded_count:
                output += f" | superseded {superseded_count} reminder(s)"
                logger.info(f"Auto-superseded {superseded_count} reminders for {node_id}")
        except Exception as e:
            logger.debug(f"Auto-supersede failed for {node_id}: {e}")

    # ------------------------------------------------------------------
    # Phase 5: Implicit positive feedback — retrieval-then-store signal
    # ------------------------------------------------------------------
    _IMPLICIT_FB_TYPES = {"decision", "lesson_learned", "error_pattern"}
    if event_type in _IMPLICIT_FB_TYPES:
        try:
            ctx_entries = store.get_retrieval_context()
            if ctx_entries:
                content_words = {w.lower() for w in content.split() if len(w) > 3}
                implicit_count = 0
                for entry in ctx_entries:
                    query_text = entry.get("query_text", "")
                    if not query_text:
                        continue
                    query_words = {w.lower() for w in query_text.split() if len(w) > 3}
                    if not query_words:
                        continue
                    overlap = len(content_words & query_words) / len(query_words)
                    if overlap >= 0.30:
                        entry_nid = entry.get("node_id", "")
                        if entry_nid and entry_nid != node_id:
                            store.record_feedback(
                                entry_nid, "helpful",
                                reason="implicit: retrieval-then-store",
                            )
                            implicit_count += 1
                if implicit_count:
                    store.stats.setdefault("implicit_feedback_boosts", 0)
                    store.stats["implicit_feedback_boosts"] += implicit_count
                    logger.debug(f"Implicit feedback: boosted {implicit_count} memories for {node_id[:12]}")
        except Exception as e:
            logger.debug(f"Implicit feedback failed for {node_id[:12]}: {e}")

    logger.info(f"Auto-captured {event_type}: {node_id}")
    return output


def store(
    content: str,
    event_type: str = "memory",
    metadata: Optional[Dict[str, Any]] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
) -> str:
    """Direct store -- wraps auto_capture with a default event type."""
    return auto_capture(
        content=content,
        event_type=event_type,
        metadata=metadata,
        session_id=session_id,
        project=project,
        entity_id=entity_id,
        agent_type=agent_type,
    )


def remember(text: str, session_id: Optional[str] = None, entity_id: Optional[str] = None) -> str:
    """User-facing 'remember this' -- stores with user_preference type."""
    return auto_capture(
        content=text,
        event_type=AutoCaptureEventType.USER_PREFERENCE,
        session_id=session_id,
        entity_id=entity_id or "cairn",
        metadata={"source": "user_remember"},
    )


def delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a memory by its node ID."""
    db = _get_store()
    try:
        success = db.delete_node(memory_id)
        if success:
            logger.info(f"Deleted memory {memory_id[:12]}")
            return {"success": True, "deleted_id": memory_id}
        return {"success": False, "error": f"Memory {memory_id} not found"}
    except Exception as e:
        logger.error(f"Failed to delete memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def edit_memory(memory_id: str, new_content: str) -> Dict[str, Any]:
    """Edit a memory's content by its node ID.

    After editing, extracts style observations from the diff and stores
    them as user_preference memories (memory-from-edits pattern).
    """
    db = _get_store()
    try:
        node = db.get_node(memory_id)
        if node is None:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        old_content = node.content
        old_preview = old_content[:80]
        emeta = dict(node.metadata or {})
        emeta["edited_at"] = datetime.now(timezone.utc).isoformat()
        emeta["edit_count"] = emeta.get("edit_count", 0) + 1

        db.update_node(memory_id, content=new_content, metadata=emeta)

        logger.info(f"Edited memory {memory_id[:12]}")

        # Memory-from-edits: extract style observations from the diff
        edit_observation = _extract_edit_observation(
            old_content, new_content,
            event_type=(node.metadata or {}).get("event_type", "memory"),
            memory_id=memory_id,
        )

        result = {
            "success": True,
            "id": memory_id,
            "old_content_preview": old_preview,
            "new_content_preview": new_content[:80],
        }

        if edit_observation:
            result["style_observation"] = edit_observation

        return result
    except Exception as e:
        logger.error(f"Failed to edit memory {memory_id[:12]}: {e}", exc_info=True)
        return {"success": False, "error": "Failed to edit memory"}


def _extract_edit_observation(
    old_content: str,
    new_content: str,
    event_type: str = "memory",
    memory_id: str = "",
) -> Optional[str]:
    """Extract a style observation from a human edit and store it.

    Analyzes the diff between old and new content to detect patterns:
    - Length changes (conciseness preference)
    - Word additions/removals (terminology preferences)
    - Structural changes (formatting preferences)

    Returns the observation text if one was stored, None otherwise.
    """
    # Skip trivial edits
    if not old_content or not new_content:
        return None
    if old_content.strip() == new_content.strip():
        return None

    old_words = set(old_content.lower().split())
    new_words = set(new_content.lower().split())
    added_words = new_words - old_words
    removed_words = old_words - new_words

    # Skip if change is too small to learn from
    if len(added_words) + len(removed_words) < 3:
        return None

    observations = []

    # Length change
    old_len = len(old_content)
    new_len = len(new_content)
    if new_len < old_len * 0.7:
        observations.append("prefers more concise/shorter content")
    elif new_len > old_len * 1.5:
        observations.append("prefers more detailed/longer content")

    # Significant word additions (filter noise words)
    _noise = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "but",
              "in", "on", "at", "to", "for", "of", "with", "by", "from", "that",
              "this", "it", "not", "be", "have", "has", "had", "do", "does", "did"}
    meaningful_added = {w for w in added_words if w not in _noise and len(w) > 2}
    meaningful_removed = {w for w in removed_words if w not in _noise and len(w) > 2}

    if meaningful_removed and meaningful_added and len(meaningful_removed) <= 5:
        # Word replacement pattern — most valuable signal
        removed_sample = ", ".join(sorted(meaningful_removed)[:3])
        added_sample = ", ".join(sorted(meaningful_added)[:3])
        observations.append(f"replaced terms ({removed_sample}) with ({added_sample})")

    # Formatting changes
    old_has_bullets = "- " in old_content or "* " in old_content
    new_has_bullets = "- " in new_content or "* " in new_content
    if not old_has_bullets and new_has_bullets:
        observations.append("prefers bullet-point formatting")
    elif old_has_bullets and not new_has_bullets:
        observations.append("prefers prose over bullet points")

    old_has_headers = "## " in old_content or "# " in old_content
    new_has_headers = "## " in new_content or "# " in new_content
    if not old_has_headers and new_has_headers:
        observations.append("prefers headers/structure in content")

    if not observations:
        return None

    # Build and store the observation
    observation_text = (
        f"Edit pattern on {event_type} memory: " + "; ".join(observations) + "."
    )

    try:
        auto_capture(
            content=observation_text,
            event_type="user_preference",
            metadata={
                "source": "edit_observation",
                "derived_from": memory_id,
                "edited_event_type": event_type,
                "observation_type": "style",
            },
        )
        logger.info("Stored edit observation for %s: %s", memory_id[:12], observation_text[:80])
        return observation_text
    except Exception as e:
        logger.warning("Failed to store edit observation: %s", e)
        return None


# --- Welcome (peeled to cairn.bridge.welcome, Wave 3) ---
from cairn.bridge.welcome import welcome, get_session_context  # noqa: E402,F401


# --- Admin/profile (peeled to cairn.bridge.admin, Wave 5) ---
from cairn.bridge.admin import (  # noqa: E402,F401
    check_health, status, get_dedup_stats, export_memories, import_memories,
    deduplicate, extract_preferences, list_preferences, get_profile,
    save_profile, get_cross_session_lessons,
)
# --- Trajectory distillation (peeled to cairn.bridge.distillation, Wave 4) ---
from cairn.bridge.distillation import distill_trajectory  # noqa: E402,F401
# Public API -- Constraint Enforcement
# ---------------------------------------------------------------------------

CONSTRAINTS_DIR = CAIRN_HOME / "constraints"
# --- Constraints (peeled to cairn.bridge.constraints) ---
from cairn.bridge.constraints import (  # noqa: E402,F401
    _load_constraints, check_constraints, list_constraints, save_constraints,
)
# --- Learning (peeled to cairn.bridge.learning) ---
from cairn.bridge.learning import (  # noqa: E402,F401
    get_cross_project_lessons, reingest,
)
# --- Feedback (peeled to cairn.bridge.feedback) ---
from cairn.bridge.feedback import (  # noqa: E402,F401
    record_feedback, batch_record_feedback, _check_graduation, backfill_embeddings,
)


# ---------------------------------------------------------------------------
# Public API -- Session management
# ---------------------------------------------------------------------------


def clear_session(session_id: str) -> Dict[str, Any]:
    """Clear all memories for a session."""
    db = _get_store()
    count = db.clear_session(session_id)
    logger.info(f"Cleared session {session_id[:16]}: {count} memories removed")
    return {"session_id": session_id, "removed": count}


# ---------------------------------------------------------------------------
# Public API -- Batch operations
# ---------------------------------------------------------------------------


def batch_store(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Store multiple memories efficiently."""
    db = _get_store()
    ids = db.batch_store(items)
    return {"ids": ids, "count": len(ids)}
# --- Retrieval reads (peeled to cairn.bridge.retrieval, Wave 7) ---
from cairn.bridge.retrieval import (  # noqa: E402,F401
    find_similar_memories, get_memory, timeline, forgetting_log,
    traverse, phrase_search, regex_search,
)

# --- Maintenance (peeled to cairn.bridge.maintenance, Wave 6) ---
from cairn.bridge.maintenance import (  # noqa: E402,F401
    consolidate, compact, discover_connections, synthesize_system_insights,
    _smart_extract,  # underscore internal imported by tests
)


# --- Analytics (peeled to cairn.bridge.analytics, Wave 2) ---
from cairn.bridge.analytics import (  # noqa: E402,F401
    type_stats, stats_card_data, session_stats, retrieval_context,
    access_rate_stats, diagnostic_report, get_weekly_digest, get_activity_summary,
)
# --- Reminders (peeled to cairn.bridge.reminders, Wave 1) ---
from cairn.bridge.reminders import (  # noqa: E402,F401
    parse_duration, create_reminder, list_reminders,
    dismiss_reminder, get_due_reminders,
)
# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "auto_capture",
    "query",
    "query_structured",
    "check_health",
    "get_dedup_stats",
    "export_memories",
    "import_memories",
    "welcome",
    "get_profile",
    "save_profile",
    "remember",
    "store",
    "delete_memory",
    "edit_memory",
    "extract_preferences",
    "list_preferences",
    "deduplicate",
    "reingest",
    "status",
    "get_cross_session_lessons",
    "get_cross_project_lessons",
    "distill_trajectory",
    "reset_memory",
    "record_feedback",
    "clear_session",
    "batch_store",
    "find_similar_memories",
    "timeline",
    "consolidate",
    "traverse",
    "compact",
    "phrase_search",
    "type_stats",
    "session_stats",
    "check_constraints",
    "list_constraints",
    "save_constraints",
    "get_activity_summary",
    "get_weekly_digest",
    "parse_duration",
    "create_reminder",
    "list_reminders",
    "dismiss_reminder",
    "get_due_reminders",
]


def _install_bridge_submodule_aliases() -> None:
    """Expose 1.5.1-style bridge submodules for Pro compatibility.

    Core 1.5.0 shipped `cairn.bridge` as a flat module. Some Pro 1.5.x builds
    import helpers from `cairn.bridge._core`, `cairn.bridge._ingest`, and
    `cairn.bridge._query`. Until Core fully moves to a package layout, alias
    those submodules to this module so the published flat layout remains
    import-compatible.
    """
    import sys

    module = sys.modules[__name__]
    if not hasattr(module, "__path__"):
        module.__path__ = []  # type: ignore[attr-defined]
    for name in ("_core", "_ingest", "_query"):
        fullname = f"{__name__}.{name}"
        sys.modules.setdefault(fullname, module)
        setattr(module, name, module)


_install_bridge_submodule_aliases()
