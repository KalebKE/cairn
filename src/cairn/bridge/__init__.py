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
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from cairn.llm import llm_complete  # noqa: F401 — used in distill_trajectory, module-level for test patchability

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
# --- Core (peeled to cairn.bridge.core) ---
from cairn.bridge.core import (  # noqa: E402,F401
    auto_capture, store, remember, delete_memory, edit_memory, _extract_edit_observation, clear_session, batch_store,
)


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
