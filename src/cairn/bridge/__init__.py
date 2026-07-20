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

# Per-event-type dedup thresholds for Jaccard similarity.
DEDUP_THRESHOLDS: Dict[str, float] = {
    AutoCaptureEventType.ERROR_PATTERN: 0.70,
    AutoCaptureEventType.SESSION_SUMMARY: 0.75,
    AutoCaptureEventType.TASK_COMPLETION: 0.85,
    AutoCaptureEventType.DECISION: 0.80,
    AutoCaptureEventType.LESSON_LEARNED: 0.85,
    AutoCaptureEventType.CHECKPOINT: 0.90,
    AutoCaptureEventType.CONSTRAINT: 0.90,
    AutoCaptureEventType.ADVISOR_INSIGHT: 0.75,  # lowered from 0.85 to catch broader restatements
    AutoCaptureEventType.USER_FACT: 0.80,
    AutoCaptureEventType.SKILL_TEMPLATE: 0.85,
    AutoCaptureEventType.PROJECT_STATUS: 0.85,
    "memory": 0.80,  # Generic fallback type — dedup to prevent accumulation
}

# Event types that participate in memory evolution (Zettelkasten-style).
EVOLUTION_TYPES = {
    AutoCaptureEventType.LESSON_LEARNED,
    AutoCaptureEventType.DECISION,
    AutoCaptureEventType.ERROR_PATTERN,
    AutoCaptureEventType.CONSTRAINT,
    AutoCaptureEventType.SKILL_TEMPLATE,
    AutoCaptureEventType.PROJECT_STATUS,
    AutoCaptureEventType.ADVISOR_INSIGHT,
    AutoCaptureEventType.SESSION_SUMMARY,
}
EVOLUTION_THRESHOLD = 0.65

# Content blocklist — reject system noise at ingestion time.
# Startswith patterns (checked against content[:50])
_BLOCKLIST_STARTSWITH = [
    "[BROADCAST",
    "[WORK BREADCRUMB",
    "[WORK DISPATCH",
    "<task-notification>",
    "Decision: <task-notification>",
]
# Substring patterns (checked anywhere in content)
_BLOCKLIST_CONTAINS = [
    '"error":',
    '"stderr":',
    '"stdout":',
    "[BROADCAST",
]

# Minimum content length for auto-capture (reject very short noise).
# Raised from 40 to 80 to filter infrastructure noise that inflates never-accessed count.
_MIN_CONTENT_LENGTH = 80

# Event types from hooks that generate infrastructure noise (never accessed, inflate DB).
_INFRASTRUCTURE_EVENT_TYPES = frozenset({
    "consolidate", "compact", "checkpoint", "coordination_snapshot",
    "session_respawn", "file_summary", "code_chunk",
})


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


def semantic_search(
    query: str,
    top_k: int = 10,
    project: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform offline semantic (vector) search using the ONNX embed model.

    Requires the bridge to have been initialised with
    ``enable_vector_search=True`` and the underlying SQLiteStore to support
    the ``semantic_search`` method (sqlite-vec extension + ONNX runtime).

    Args:
        query:   Natural-language query string.
        top_k:   Maximum number of results to return.
        project: Optional project filter applied before vector ranking.

    Returns:
        List of memory dicts ordered by descending similarity score, each
        containing at least ``{id, content, score}``.

    Raises:
        RuntimeError: If vector search is not enabled or not supported by
            the current store backend.
    """
    if not _bridge_enable_vector_search:
        raise RuntimeError(
            "semantic_search requires enable_vector_search=True passed to "
            "initialize_bridge() before the store is created."
        )
    store = _get_store()
    if not hasattr(store, "semantic_search"):
        raise RuntimeError(
            "The current SQLiteStore backend does not expose semantic_search. "
            "Ensure sqlite-vec and onnxruntime are installed."
        )
    kwargs: Dict[str, Any] = {"query": query, "top_k": top_k}
    if project is not None:
        kwargs["project"] = project
    return store.semantic_search(**kwargs)


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


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for dedup comparison by stripping variable parts."""
    t = text.lower()
    t = re.sub(r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}", "<ID>", t)
    t = re.sub(r"/[\w/.\-]+\.\w{1,5}", "<PATH>", t)
    t = re.sub(r"'[^']{1,80}'", "<NAME>", t)
    t = re.sub(r'"[^"]{1,80}"', "<NAME>", t)
    t = re.sub(r"\b\d+\b", "<N>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


_TAG_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "ruby",
    "swift",
    "kotlin",
    "c++",
    "c#",
    "php",
    "sql",
    "bash",
    "zsh",
    "html",
    "css",
}
_TAG_TOOLS = {
    "react",
    "next.js",
    "nextjs",
    "django",
    "flask",
    "fastapi",
    "docker",
    "kubernetes",
    "git",
    "npm",
    "pip",
    "pytest",
    "webpack",
    "vite",
    "redis",
    "postgres",
    "sqlite",
    "mongodb",
    "aws",
    "gcp",
    "azure",
    "vercel",
    "nginx",
    "mysql",
    "jest",
    "vitest",
    "yarn",
    "pnpm",
    "bun",
    "deno",
    "supabase",
    "onnx",
    "mcp",
    "asyncio",
    "threading",
    "sqlalchemy",
    "celery",
    "graphql",
    "prisma",
    "terraform",
    "ansible",
    "helm",
    "zustand",
    "tailwind",
    "shadcn",
    "storybook",
    "playwright",
    "cypress",
    "openai",
    "anthropic",
    "langchain",
    "chromadb",
    "pinecone",
    "homebrew",
    "launchd",
    "systemd",
    "cron",
}
_TAG_ALIASES = {
    "postgresql": "postgres",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "tf": "terraform",
    "cdk": "aws-cdk",
    "nextjs": "next.js",
    "reactjs": "react",
    "sqlite3": "sqlite",
    "onnxruntime": "onnx",
    "fts5": "sqlite",
    "sqlitevec": "sqlite",
}
_GO_CONTEXT_WORDS = {"goroutine", "goroutines", "chan", "defer", "func", "gomod", "gofmt"}


_TAG_CONCEPTS = {
    "hook",
    "hooks",
    "daemon",
    "migration",
    "api",
    "config",
    "configuration",
    "testing",
    "test",
    "tests",
    "debug",
    "debugging",
    "performance",
    "cache",
    "caching",
    "auth",
    "authentication",
    "authorization",
    "deploy",
    "deployment",
    "refactor",
    "refactoring",
    "schema",
    "embedding",
    "embeddings",
    "vector",
    "coordination",
    "concurrency",
    "async",
    "sync",
}

# Map concept words to canonical tags
_CONCEPT_CANONICAL = {
    "hooks": "hook",
    "tests": "testing",
    "test": "testing",
    "debugging": "debug",
    "caching": "cache",
    "authentication": "auth",
    "authorization": "auth",
    "deployment": "deploy",
    "refactoring": "refactor",
    "configuration": "config",
    "embeddings": "embedding",
}


def _extract_tags(content: str, project: Optional[str] = None) -> List[str]:
    """Extract auto-tags from content (languages, tools, file paths, concepts, project)."""
    tags: set = set()
    words = set(re.findall(r"\b[\w.+#]+\b", content.lower()))
    # Apply aliases first (e.g. "postgresql" -> "postgres")
    for alias, canonical in _TAG_ALIASES.items():
        if alias in words:
            tags.add(canonical)
    # Languages (with Go disambiguation)
    for w in words:
        if w in _TAG_LANGUAGES:
            if w == "go":
                # Only tag "go" if Go-specific context words are present
                if words & _GO_CONTEXT_WORDS:
                    tags.add(w)
            else:
                tags.add(w)
    tags.update(w for w in words if w in _TAG_TOOLS)
    # Concepts (hook, testing, auth, etc.)
    for w in words:
        if w in _TAG_CONCEPTS:
            tags.add(_CONCEPT_CANONICAL.get(w, w))
    # File paths
    for match in re.findall(r"(?:/[\w.\-]+){2,}", content):
        tags.add(match)
    # File extensions mentioned inline (e.g. ".py", ".ts")
    for ext in re.findall(r"\b\w+\.(py|js|ts|tsx|rs|go|rb|java|swift|sql|sh|yaml|json|toml)\b", content.lower()):
        ext_map = {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
            "tsx": "typescript",
            "rs": "rust",
            "rb": "ruby",
            "sh": "bash",
        }
        if ext in ext_map:
            tags.add(ext_map[ext])
    # Project name
    if project:
        tags.add(Path(project).name.lower())
    return sorted(tags)[:10]


def _infer_temporal_range(query_text: str) -> Optional[tuple]:
    """Infer a (start_iso, end_iso) temporal range from natural-language time references.

    Supports: "last week", "yesterday", "N days/hours ago", "today",
    "this week/month/year", month names, day-of-week references,
    "the week/month of <date>", ISO dates.
    Returns None if no temporal signal is detected.
    """
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    text = query_text.lower()

    # "last N days/hours/weeks/months/years"
    m = re.search(r"last\s+(\d+)\s+(day|hour|week|month|year)s?", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    # "N days/hours ago"
    m = re.search(r"(\d+)\s+(day|hour|week|month|year)s?\s+ago", text)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {
            "day": timedelta(days=n),
            "hour": timedelta(hours=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
            "year": timedelta(days=n * 365),
        }[unit]
        return ((now - delta).isoformat(), now.isoformat())

    if "yesterday" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    if "today" in text:
        start = now.replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last week" in text:
        # Previous Mon-Sun week
        days_since_monday = now.weekday()
        last_monday = now - timedelta(days=days_since_monday + 7)
        start = last_monday.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=7)
        return (start.isoformat(), end.isoformat())

    if "this week" in text:
        days_since_monday = now.weekday()
        start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "last month" in text:
        # Previous calendar month
        first_this_month = now.replace(day=1, hour=0, minute=0, second=0)
        end = first_this_month
        if now.month == 1:
            start = datetime(now.year - 1, 12, 1, tzinfo=timezone.utc)
        else:
            start = datetime(now.year, now.month - 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    if "this month" in text:
        start = now.replace(day=1, hour=0, minute=0, second=0)
        return (start.isoformat(), now.isoformat())

    if "this year" in text:
        start = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), now.isoformat())

    if "last year" in text:
        start = datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)
        end = datetime(now.year, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    # Temporal context words: a bare month/weekday token is only treated as a
    # date reference when preceded by one of these — "we may need", "the march
    # of progress", and "monday-morning quarterbacking" must NOT trigger a
    # temporal range (they used to, skewing non-temporal knowledge queries).
    _TEMPORAL_PREP = r"(?:in|from|during|since|last|until|before|after|on|this|next|every)"

    # Day-of-week references: "last Monday", "on Friday", etc.
    _DAYS = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
             "friday": 4, "saturday": 5, "sunday": 6}
    for day_name, day_num in _DAYS.items():
        if re.search(rf"\b{_TEMPORAL_PREP}\s+{day_name}\b", text):
            # Find the most recent occurrence of this day
            days_ago = (now.weekday() - day_num) % 7
            if days_ago == 0:
                days_ago = 7  # "last Monday" means previous, not today
            target = now - timedelta(days=days_ago)
            start = target.replace(hour=0, minute=0, second=0)
            end = start + timedelta(days=1)
            return (start.isoformat(), end.isoformat())

    # "Month YYYY" or "in Month YYYY" (e.g., "January 2025")
    months = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for name, num in months.items():
        # Check for "Month YYYY" first
        m = re.search(rf"\b{name}\s+(\d{{4}})\b", text)
        if m:
            year = int(m.group(1))
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())
        # Bare month name (assume most recent occurrence) — requires a
        # temporal preposition so incidental words ("may", "march") don't fire.
        if re.search(rf"\b{_TEMPORAL_PREP}\s+{name}\b", text):
            year = now.year if num <= now.month else now.year - 1
            start = datetime(year, num, 1, tzinfo=timezone.utc)
            if num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, num + 1, 1, tzinfo=timezone.utc)
            return (start.isoformat(), end.isoformat())

    # ISO date (YYYY-MM-DD)
    m = re.search(r"(\d{4}-\d{2}-\d{2})", text)
    if m:
        date_str = m.group(1)
        start = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        end = start + timedelta(days=1)
        return (start.isoformat(), end.isoformat())

    # Bare year reference (e.g., "in 2024")
    m = re.search(r"\bin\s+(20\d{2})\b", text)
    if m:
        year = int(m.group(1))
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        return (start.isoformat(), end.isoformat())

    return None


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


def _extract_facts(content: str) -> List[str]:
    """Extract atomic facts from content for multi-key retrieval (no LLM).

    Extracts:
    - Technical terms (CamelCase, UPPER_CASE, dotted.paths)
    - Quoted strings and backtick-delimited tokens
    - Decision verbs with their objects ("chose X", "switched to Y")
    - Key noun phrases from short sentences

    Returns deduplicated list of fact strings, capped at 20.
    """
    facts: set = set()

    # 1. CamelCase identifiers (e.g., SQLiteStore, MemoryResult)
    # Match words starting with uppercase that have at least one lower-to-upper transition
    for m in re.findall(r"\b([A-Z][a-zA-Z]*[a-z][A-Z][a-zA-Z]*)\b", content):
        facts.add(m.lower())

    # 2. UPPER_CASE constants (e.g., MAX_NODES, API_KEY)
    for m in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", content):
        facts.add(m.lower())

    # 3. Backtick-delimited tokens (e.g., `jwt`, `sqlite_store.py`)
    for m in re.findall(r"`([^`]{2,40})`", content):
        facts.add(m.lower().strip())

    # 4. Quoted strings (e.g., "refresh token", 'auth method')
    for m in re.findall(r"""["']([^"']{2,40})["']""", content):
        facts.add(m.lower().strip())

    # 5. Decision/action verb phrases — extract the object of key verbs
    _DECISION_VERBS = (
        r"(?:chose|choose|decided|switched?\s+to|migrated?\s+to|"
        r"replaced?\s+with|use[ds]?|adopted?|selected?|"
        r"implemented?|configured?|set\s+up|enabled?|disabled?)"
    )
    for m in re.finditer(
        rf"\b{_DECISION_VERBS}\s+([A-Za-z0-9][\w\s./-]{{1,30}}?)(?:[.,;!?\n]|$)",
        content, re.IGNORECASE,
    ):
        phrase = m.group(1).strip().rstrip(".")
        if len(phrase) > 2:
            facts.add(phrase.lower())

    # 6. Dotted paths / module references (e.g., cairn.sqlite_store, src/cairn)
    for m in re.findall(r"\b([\w]+(?:\.[\w]+){1,4})\b", content):
        if not re.match(r"^\d+\.\d+", m):  # Skip version numbers like 1.0.0
            facts.add(m.lower())

    # 7. Technical compound terms (hyphenated, e.g., "multi-session", "cross-agent")
    for m in re.findall(r"\b([a-z]+-[a-z]+(?:-[a-z]+)?)\b", content.lower()):
        if len(m) > 4:
            facts.add(m)

    # Filter out very short or stopword-only facts
    _STOP = {"the", "and", "for", "with", "that", "this", "from", "have", "been", "will", "not"}
    filtered = []
    for f in facts:
        words = f.split()
        meaningful = [w for w in words if w not in _STOP and len(w) > 1]
        if meaningful:
            filtered.append(f)

    return sorted(set(filtered))[:20]


def _compress_to_observation(content: str, event_type: str = "") -> Optional[str]:
    """Compress content to a concise observation (extractive, no LLM).

    Selects the 1-2 most information-dense sentences from the content.
    Returns None if content is already concise (< 150 chars) or compression fails.
    """
    if len(content) < 150:
        return None  # Already concise

    # Split into sentences (preserve abbreviations, version numbers)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", content)
    if len(sentences) <= 1:
        # Try simpler split
        sentences = re.split(r"(?<=[.!?])\s+", content)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 15]

    if not sentences:
        return None

    # Score each sentence by information density
    scored = []
    for s in sentences:
        words = s.split()
        unique_words = len(set(w.lower() for w in words if len(w) > 3))
        # Bonus for code tokens (backticks, paths, CamelCase)
        code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", s))
        # Diminishing returns on length
        length_score = min(len(s), 200) / 200.0
        density = unique_words * 1.0 + code_tokens * 2.0 + length_score * 3.0
        scored.append((density, s))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Select top 1-2 diverse sentences
    selected = [scored[0][1]]
    if len(scored) > 1:
        # Add second only if sufficiently different
        s2 = scored[1][1]
        if _jaccard(selected[0].lower(), s2.lower(), min_word_len=3) < 0.7:
            selected.append(s2)

    observation = " ".join(selected)
    if len(observation) > 200:
        observation = observation[:197] + "..."

    return observation


def _jaccard(text_a: str, text_b: str, min_word_len: int = 4) -> float:
    """Jaccard similarity on word sets (fast, no embeddings)."""
    words_a = {w for w in text_a.split() if len(w) >= min_word_len}
    words_b = {w for w in text_b.split() if len(w) >= min_word_len}
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _auto_relate(store, node_id: str, max_related: int = 3, min_similarity: float = 0.65) -> int:
    """Create typed edges from node_id to its most similar existing memories.

    Edge types are inferred from metadata signals:
      - same_entity: both memories share the same entity_id
      - evolution: same event_type with high similarity (concept update)
      - temporal_cluster: created within 1 hour of each other
      - related: fallback for very high similarity (>= 0.80) with no stronger type

    Returns the number of edges created. Silently returns 0 on any error.
    """
    try:
        embedding = store.get_embedding(node_id)
        if not embedding:
            return 0
        similar = store.find_similar(embedding, limit=max_related + 1)
        candidates = [r for r in similar if r.id != node_id and r.relevance >= min_similarity][:max_related]
        if not candidates:
            return 0

        source_node = store.get_node(node_id)
        if not source_node:
            return 0
        src_meta = source_node.metadata or {}
        src_entity = src_meta.get("entity_id", "")
        src_event = src_meta.get("event_type", "")
        src_created = source_node.created_at

        count = 0
        for r in candidates:
            r_meta = r.metadata or {}
            r_entity = r_meta.get("entity_id", "")
            r_event = r_meta.get("event_type", "")

            # Classify edge type from strongest to weakest signal
            if src_entity and r_entity and src_entity == r_entity:
                edge_type = "same_entity"
            elif src_event and r_event and src_event == r_event and r.relevance >= 0.75:
                edge_type = "evolution"
            elif src_created and r.created_at:
                delta = abs((src_created - r.created_at).total_seconds())
                if delta <= 3600:
                    edge_type = "temporal_cluster"
                elif r.relevance >= 0.80:
                    edge_type = "related"
                else:
                    continue  # Below 0.80 with no typed signal: skip
            elif r.relevance >= 0.80:
                edge_type = "related"
            else:
                continue  # No strong signal: skip the generic edge

            if store.add_edge(node_id, r.id, edge_type, r.relevance):
                count += 1
        if count:
            logger.debug(f"Auto-related {node_id[:12]} to {count} memories (typed)")
        return count
    except Exception as e:
        logger.debug(f"_auto_relate failed for {node_id[:12]}: {e}")
        return 0


def _schedule_auto_relate(store, node_id: str) -> None:
    """Fire _auto_relate in a background daemon thread (non-blocking).

    Set ``CAIRN_AUTO_RELATE=0`` (also ``false``/``off``) to skip enrichment
    entirely — parity with ``CAIRN_ENTITY_EXTRACTION``. Requested by issue #58
    reporter as a safety valve for environments where auto-relate's thread
    behavior is undesirable.

    Registers the thread on the store so close() can join it before tearing
    down the sqlite connection (prevents use-after-close segfaults in
    sqlite-vec native code during test teardown).
    """
    if os.environ.get("CAIRN_AUTO_RELATE", "").lower() in ("0", "false", "off"):
        return

    t_ref: list[threading.Thread] = []

    def _run():
        try:
            _auto_relate(store, node_id)
        except Exception as e:
            logger.debug(f"Background _auto_relate failed for {node_id[:12]}: {e}")
        finally:
            if t_ref and hasattr(store, "unregister_background_thread"):
                try:
                    store.unregister_background_thread(t_ref[0])
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True, name="auto-relate")
    t_ref.append(t)
    if hasattr(store, "register_background_thread"):
        store.register_background_thread(t)
    t.start()


_CROSS_TYPE_SUPERSEDE = {
    "user_preference": {"decision"},
}


def _detect_and_supersede(
    store, node_id: str, content: str, event_type: str,
    entity_id: Optional[str] = None,
) -> int:
    """Detect contradicting memories and mark old ones as superseded.

    Only runs for decision, user_preference, user_fact types.
    Uses embedding similarity to find candidates, then checks for topic
    overlap with different content — indicating a contradiction/update.

    Cross-type supersession: user_preference can supersede decision memories
    (e.g. "stop suggesting HN" supersedes "post Show HN on Tuesday").

    Returns count of superseded memories.
    """
    _SUPERSEDE_TYPES = {"decision", "user_preference", "user_fact"}
    if event_type not in _SUPERSEDE_TYPES:
        return 0
    try:
        embedding = store.get_embedding(node_id)
        if not embedding:
            return 0
        similar = store.find_similar(embedding, limit=5)
        superseded = 0
        content_norm = content[:100].strip().lower()
        cross_targets = _CROSS_TYPE_SUPERSEDE.get(event_type)
        for r in similar:
            if r.id == node_id:
                continue
            if (r.metadata or {}).get("superseded"):
                continue
            r_type = (r.metadata or {}).get("event_type", "")
            if r_type != event_type:
                if not cross_targets or r_type not in cross_targets:
                    continue
            if r.relevance < 0.80:
                continue
            if entity_id:
                r_entity = (r.metadata or {}).get("entity_id", "")
                if r_entity and r_entity != entity_id:
                    continue
            existing_norm = r.content[:100].strip().lower()
            if content_norm == existing_norm:
                continue
            store.mark_superseded(r.id, superseded_by=node_id)
            store.add_edge(node_id, r.id, "supersedes", r.relevance)
            superseded += 1
            logger.info(
                "Ingest superseded %s (sim=%.2f) by %s",
                r.id[:12], r.relevance, node_id[:12],
            )
        if superseded:
            store.stats.setdefault("ingest_superseded", 0)
            store.stats["ingest_superseded"] += superseded
        return superseded
    except Exception as e:
        logger.debug("_detect_and_supersede failed for %s: %s", node_id[:12], e)
        return 0


def _split_atomic_facts(content: str, event_type: str) -> List[str]:
    """Extract sentence-level atomic facts from content.

    Identifies standalone factual statements for storage as separate
    user_fact nodes to improve single-mention recall.

    Returns list of fact strings (max 5).

    Gated behind CAIRN_ATOMIC_FACTS=1 (off by default).
    """
    if os.environ.get("CAIRN_ATOMIC_FACTS", "0") != "1":
        return []
    if len(content) < 50:
        return []
    # Only split facts from user-authored types, not agent-generated content
    _FACT_SPLIT_TYPES = {
        "decision", "user_fact",
    }
    if event_type not in _FACT_SPLIT_TYPES:
        return []
    facts = []
    sentences = re.split(r"(?<=[.!?])\s+", content)
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15 or len(sentence) > 200:
            continue
        s_lower = sentence.lower()
        # Require first-person or infrastructure context to confirm it's a user fact,
        # not an agent observation like "The component was a God object"
        _has_user_signal = bool(re.search(
            r"\b(?:we|our|my|i)\s+|"
            r"\b(?:the|our)\s+(?:db|database|server|api|config|project|repo|app|service|endpoint|url|path|port)\b",
            s_lower,
        ))
        if not _has_user_signal:
            continue
        if re.search(r"\b(?:is|are|was|were)\s+\w", s_lower):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:we\s+use|using|uses?|adopted?|switched?\s+to)\s+\w",
            s_lower,
        ):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:moved?\s+to|lives?\s+in|located?\s+(?:in|at)"
            r"|based\s+in)\s+\w",
            s_lower,
        ):
            facts.append(sentence)
            continue
        if re.search(
            r"\b(?:password|key|token|secret|api.?key)"
            r"\s+(?:is|=|:)\s*\S",
            s_lower,
        ):
            facts.append(sentence)
            continue
    seen = set()
    unique = []
    for f in facts:
        f_norm = f.strip().lower()
        if f_norm not in seen:
            seen.add(f_norm)
            unique.append(f)
    return unique[:5]


def _schedule_entity_extraction(
    store: Any,
    node_id: str,
    content: str,
    event_type: str,
) -> None:
    """Fire entity extraction in a background daemon thread.

    Non-blocking. Silently skipped if API key missing or extraction disabled.
    """
    import os as _os
    if _os.environ.get("CAIRN_ENTITY_EXTRACTION", "").lower() in ("0", "false", "off"):
        return
    if not _os.environ.get("ANTHROPIC_API_KEY"):
        return

    t_ref: list[threading.Thread] = []

    def _run():
        try:
            from cairn_platform.entity.extraction import extract_entities, resolve_and_link
            from cairn_platform.entity.engine import get_entity_manager
            from pathlib import Path as _Path

            extraction = extract_entities(content, event_type)
            if extraction["entities"]:
                em = get_entity_manager(_Path(store.db_path))
                resolve_and_link(store, em, node_id, extraction)
        except Exception as e:
            logger.debug("Async entity extraction failed: %s", e)
        finally:
            if t_ref and hasattr(store, "unregister_background_thread"):
                try:
                    store.unregister_background_thread(t_ref[0])
                except Exception:
                    pass

    t = threading.Thread(target=_run, daemon=True, name="entity-extraction")
    t_ref.append(t)
    if hasattr(store, "register_background_thread"):
        store.register_background_thread(t)
    t.start()


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


# ---------------------------------------------------------------------------
# Public API -- Query
# ---------------------------------------------------------------------------


def query(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    scope: Optional[str] = None,
    surfacing_context: Optional[Any] = None,
    perspective: Optional[str] = None,
    strength_min: Optional[float] = None,
    memory_type: Optional[str] = None,
    include_contradicted: bool = False,
    valid_at: Optional[str] = None,
    status: Optional[str] = None,
) -> str:
    """Search memories with optional intent-aware routing.

    Args:
        context_file: Current file being edited (for contextual re-ranking).
        context_tags: Current context tags like language, tools (for re-ranking boost).
        filter_tags: Hard filter -- only return memories containing ALL specified tags.
        temporal_range: Optional (start_iso, end_iso) tuple. Auto-inferred from query if not given.
        surfacing_context: SurfacingContext enum for context-aware scoring (error_debug, planning, etc.).
        strength_min: Minimum strength score (0.0-1.0). Filters out weak/decayed memories.

    Returns:
        Formatted markdown string with results.
    """
    db = _get_store()
    query_text = unicodedata.normalize("NFC", query_text)

    try:
        # Auto-infer temporal range from query text if not explicitly provided
        effective_temporal = temporal_range or _infer_temporal_range(query_text)
        # When range was auto-inferred, use soft scoring (boost only, no harsh penalty)
        _temporal_boost_only = temporal_range is None and effective_temporal is not None

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        # Pass scope through to store; "session" restricts to caller's session
        _scope = scope if scope in ("project", "session") else "project"
        query_kwargs: Dict[str, Any] = {
            "limit": limit * 3 if (filter_tags or entity_id or agent_type) else limit,
            "session_id": session_id,
            "context_file": context_file or "",
            "context_tags": context_tags,
            "temporal_range": effective_temporal,
            "entity_id": entity_id,
            "agent_type": agent_type,
            "query_hint": event_type,
            "temporal_boost_only": _temporal_boost_only,
            "scope": _scope,
        }
        if surfacing_context is not None:
            query_kwargs["surfacing_context"] = surfacing_context
        if perspective:
            query_kwargs["perspective"] = perspective
        if valid_at:
            query_kwargs["valid_at"] = valid_at
        results = db.query(enhanced, **query_kwargs)

        # Filter by event_type if specified
        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        # Filter by memory_type
        if memory_type and results:
            results = [r for r in results if (r.metadata or {}).get("memory_type") == memory_type]

        # Filter by lifecycle status (active, superseded, speculative, archived)
        if status and results:
            results = [r for r in results if (r.metadata or {}).get("status", "active") == status]

        # Filter to only contradicted memories
        if include_contradicted and results:
            results = [r for r in results if (r.metadata or {}).get("contradicted_by")]

        results = results[:limit]

        # Filter by minimum strength score
        if strength_min is not None and strength_min > 0:
            results = [r for r in results if getattr(r, "strength", 0.0) >= strength_min]

        # Extract query confidence from results
        _qconf = None
        if results:
            _qconf = (results[0].metadata or {}).get("_query_confidence")

        # Format
        _conf_label = ""
        if _qconf is not None and _qconf < 0.3:
            _conf_label = " (confidence: low -- results may not be relevant)"
        elif _qconf is not None and _qconf <= 0.7:
            _conf_label = " (confidence: medium)"
        output = f"Results: {len(results)}{_conf_label}\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                _str = getattr(node, "strength", 0.0)
                _meta = node.metadata or {}
                _status = _meta.get("status", "active")
                _status_tag = f" [{_status}]" if _status != "active" else ""
                output += f"## {i}. [{ntype}] `{node.id}` (str: {_str:.2f}){_status_tag}\n"
                output += f"{preview}\n"
                created = node.created_at.isoformat()[:16] if node.created_at else ""
                _extras = []
                if _meta.get("source_uri"):
                    _extras.append(f"source: {_meta['source_uri']}")
                if _meta.get("derived_from"):
                    _extras.append(f"derived from: {_meta['derived_from']}")
                _extras_str = f" | {' | '.join(_extras)}" if _extras else ""
                output += f"*{created}{_extras_str}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        # Auto-inject relevant constraints (always, regardless of event_type filter)
        if event_type != "constraint":
            try:
                result_ids = {n.id for n in results}
                constraint_nodes = db.get_by_type("constraint", limit=10)
                matching_constraints = []
                if constraint_nodes:
                    query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                    for cn in constraint_nodes:
                        if cn.id in result_ids:
                            continue
                        if (cn.metadata or {}).get("superseded"):
                            continue
                        content_words = {w.lower() for w in cn.content.split() if len(w) > 2}
                        if query_words & content_words:
                            matching_constraints.append(cn)
                if matching_constraints:
                    output += "\n---\n**Active Constraints:**\n"
                    for cr in matching_constraints[:3]:
                        preview = cr.content[:150]
                        output += f"- [`{cr.id}`] {preview}\n"
            except Exception as e:
                logger.debug("Constraint injection failed: %s", e)

        # Auto-inject relevant user preferences for preference-intent queries
        _PREF_SIGNAL_WORDS = {
            "rule", "rules", "preference", "setting", "configured",
            "should", "allowed", "policy", "default", "location",
            "timezone", "where", "how",
        }
        if event_type != "user_preference":
            try:
                query_words_lower = {w.lower().rstrip("?.,!") for w in query_text.split() if len(w) > 1}
                if query_words_lower & _PREF_SIGNAL_WORDS:
                    result_ids = {n.id for n in results}
                    pref_nodes = db.get_by_type("user_preference", limit=20)
                    matching_prefs = []
                    if pref_nodes:
                        query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                        for pn in pref_nodes:
                            if pn.id in result_ids:
                                continue
                            if (pn.metadata or {}).get("superseded"):
                                continue
                            content_words = {w.lower() for w in pn.content.split() if len(w) > 2}
                            if query_words & content_words:
                                matching_prefs.append(pn)
                    if matching_prefs:
                        output += "\n---\n**User Preferences:**\n"
                        for pr in matching_prefs[:3]:
                            preview = pr.content[:150]
                            output += f"- [`{pr.id}`] {preview}\n"
            except Exception as e:
                logger.debug("Preference injection failed: %s", e)

        # Warn if embedding model is degraded (hash fallback active)
        try:
            from cairn.embedding import is_embedding_degraded
            if is_embedding_degraded() and results:
                output += "\n**Note:** Semantic search is degraded (embedding model unavailable). Results are text-match only.\n"
        except Exception as e:
            logger.warning("Embedding degradation check failed: %s", e)

        logger.info(f"Query '{query_text[:30]}...' returned {len(results)} results")
        return output

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return f"# Query Error\n\n**Error:** {str(e)}\n"


def query_structured(
    query_text: str,
    limit: int = 10,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
    event_type: Optional[str] = None,
    context_file: Optional[str] = None,
    context_tags: Optional[List[str]] = None,
    filter_tags: Optional[List[str]] = None,
    temporal_range: Optional[tuple] = None,
    entity_id: Optional[str] = None,
    agent_type: Optional[str] = None,
    surfacing_context: Optional["SurfacingContext"] = None,
    strength_min: Optional[float] = None,
    memory_type: Optional[str] = None,
    include_contradicted: bool = False,
    valid_at: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Query memories and return structured dicts (machine-readable)."""
    db = _get_store()

    try:
        effective_temporal = temporal_range or _infer_temporal_range(query_text)
        _temporal_boost_only = temporal_range is None and effective_temporal is not None

        enhanced = query_text
        if event_type:
            enhanced = f"{event_type} {enhanced}"
        if project:
            enhanced = f"{Path(project).name} {enhanced}"

        query_kwargs_s: Dict[str, Any] = {
            "limit": limit * 2 if (filter_tags or entity_id or agent_type) else limit,
            "session_id": session_id,
            "context_file": context_file or "",
            "context_tags": context_tags,
            "temporal_range": effective_temporal,
            "entity_id": entity_id,
            "agent_type": agent_type,
            "query_hint": event_type,
            "surfacing_context": surfacing_context,
            "temporal_boost_only": _temporal_boost_only,
        }
        if valid_at:
            query_kwargs_s["valid_at"] = valid_at
        results = db.query(enhanced, **query_kwargs_s)

        if event_type and results:
            results = [r for r in results if (r.metadata or {}).get("event_type") == event_type]

        # Hard filter by tags (AND logic — all specified tags must be present)
        if filter_tags and results:
            filter_set = {t.lower() for t in filter_tags}
            results = [
                r for r in results if filter_set.issubset({str(t).lower() for t in (r.metadata or {}).get("tags", [])})
            ]

        # Filter by memory_type
        if memory_type and results:
            results = [r for r in results if (r.metadata or {}).get("memory_type") == memory_type]

        # Filter to only contradicted memories
        if include_contradicted and results:
            results = [r for r in results if (r.metadata or {}).get("contradicted_by")]

        results = results[:limit]

        # Filter by minimum strength score
        if strength_min is not None and strength_min > 0:
            results = [r for r in results if getattr(r, "strength", 0.0) >= strength_min]

        structured = []
        for node in results:
            structured.append(
                {
                    "id": node.id,
                    "content": node.content,
                    "event_type": (node.metadata or {}).get("event_type", "memory"),
                    "session_id": (node.metadata or {}).get("session_id", ""),
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "tags": (node.metadata or {}).get("tags", []),
                    "metadata": node.metadata,
                    "relevance": getattr(node, "relevance", 0.0),
                    "_query_confidence": (node.metadata or {}).get("_query_confidence", 0.0),
                    "strength": round(getattr(node, "strength", 0.0), 3),
                    "valid_from": node.valid_from.isoformat() if hasattr(node, "valid_from") and node.valid_from else None,
                    "valid_until": node.valid_until.isoformat() if hasattr(node, "valid_until") and node.valid_until else None,
                }
            )

        # Auto-inject relevant constraints
        if event_type != "constraint":
            try:
                result_ids = {node.id for node in results}
                constraint_nodes = db.get_by_type("constraint", limit=10)
                if constraint_nodes:
                    query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                    injected = 0
                    for cn in constraint_nodes:
                        if cn.id in result_ids:
                            continue
                        if (cn.metadata or {}).get("superseded"):
                            continue
                        content_words = {w.lower() for w in cn.content.split() if len(w) > 2}
                        if query_words & content_words:
                            structured.insert(0, {
                                "id": cn.id,
                                "content": cn.content,
                                "event_type": "constraint",
                                "session_id": (cn.metadata or {}).get("session_id", ""),
                                "created_at": cn.created_at.isoformat() if cn.created_at else "",
                                "tags": (cn.metadata or {}).get("tags", []),
                                "metadata": cn.metadata,
                                "relevance": getattr(cn, "relevance", 0.0),
                                "is_constraint": True,
                            })
                            injected += 1
                            if injected >= 3:
                                break
            except Exception as e:
                logger.debug("Constraint injection failed (structured): %s", e)

        # Auto-inject relevant user preferences for preference-intent queries
        _PREF_SIGNAL_WORDS_S = {
            "rule", "rules", "preference", "setting", "configured",
            "should", "allowed", "policy", "default", "location",
            "timezone", "where", "how",
        }
        if event_type != "user_preference":
            try:
                query_words_lower = {w.lower().rstrip("?.,!") for w in query_text.split() if len(w) > 1}
                if query_words_lower & _PREF_SIGNAL_WORDS_S:
                    result_ids = {node.id for node in results}
                    pref_nodes = db.get_by_type("user_preference", limit=20)
                    if pref_nodes:
                        query_words = {w.lower() for w in query_text.split() if len(w) > 2}
                        injected = 0
                        for pn in pref_nodes:
                            if pn.id in result_ids:
                                continue
                            if (pn.metadata or {}).get("superseded"):
                                continue
                            content_words = {w.lower() for w in pn.content.split() if len(w) > 2}
                            if query_words & content_words:
                                structured.insert(0, {
                                    "id": pn.id,
                                    "content": pn.content,
                                    "event_type": "user_preference",
                                    "session_id": (pn.metadata or {}).get("session_id", ""),
                                    "created_at": pn.created_at.isoformat() if pn.created_at else "",
                                    "tags": (pn.metadata or {}).get("tags", []),
                                    "metadata": pn.metadata,
                                    "relevance": getattr(pn, "relevance", 0.0),
                                    "is_preference": True,
                                })
                                injected += 1
                                if injected >= 3:
                                    break
            except Exception as e:
                logger.debug("Preference injection failed (structured): %s", e)

        return structured

    except Exception as e:
        logger.error(f"Structured query failed: {e}", exc_info=True)
        return []


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


def _load_constraints(project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load constraint rules for a project from ~/.cairn/constraints/.

    Loads global.json first, then <project-name>.json if project is given.
    Returns merged list of rule dicts.
    """
    rules: List[Dict[str, Any]] = []
    if not CONSTRAINTS_DIR.exists():
        return rules

    # Global constraints
    global_file = CONSTRAINTS_DIR / "global.json"
    if global_file.exists():
        try:
            data = json.loads(global_file.read_text())
            for r in data.get("rules", []):
                r["source"] = "global"
                rules.append(r)
        except Exception as e:
            logger.debug(f"Failed to load global constraints: {e}")

    # Project-specific constraints
    if project:
        proj_name = Path(project).name
        proj_file = CONSTRAINTS_DIR / f"{proj_name}.json"
        if proj_file.exists():
            try:
                data = json.loads(proj_file.read_text())
                for r in data.get("rules", []):
                    r["source"] = proj_name
                    rules.append(r)
            except Exception as e:
                logger.debug(f"Failed to load {proj_name} constraints: {e}")

    return rules


def check_constraints(file_path: str, project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check a file path against loaded constraint rules.

    Returns list of matching constraints with severity and message.
    """
    import fnmatch

    rules = _load_constraints(project)
    if not rules:
        return []

    matches = []
    filename = os.path.basename(file_path)

    for rule in rules:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        # Match against filename or full path
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
            matches.append(
                {
                    "pattern": pattern,
                    "constraint": rule.get("constraint", ""),
                    "severity": rule.get("severity", "warn"),
                    "source": rule.get("source", "unknown"),
                }
            )

    return matches


def list_constraints(project: Optional[str] = None) -> Dict[str, Any]:
    """List all loaded constraint rules for a project."""
    rules = _load_constraints(project)
    return {
        "count": len(rules),
        "rules": rules,
        "constraints_dir": str(CONSTRAINTS_DIR),
    }


def save_constraints(
    rules: List[Dict[str, Any]],
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Save constraint rules to the appropriate file.

    If project is given, saves to <project-name>.json, else global.json.
    """
    CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if project:
        target = CONSTRAINTS_DIR / f"{Path(project).name}.json"
    else:
        target = CONSTRAINTS_DIR / "global.json"

    # Clean source field from rules before saving
    clean_rules = []
    for r in rules:
        clean = {k: v for k, v in r.items() if k != "source"}
        clean_rules.append(clean)

    data = {"rules": clean_rules}
    target.write_text(json.dumps(data, indent=2))

    return {"saved": str(target), "count": len(clean_rules)}


# ---------------------------------------------------------------------------
# Public API -- Cross-project Learning
# ---------------------------------------------------------------------------


def get_cross_project_lessons(
    task: Optional[str] = None,
    exclude_project: Optional[str] = None,
    exclude_session: Optional[str] = None,
    limit: int = 5,
    agent_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve lessons from ALL projects (cross-project knowledge transfer).

    Unlike get_cross_session_lessons which may filter by project,
    this explicitly queries without project scope to find patterns
    that recur across different codebases.
    """
    db = _get_store()
    lessons: List[Dict[str, Any]] = []
    seen_content: set = set()
    project_sets: Dict[str, set] = {}

    try:
        if task:
            nodes = db.query_by_type(query=task, event_type="lesson_learned", limit=limit * 4)
        else:
            nodes = db.get_by_type("lesson_learned", limit=limit * 4)

        for node in nodes:
            meta = node.metadata or {}
            node_project = meta.get("project", "")

            if exclude_session and meta.get("session_id") == exclude_session:
                continue
            if exclude_project and node_project == exclude_project:
                continue
            if agent_type and meta.get("agent_type") != agent_type:
                continue

            key = node.content[:80].lower()

            if key in seen_content:
                if node_project and key in project_sets:
                    project_sets[key].add(node_project)
                continue

            seen_content.add(key)
            project_sets[key] = {node_project} if node_project else set()

            lessons.append(
                {
                    "content": node.content,
                    "source_project": node_project,
                    "lesson_id": meta.get("lesson_id") or node.id,
                    "session_id": meta.get("session_id", ""),
                    "access_count": getattr(node, "access_count", 0) or 0,
                    "created_at": node.created_at.isoformat() if node.created_at else "",
                    "projects_seen": 1,
                    "_key": key,
                }
            )
    except Exception as e:
        logger.debug(f"Cross-project lesson query failed: {e}")

    # Enrich with cross-project counts
    for lesson in lessons:
        key = lesson.get("_key", "")
        proj_count = len(project_sets.get(key, set()))
        lesson["projects_seen"] = max(1, proj_count)
        lesson["cross_project"] = proj_count > 1
        lesson.pop("_key", None)

    # Sort by cross-project occurrence, then access count
    lessons.sort(
        key=lambda lesson: (lesson.get("projects_seen", 0), lesson.get("access_count", 0)),
        reverse=True,
    )

    return lessons[:limit]


# ---------------------------------------------------------------------------
# Public API -- Reingest (legacy JSONL → SQLite)
# ---------------------------------------------------------------------------


def reingest(
    store_path: Optional[Path] = None,
    batch_size: int = 50,
    skip_types: Optional[set] = None,
) -> Dict[str, Any]:
    """Bulk-load JSONL store entries into SQLite.

    Reads every line from store.jsonl and inserts into the SQLite database.
    Content-hash dedup prevents duplicates automatically.
    """
    db = _get_store()
    src = store_path or (CAIRN_HOME / "store.jsonl")

    if not src.exists():
        return {"error": f"Store file not found: {src}", "ingested": 0}

    skip_types = skip_types or set()
    stats = {"ingested": 0, "skipped": 0, "duplicates": 0, "errors": 0, "total": 0}

    logger.info(f"Reingesting from {src}")

    from cairn.crypto import decrypt_line

    with open(src, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            stats["total"] += 1

            try:
                entry = json.loads(decrypt_line(line))
            except Exception as e:
                logger.debug("Import line parse failed at line %d: %s", line_num, e)
                stats["errors"] += 1
                continue

            content = entry.get("content", "").strip()
            if not content:
                stats["skipped"] += 1
                continue

            meta = entry.get("metadata", {})
            event_type = meta.get("event_type", "memory")

            if event_type in skip_types:
                stats["skipped"] += 1
                continue

            session_id = meta.get("session_id")
            ttl = TTLCategory.for_event_type(event_type)

            try:
                db.store(
                    content=content[:2000],
                    session_id=session_id,
                    metadata=meta,
                    ttl_seconds=ttl,
                    skip_inference=True,
                )
                stats["ingested"] += 1
            except Exception as e:
                stats["errors"] += 1
                if stats["errors"] <= 5:
                    logger.warning(f"Reingest error line {line_num}: {e}")

            if stats["ingested"] > 0 and stats["ingested"] % batch_size == 0:
                logger.info(f"  Progress: {stats['ingested']} ingested, {stats['total']} processed")

    logger.info(
        f"Reingest complete: {stats['ingested']} ingested, "
        f"{stats['duplicates']} duplicates, {stats['errors']} errors "
        f"out of {stats['total']} entries"
    )
    return stats


# ---------------------------------------------------------------------------
# Public API -- Feedback
# ---------------------------------------------------------------------------


def record_feedback(
    memory_id: str,
    rating: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Record feedback on a surfaced memory."""
    db = _get_store()
    return db.record_feedback(node_id=memory_id, rating=rating, reason=reason)


def batch_record_feedback(items: List[tuple]) -> int:
    """Record feedback for multiple memories in a single transaction.

    Each item is (node_id, rating, reason). Returns count of updated memories.
    """
    db = _get_store()
    return db.batch_record_feedback(items)


def _check_graduation(memory_id: str) -> Optional[str]:
    """Check if a memory should graduate or decay based on diff-correlation history.

    Graduation: memory was diff-correlated (positive) in 2+ feedback signals -> promote priority.
    Decay: memory was surfaced 3+ times with zero correlation -> demote priority.

    Reads from the feedback_signals list stored in memory metadata by record_feedback().

    Returns "graduated", "decayed", or None.
    """
    db = _get_store()
    try:
        row = db._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?",
            (memory_id,),
        ).fetchone()

        if not row or not row[0]:
            return None

        meta = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
        signals = meta.get("feedback_signals", [])

        if not signals:
            return None

        diff_positive = sum(
            1 for s in signals
            if s.get("rating") == "helpful" and s.get("reason") and "diff-correlated" in s["reason"]
        )
        surfaced_not_committed = sum(
            1 for s in signals
            if s.get("rating") == "unhelpful" and s.get("reason") and "not committed" in s["reason"]
        )

        if diff_positive >= 2:
            # Graduate: boost priority
            db._conn.execute(
                "UPDATE memories SET priority = MIN(COALESCE(priority, 3) + 1, 5) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "graduated"
        elif surfaced_not_committed >= 3 and diff_positive == 0:
            # Decay: reduce priority
            db._conn.execute(
                "UPDATE memories SET priority = MAX(COALESCE(priority, 3) - 1, 1) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "decayed"

        return None
    except Exception as e:
        logger.debug("_check_graduation failed for %s: %s", memory_id[:12], e)
        return None


def backfill_embeddings(batch_size: int = 50) -> dict:
    """Backfill missing embeddings for memories not in memories_vec."""
    db = _get_store()
    return db.backfill_embeddings(batch_size=batch_size)


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


# ---------------------------------------------------------------------------
# Public API -- Similar memories
# ---------------------------------------------------------------------------


def find_similar_memories(memory_id: str, limit: int = 5) -> str:
    """Find memories similar to a given memory via vector search."""
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."
    embedding = db.get_embedding(memory_id)
    if embedding is None:
        return f"No embedding found for `{memory_id[:12]}`. Vector search unavailable."
    # limit+1 because the source memory will be in results
    results = db.find_similar(embedding, limit=limit + 1)
    # Filter out the source memory itself
    results = [r for r in results if r.id != memory_id][:limit]
    # Format output
    output = f"# Similar Memories ({len(results)})\n\n"
    output += f"**Source:** `{memory_id[:12]}` — {node.content[:100]}\n\n"
    for i, r in enumerate(results, 1):
        ntype = (r.metadata or {}).get("event_type", "memory")
        preview = r.content[:200]
        output += f"## {i}. [{ntype}] `{r.id[:12]}` (similarity: {r.relevance:.2f})\n"
        output += f"{preview}\n\n"
    if not results:
        output += "*No similar memories found.*\n"
    return output


def get_memory(memory_id: str, include_related: bool = True) -> Dict[str, Any]:
    """Fetch one memory by full id or unique id prefix (>=4 chars; the
    surfaced-prefix contract is >=8). The hydration half of the
    pointers-not-payloads loop: surfacing blocks and context packets emit
    truncated ids, this resolves them to full untruncated content.

    Returns one of:
      {"ambiguous": [candidate_ids]}   — prefix matched several memories
      {"error": "..."}                 — nothing matched
      {full memory dict}               — unique hit (includes status and,
                                          unlike search paths, is returned
                                          even for archived/superseded rows)
    """
    db = _get_store()
    node, candidates = db.get_node_by_prefix((memory_id or "").strip())
    if candidates:
        return {"ambiguous": candidates}
    if node is None:
        return {"error": f"Memory '{memory_id}' not found"}

    meta = dict(node.metadata or {})
    with db._lock:
        row = db._conn.execute(
            """SELECT event_type, session_id, project, entity_id, status
               FROM memories WHERE node_id = ?""",
            (node.id,),
        ).fetchone()
    event_type, session_id, project, entity_id, status = row if row else (None,) * 5

    result: Dict[str, Any] = {
        "id": node.id,
        "content": node.content,
        "event_type": event_type or meta.get("event_type", "memory"),
        "tags": meta.get("tags", []),
        "metadata": meta,
        "status": status or "active",
        "project": project or "",
        "entity_id": entity_id or "",
        "session_id": session_id or "",
        "created_at": node.created_at,
        "last_accessed": node.last_accessed,
        "access_count": node.access_count,
        "ttl": _human_ttl(node.ttl_seconds),
    }

    if include_related:
        with db._lock:
            edge_rows = db._conn.execute(
                """SELECT source_id, target_id, edge_type, weight FROM edges
                   WHERE source_id = ? OR target_id = ?
                   ORDER BY weight DESC LIMIT 20""",
                (node.id, node.id),
            ).fetchall()
        result["edges"] = [
            {
                "direction": "out" if src == node.id else "in",
                "other_id": tgt if src == node.id else src,
                "edge_type": etype,
                "weight": weight,
            }
            for src, tgt, etype, weight in edge_rows
        ]
    return result


# ---------------------------------------------------------------------------
# Public API -- Timeline
# ---------------------------------------------------------------------------


def timeline(days: int = 7, limit_per_day: int = 10) -> str:
    """Show memory timeline grouped by day."""
    db = _get_store()
    data = db.get_timeline(days=days, limit_per_day=limit_per_day)
    if not data:
        return f"No memories in the last {days} days."
    total = sum(len(v) for v in data.values())
    output = f"Timeline ({total} memories, last {days}d)\n\n"
    for day in sorted(data.keys(), reverse=True):
        memories = data[day]
        output += f"{day} ({len(memories)})\n"
        for m in memories:
            etype = (m.metadata or {}).get("event_type", "memory")
            preview = m.content[:120].replace("\n", " ")
            output += f"- [{etype}] {preview} ({m.id[:8]} {m.created_at.strftime('%H:%M')})\n"
        output += "\n"
    return output


# ---------------------------------------------------------------------------
# Public API -- Consolidation
# ---------------------------------------------------------------------------


def _auto_backup_before_consolidate():
    """Create a backup before consolidation (rotate to keep last 3)."""
    db_path = CAIRN_HOME / "cairn.db"
    if not db_path.exists():
        return
    try:
        import sqlite3
        from cairn.crypto import secure_connect

        backups_dir = CAIRN_HOME / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backups_dir / f"pre-consolidate-{timestamp}.db"
        src = sqlite3.connect(str(db_path))
        dst = secure_connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()
        logger.info(f"Pre-consolidation backup: {backup_path}")
        # Rotate — keep only last 3
        backups = sorted(backups_dir.glob("pre-consolidate-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[3:]:
            old.unlink()
    except Exception as e:
        logger.warning(f"Auto-backup before consolidation failed: {e}")


def consolidate(prune_days: int = 14, max_summaries: int = 50) -> str:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges.

    Returns formatted markdown report.
    """
    _auto_backup_before_consolidate()
    db = _get_store()
    before = db.node_count()
    stats = db.consolidate(prune_days=prune_days, max_summaries=max_summaries)
    after = stats.get("node_count_after", before)
    removed = before - after

    output = "# Memory Consolidation Report\n\n"
    output += f"**Before:** {before} memories\n"
    output += f"**After:** {after} memories\n"
    output += f"**Removed:** {removed} total\n\n"
    output += "## Breakdown\n\n"
    output += f"- **Stale (0 access, >{prune_days}d old):** {stats.get('pruned_stale', 0)}\n"
    output += f"- **Session summaries (beyond cap of {max_summaries}):** {stats.get('pruned_summaries', 0)}\n"
    output += f"- **Orphaned edges:** {stats.get('pruned_edges', 0)}\n"
    output += f"- **Strength-decayed:** {stats.get('decayed_memories', 0)}\n"
    output += f"- **Merged entities:** {stats.get('merged_entities', 0)}\n"

    if removed == 0:
        output += "\n*Nothing to consolidate — memory store is clean.*\n"
    else:
        logger.info(f"Consolidation: removed {removed} memories ({stats})")

    return output


# ---------------------------------------------------------------------------
# Public API -- Memory Compaction
# ---------------------------------------------------------------------------


def _smart_extract(cluster) -> str:
    """Extract diverse, information-dense sentences from a cluster of memories.

    Scores sentences by: unique-word count (IDF-like), sentence length
    (diminishing returns), presence of proper nouns / code tokens, and
    cross-memory term frequency (words appearing in 2+ cluster members
    are more generalizable — ALMA-inspired strategy extraction).

    For large clusters (5+), extracts a strategy header from the most
    common bigram theme across cluster members.

    Skips near-duplicate sentences (Jaccard > 0.7).
    Orders selected sentences chronologically by source memory.
    Returns consolidated text capped at 1000 chars.
    """
    # Build cross-memory word frequency map (words appearing in 2+ members)
    from collections import Counter
    word_to_members: dict = {}  # word -> set of node indices
    for idx, node in enumerate(cluster):
        for w in set(node.content.lower().split()):
            if len(w) > 3:
                word_to_members.setdefault(w, set()).add(idx)
    cross_freq_words = {w for w, members in word_to_members.items() if len(members) >= 2}

    # Collect all sentences with source metadata
    all_sentences = []  # [(sentence, density_score, created_at)]
    seen_keys: set = set()

    for node in cluster:
        created = node.created_at.isoformat() if node.created_at else ""
        for sentence in re.split(r"(?<=[.!?])\s+", node.content):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            key = " ".join(sentence.lower().split())[:100]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            words = sentence.split()
            unique_words = len(set(w.lower() for w in words if len(w) > 3))

            # Proper nouns / capitalized words (not sentence-start)
            proper_nouns = len([w for w in words[1:] if w[0].isupper()]) if len(words) > 1 else 0

            # Code tokens: backtick spans, paths, CamelCase
            code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", sentence))

            # Diminishing returns on length
            length_score = min(len(sentence), 200) / 200.0

            # Cross-memory term frequency boost (ALMA-inspired)
            cross_freq = sum(1 for w in words if w.lower() in cross_freq_words)

            density = (unique_words * 1.0 + proper_nouns * 1.5 + code_tokens * 2.0
                       + length_score * 3.0 + cross_freq * 0.8)
            all_sentences.append((sentence, density, created))

    if not all_sentences:
        return ""

    # Sort by density (highest first)
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select top-K diverse sentences
    selected = []
    for sentence, _score, created in all_sentences:
        # Check diversity against already selected
        is_diverse = all(_jaccard(sentence.lower(), sel[0].lower(), min_word_len=3) < 0.7 for sel in selected)
        if is_diverse:
            selected.append((sentence, created))
            if len(selected) >= 8:  # Max sentences to consider
                break

    # Order chronologically by source memory created_at
    selected.sort(key=lambda x: x[1])

    # Build consolidated text (cap at 1000 chars)
    consolidated = " ".join(s for s, _ in selected)

    # Strategy header for large clusters (5+ members): extract common bigram theme
    if len(cluster) >= 5:
        bigram_counter: Counter = Counter()
        for node in cluster:
            words = [w.lower() for w in node.content.split() if len(w) > 3]
            for w1, w2 in zip(words, words[1:]):
                bigram_counter[(w1, w2)] += 1
        if bigram_counter:
            top_bigram, top_count = bigram_counter.most_common(1)[0]
            if top_count >= 3:  # Only if bigram appears in 3+ members
                theme = f"{top_bigram[0]} {top_bigram[1]}"
                consolidated = f"Strategy: {theme}. {consolidated}"

    if len(consolidated) > 1000:
        consolidated = consolidated[:997] + "..."

    return consolidated


def compact(
    event_type: str = "lesson_learned",
    similarity_threshold: float = 0.60,
    min_cluster_size: int = 3,
    dry_run: bool = False,
) -> str:
    """Compact clusters of related memories into consolidated knowledge nodes.

    Unlike deduplicate() which removes exact/near duplicates, compact() finds
    clusters of semantically related memories and creates new summary nodes
    that consolidate the knowledge, marking originals as superseded.

    Returns formatted markdown report.
    """
    db = _get_store()
    all_candidates = db.get_by_type(event_type, limit=500)
    # Filter out superseded memories — these were already compacted into a
    # consolidated node.  Re-including them causes nested "[Consolidated from]"
    # prefixes and duplicate consolidated nodes.
    candidates = [
        n for n in all_candidates
        if not (n.metadata or {}).get("superseded")
    ]

    if len(candidates) < min_cluster_size:
        return (
            f"# Memory Compaction\n\n"
            f"Only {len(candidates)} `{event_type}` memories found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to compact.\n"
        )

    # Build word sets for Jaccard clustering
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style clustering
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break  # All items assigned, no more clusters possible
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# Memory Compaction\n\n"
            f"No clusters found with >= {min_cluster_size} similar `{event_type}` memories "
            f"at {similarity_threshold:.0%} similarity. Store is already compact.\n"
        )

    # Build report and optionally perform compaction
    output = f"# Memory Compaction {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**Event type:** {event_type}\n"
    output += f"**Similarity threshold:** {similarity_threshold:.0%}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        # Sort by content length (longest first — most information)
        cluster.sort(key=lambda n: len(n.content), reverse=True)

        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        output += f"## Cluster {ci} ({len(cluster)} memories)\n\n"
        output += f"**Summary:** {consolidated[:200]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            # Strip any existing "[Consolidated from ...]" prefix from the
            # extracted content to prevent nested consolidation headers.
            consolidated = re.sub(
                r"^(\[Consolidated from \d+ memories\]\s*)+",
                "",
                consolidated,
            ).lstrip()
            # Prefix consolidated content to distinguish from originals (avoids dedup)
            compact_header = f"[Consolidated from {len(cluster)} memories] "
            compact_content = compact_header + consolidated

            # Create the consolidated node with quality metadata
            # Quality scale: 1.0 (min cluster) to 3.0 (10+ members)
            consolidation_quality = min(3.0, 1.0 + (len(cluster) - min_cluster_size) * 0.3)
            meta = {
                "event_type": event_type,
                "source": "compaction",
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
                "consolidation_quality": round(consolidation_quality, 2),
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=TTLCategory.for_event_type(event_type),
                skip_inference=True,  # Bypass embedding dedup
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded + log to forgetting audit trail
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["compacted_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, event_type,
                    "compaction_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} memories\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would compact:** {sum(len(c) for c in clusters)} memories into {len(clusters)} nodes\n"
    else:
        output += f"**Compacted:** {total_compacted} memories into {total_created} consolidated nodes\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Active Connection Discovery (Consolidation Daemon)
# ---------------------------------------------------------------------------


def discover_connections(
    lookback_hours: int = 24,
    similarity_threshold: float = 0.70,
    max_memories: int = 100,
    max_connections_per_memory: int = 3,
    dry_run: bool = False,
) -> str:
    """Actively discover and link related memories that aren't yet connected.

    Scans recent memories, finds semantically similar ones that lack edges,
    and creates 'related' edges between them. When cross-cutting patterns
    are found (clusters spanning multiple event types), generates
    advisor_insight entries.

    This is the core of the active consolidation daemon — it generates
    new knowledge from existing memories rather than just pruning.

    Args:
        lookback_hours: How far back to scan for unlinked memories.
        similarity_threshold: Minimum cosine similarity to create an edge (0.0-1.0).
        max_memories: Maximum memories to process per run.
        max_connections_per_memory: Maximum new edges per memory.
        dry_run: If True, report what would be linked without modifying.

    Returns:
        Formatted markdown report.
    """
    db = _get_store()

    if not db._vec_available:
        return "# Connection Discovery\n\nVector search unavailable — cannot discover connections.\n"

    # Phase 1: Find recent memories without many edges
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    candidates = db._conn.execute(
        """SELECT m.node_id, m.content, m.event_type, m.id, m.created_at,
                  m.entity_id, m.status
           FROM memories m
           WHERE m.created_at > ?
             AND (m.status IS NULL OR m.status = 'active')
             AND m.event_type NOT IN ('session_summary', 'coordination_snapshot',
                                       'session_respawn', 'code_chunk', 'file_summary')
           ORDER BY m.created_at DESC
           LIMIT ?""",
        (cutoff, max_memories),
    ).fetchall()

    if not candidates:
        return (
            f"# Connection Discovery\n\n"
            f"No recent active memories in the last {lookback_hours}h to analyze.\n"
        )

    # Phase 2: For each candidate, find similar memories and create edges
    edges_created = 0
    edges_skipped = 0
    cross_type_clusters = []  # Track cross-type connections for insight generation

    # Get existing edges for candidates to avoid redundant checks
    candidate_ids = {c[0] for c in candidates}
    existing_edges = set()
    if candidate_ids:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = db._conn.execute(
            f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            list(candidate_ids) + list(candidate_ids),
        ).fetchall()
        for r in rows:
            existing_edges.add((r[0], r[1]))
            existing_edges.add((r[1], r[0]))  # Bidirectional check

    report_lines = []

    for node_id, content, event_type, rowid, created_at, entity_id, status in candidates:
        # Get embedding for this memory
        try:
            emb_row = db._conn.execute(
                "SELECT embedding FROM memories_vec WHERE rowid = ?", (rowid,)
            ).fetchone()
            if not emb_row:
                continue
        except Exception:
            continue

        # Find similar memories
        import struct
        _EMBED_DIM = 384
        expected_size = _EMBED_DIM * 4  # 4 bytes per float
        if len(emb_row[0]) != expected_size:
            continue
        embedding = list(struct.unpack(f"{_EMBED_DIM}f", emb_row[0]))
        similar = db._vec_query(embedding, limit=max_connections_per_memory + 5)

        connections_made = 0
        for sim_rowid, distance in similar:
            if connections_made >= max_connections_per_memory:
                break

            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            # Look up the similar memory
            sim_row = db._conn.execute(
                "SELECT node_id, event_type, content FROM memories WHERE id = ?",
                (sim_rowid,),
            ).fetchone()
            if not sim_row or sim_row[0] == node_id:
                continue

            sim_node_id, sim_event_type, sim_content = sim_row

            # Skip if edge already exists
            if (node_id, sim_node_id) in existing_edges:
                edges_skipped += 1
                continue

            # Create the edge
            if not dry_run:
                db.add_edge(
                    source_id=node_id,
                    target_id=sim_node_id,
                    edge_type="related",
                    weight=round(similarity, 3),
                    metadata={"source": "discover_connections", "auto": True},
                )

            existing_edges.add((node_id, sim_node_id))
            existing_edges.add((sim_node_id, node_id))
            edges_created += 1
            connections_made += 1

            # Track cross-type connections for insight generation
            if event_type != sim_event_type:
                cross_type_clusters.append({
                    "source_id": node_id,
                    "source_type": event_type,
                    "source_preview": content[:80],
                    "target_id": sim_node_id,
                    "target_type": sim_event_type,
                    "target_preview": sim_content[:80],
                    "similarity": round(similarity, 3),
                })

            report_lines.append(
                f"  - `{node_id[:16]}` ({event_type}) ↔ `{sim_node_id[:16]}` "
                f"({sim_event_type}) [{similarity:.0%}]"
            )

    # Phase 3: Generate insights from cross-type patterns
    insights_generated = 0
    insight_lines = []

    if cross_type_clusters and not dry_run:
        # Group cross-type connections by type pairs
        type_pairs: Dict[tuple, list] = {}
        for conn in cross_type_clusters:
            pair = tuple(sorted([conn["source_type"], conn["target_type"]]))
            type_pairs.setdefault(pair, []).append(conn)

        # Generate insight for type pairs with 3+ connections
        for pair, connections in type_pairs.items():
            if len(connections) >= 3:
                previews = [
                    f"- {c['source_preview']}... ↔ {c['target_preview']}..."
                    for c in connections[:5]
                ]
                insight_content = (
                    f"Cross-cutting pattern: {len(connections)} connections discovered "
                    f"between {pair[0]} and {pair[1]} memories.\n"
                    f"Examples:\n" + "\n".join(previews)
                )
                try:
                    auto_capture(
                        content=insight_content,
                        event_type="advisor_insight",
                        metadata={
                            "category": "system_insight",
                            "source": "discover_connections",
                            "type_pair": list(pair),
                            "connection_count": len(connections),
                        },
                        entity_id="cairn",
                    )
                    insights_generated += 1
                    insight_lines.append(
                        f"  - {pair[0]} ↔ {pair[1]}: {len(connections)} connections"
                    )
                except Exception as e:
                    logger.debug("Failed to store cross-type insight: %s", e)

    # Format report
    mode = "(DRY RUN) " if dry_run else ""
    output = f"# Connection Discovery {mode}Report\n\n"
    output += f"**Scanned:** {len(candidates)} memories (last {lookback_hours}h)\n"
    output += f"**New edges:** {edges_created}\n"
    output += f"**Skipped (existing):** {edges_skipped}\n"
    output += f"**Cross-type insights:** {insights_generated}\n\n"

    if report_lines:
        output += "## Connections\n"
        output += "\n".join(report_lines[:30])
        if len(report_lines) > 30:
            output += f"\n  ... and {len(report_lines) - 30} more\n"
        output += "\n\n"

    if insight_lines:
        output += "## Cross-Type Patterns\n"
        output += "\n".join(insight_lines)
        output += "\n\n"

    if not report_lines and not insight_lines:
        output += "*No new connections found. Memories are already well-linked or too diverse.*\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- System Insight Synthesis
# ---------------------------------------------------------------------------


def synthesize_system_insights(
    similarity_threshold: float = 0.50,
    min_cluster_size: int = 3,
    dry_run: bool = True,
) -> str:
    """Synthesize clusters of system insights into consolidated subsystem briefs.

    Like compact() but scoped to advisor_insight memories with category=system_insight.
    Consolidated nodes inherit the system_insight category and permanent TTL.

    Args:
        similarity_threshold: Jaccard similarity threshold for clustering (lower = broader clusters).
        min_cluster_size: Minimum insights in a cluster to trigger synthesis.
        dry_run: If True, report what would be synthesized without modifying anything.

    Returns:
        Formatted markdown report.
    """
    db = _get_store()
    all_insights = db.get_by_type("advisor_insight", limit=500)

    # Filter to system_insight category
    candidates = []
    for node in all_insights:
        meta = node.metadata or {}
        if meta.get("category") == "system_insight":
            candidates.append(node)

    if len(candidates) < min_cluster_size:
        return (
            f"# System Insight Synthesis\n\n"
            f"Only {len(candidates)} system insights found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to synthesize.\n"
        )

    # Jaccard clustering (same algorithm as compact())
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# System Insight Synthesis\n\n"
            f"No clusters found with >= {min_cluster_size} similar system insights "
            f"at {similarity_threshold:.0%} similarity. Insights are already diverse.\n"
        )

    output = f"# System Insight Synthesis {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**System insights:** {len(candidates)}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        cluster.sort(key=lambda n: len(n.content), reverse=True)
        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        # Identify primary subsystem from most common tag
        primary_subsystem = max(merged_tags, key=lambda t: sum(
            1 for n in cluster if t in (n.metadata or {}).get("tags", [])
        )) if merged_tags else "general"

        output += f"## Cluster {ci}: {primary_subsystem} ({len(cluster)} insights)\n\n"
        output += f"**Summary:** {consolidated[:300]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            compact_header = f"[Subsystem brief: {primary_subsystem}] "
            compact_content = compact_header + consolidated

            meta = {
                "event_type": "advisor_insight",
                "category": "system_insight",
                "source": "insight_synthesis",
                "subsystem": primary_subsystem,
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=None,  # Permanent
                skip_inference=True,
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["synthesized_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, "advisor_insight",
                    "insight_synthesis_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} insights\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would synthesize:** {sum(len(c) for c in clusters)} insights into {len(clusters)} subsystem briefs\n"
    else:
        output += f"**Synthesized:** {total_compacted} insights into {total_created} subsystem briefs\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Forgetting Audit Trail
# ---------------------------------------------------------------------------


def forgetting_log(limit: int = 50, reason: Optional[str] = None) -> str:
    """Retrieve the forgetting audit log as formatted markdown."""
    db = _get_store()
    entries = db.get_forgetting_log(limit=limit, reason=reason)

    if not entries:
        return "# Forgetting Log\n\nNo forgetting events recorded yet.\n"

    output = "# Forgetting Log\n\n"
    if reason:
        output += f"**Filter:** reason = `{reason}`\n\n"
    output += f"**Entries:** {len(entries)}\n\n"
    output += "| Time | Reason | Type | Node | Preview |\n"
    output += "|------|--------|------|------|---------|\n"

    for entry in entries:
        deleted = entry["deleted_at"][:19] if entry.get("deleted_at") else "?"
        reason_str = entry.get("reason", "?")
        et = entry.get("event_type", "") or ""
        nid = entry.get("node_id", "")[:12]
        preview = (entry.get("content_preview") or "")[:60].replace("|", "/").replace("\n", " ")
        output += f"| {deleted} | `{reason_str}` | {et} | `{nid}` | {preview} |\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Graph Traversal
# ---------------------------------------------------------------------------


def traverse(
    memory_id: str,
    max_hops: int = 2,
    min_weight: float = 0.0,
    edge_types: Optional[List[str]] = None,
) -> str:
    """Traverse the relationship graph from a starting memory.

    Walks the `related` edges table up to max_hops, returning all
    connected memories with their hop distance and edge weight.

    Returns formatted markdown string.
    """
    db = _get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."

    results = db.get_related_chain(
        start_id=memory_id,
        max_hops=max_hops,
        min_weight=min_weight,
        edge_types=edge_types,
    )

    output = f"# Graph Traversal ({len(results)} connected memories)\n\n"
    output += f"**Start:** `{memory_id[:12]}` — {node.content[:100]}\n"
    output += f"**Max hops:** {max_hops}\n\n"

    if not results:
        output += "*No connected memories found.*\n"
        return output

    current_hop = 0
    for r in results:
        if r["hop"] != current_hop:
            current_hop = r["hop"]
            output += f"## Hop {current_hop}\n\n"

        etype = (r.get("metadata") or {}).get("event_type", "memory")
        preview = r["content"][:200]
        output += f"- **[{etype}]** `{r['node_id'][:12]}` (weight: {r['weight']:.2f}, edge: {r['edge_type']})\n"
        output += f"  {preview}\n\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Phrase Search
# ---------------------------------------------------------------------------


def phrase_search(
    phrase: str,
    limit: int = 10,
    event_type: Optional[str] = None,
    project: Optional[str] = None,
    case_sensitive: bool = False,
) -> str:
    """Search memories for exact phrase matches using FTS5.

    Returns formatted markdown string.
    """
    db = _get_store()
    try:
        results = db.phrase_search(
            phrase=phrase,
            limit=limit,
            event_type=event_type,
            case_sensitive=case_sensitive,
            project_path=project or "",
        )

        output = f"# Phrase Search Results ({len(results)})\n\n"
        output += f'**Phrase:** "{phrase}"\n'
        if event_type:
            output += f"**Event Type:** {event_type}\n"
        output += "\n"

        if results:
            for i, node in enumerate(results[:limit], 1):
                ntype = (node.metadata or {}).get("event_type", "memory")
                preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
                output += f"## {i}. [{ntype}] `{node.id}`\n"
                output += f"{preview}\n"
                tags = (node.metadata or {}).get("tags", [])
                if tags:
                    output += f"*Tags: {', '.join(str(t) for t in tags[:5])}*\n"
                output += f"*Created: {node.created_at.isoformat()[:16]}*\n\n"
        else:
            output += "*No matching memories found.*\n"

        return output

    except Exception as e:
        logger.error(f"Phrase search failed: {e}", exc_info=True)
        return f"# Phrase Search Error\n\n**Error:** {str(e)}\n"


def regex_search(
    pattern: str,
    limit: int = 10,
    event_type: Optional[str] = None,
    project: Optional[str] = None,
    case_sensitive: bool = False,
) -> str:
    """Regex search over memory content (newest first).

    Returns formatted markdown. ValueError (bad pattern) propagates so the
    handler can surface the message verbatim.
    """
    db = _get_store()
    results = db.regex_search(
        pattern=pattern,
        limit=limit,
        event_type=event_type,
        case_sensitive=case_sensitive,
        project_path=project or "",
    )

    output = f"# Regex Search Results ({len(results)})\n\n"
    output += f"**Pattern:** `{pattern}`\n"
    if event_type:
        output += f"**Event Type:** {event_type}\n"
    output += "\n"

    if results:
        for i, node in enumerate(results[:limit], 1):
            ntype = (node.metadata or {}).get("event_type", "memory")
            preview = node.content[:200] + "..." if len(node.content) > 200 else node.content
            output += f"## {i}. [{ntype}] `{node.id}`\n"
            output += f"{preview}\n"
            tags = (node.metadata or {}).get("tags", [])
            if tags:
                output += f"*Tags: {', '.join(str(t) for t in tags[:5])}*\n"
            output += f"*Created: {node.created_at.isoformat()[:16]}*\n\n"
    else:
        output += "*No matching memories found.*\n"

    return output


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
