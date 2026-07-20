"""Ingestion vocabulary + helpers for the Cairn bridge (peeled from __init__).

The controlled tag/blocklist/dedup vocabulary plus the leaf helpers that
normalize, tag, fact-extract, relate, supersede, and split memories during
capture. Consumed by ``core.auto_capture``. Foundation helpers and the
store singleton late-bind through the package module so test monkeypatches
on ``cairn.bridge._get_store`` resolve at call time.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn.types import AutoCaptureEventType

logger = logging.getLogger("cairn.bridge.ingest")


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
# Raised from 40 to 80 to filter infrastructure noise that inflates never-accessed count.
_MIN_CONTENT_LENGTH = 80
# Event types from hooks that generate infrastructure noise (never accessed, inflate DB).
_INFRASTRUCTURE_EVENT_TYPES = frozenset({
    "consolidate", "compact", "checkpoint", "coordination_snapshot",
    "session_respawn", "file_summary", "code_chunk",
})


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
        if _bridge._jaccard(selected[0].lower(), s2.lower(), min_word_len=3) < 0.7:
            selected.append(s2)

    observation = " ".join(selected)
    if len(observation) > 200:
        observation = observation[:197] + "..."

    return observation


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
