"""Analytics & reporting (stats, diagnostic, digest, activity) — peeled from bridge/__init__.py (Wave 2)."""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import cairn.bridge as _bridge
from cairn import json_compat as json

logger = logging.getLogger("cairn.bridge")


# ---------------------------------------------------------------------------
# Public API -- Stats
# ---------------------------------------------------------------------------


def type_stats() -> Dict[str, int]:
    """Get memory counts grouped by event type."""
    db = _bridge._get_store()
    return db.get_type_stats()


def stats_card_data() -> Dict[str, Any]:
    """Get data for the shareable stats card display."""
    db = _bridge._get_store()
    return db.get_stats_card_data()


def session_stats() -> Dict[str, int]:
    """Get memory counts grouped by session ID."""
    db = _bridge._get_store()
    return db.get_session_stats()


def retrieval_context() -> List[Dict[str, Any]]:
    """Return recent retrieval context entries for diagnostics."""
    return _bridge._get_store().get_retrieval_context()


def access_rate_stats() -> Dict[str, Any]:
    """Get access rate breakdown: never-accessed count, by-type, top accessed."""
    db = _bridge._get_store()
    total = db.node_count()

    zero_access = db._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE access_count = 0"
    ).fetchone()[0]
    never_accessed_pct = (zero_access / total * 100) if total > 0 else 0

    # Retrieval count (semantic search hits) — separate from access_count
    zero_retrieval = db._conn.execute(
        "SELECT COUNT(*) FROM memories WHERE COALESCE(retrieval_count, 0) = 0"
    ).fetchone()[0]
    never_retrieved_pct = (zero_retrieval / total * 100) if total > 0 else 0

    # Breakdown by event_type: avg access_count + retrieval_count per type
    type_rows = db._conn.execute(
        """SELECT event_type, COUNT(*) as cnt,
                  AVG(access_count) as avg_access,
                  SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END) as zero_cnt,
                  AVG(COALESCE(retrieval_count, 0)) as avg_retrieval,
                  SUM(CASE WHEN COALESCE(retrieval_count, 0) = 0 THEN 1 ELSE 0 END) as zero_retr_cnt
           FROM memories
           GROUP BY event_type
           ORDER BY avg_access DESC"""
    ).fetchall()
    by_type = []
    for row in type_rows:
        by_type.append({
            "event_type": row[0] or "unknown",
            "count": row[1],
            "avg_access_count": round(row[2], 2),
            "zero_access_count": row[3],
            "zero_access_pct": round(row[3] / row[1] * 100, 1) if row[1] > 0 else 0,
            "avg_retrieval_count": round(row[4], 2),
            "zero_retrieval_count": row[5],
            "zero_retrieval_pct": round(row[5] / row[1] * 100, 1) if row[1] > 0 else 0,
        })

    # Top 10 most-accessed memories
    top_rows = db._conn.execute(
        """SELECT node_id, content, access_count, event_type
           FROM memories
           WHERE access_count > 0
           ORDER BY access_count DESC LIMIT 10"""
    ).fetchall()
    top_accessed = []
    for row in top_rows:
        top_accessed.append({
            "id": row[0],
            "content": row[1][:100],
            "access_count": row[2],
            "event_type": row[3] or "unknown",
        })

    # Overall average — computed from per-type aggregates (no extra query)
    _total_count = sum(row[1] for row in type_rows)
    avg_access = round(
        sum(row[2] * row[1] for row in type_rows) / _total_count, 2
    ) if _total_count else 0

    return {
        "total_memories": total,
        "zero_access_count": zero_access,
        "never_accessed_pct": round(never_accessed_pct, 1),
        "zero_retrieval_count": zero_retrieval,
        "never_retrieved_pct": round(never_retrieved_pct, 1),
        "avg_access_count": avg_access,
        "by_type": by_type,
        "top_accessed": top_accessed,
    }


# ---------------------------------------------------------------------------
# Public API -- Unified Diagnostic Report
# ---------------------------------------------------------------------------


def diagnostic_report(days: int = 30) -> Dict[str, Any]:
    """Unified Cairn health and value diagnostic.

    Aggregates data from memory store, coordination audit, session tracking,
    and LLM usage into a single report with a computed verdict.
    """
    report: Dict[str, Any] = {}

    # --- 1. Memory Health ---------------------------------------------------
    db = _bridge._get_store()
    rate_stats = access_rate_stats()

    # Velocity: memories created in last 7 days by event type
    velocity_rows = db._conn.execute(
        """SELECT event_type, COUNT(*) FROM memories
           WHERE created_at > datetime('now', '-7 days')
           GROUP BY event_type ORDER BY COUNT(*) DESC"""
    ).fetchall()
    velocity = [{"event_type": r[0], "count": r[1]} for r in velocity_rows]
    week_total = sum(r[1] for r in velocity_rows)

    # Dead memories: never accessed, older than 14 days
    dead_row = db._conn.execute(
        """SELECT COUNT(*) FROM memories
           WHERE access_count = 0 AND created_at < datetime('now', '-14 days')"""
    ).fetchone()
    dead_count = dead_row[0] if dead_row else 0
    total = rate_stats["total_memories"]
    dead_pct = (dead_count / max(total, 1)) * 100

    # Access buckets
    bucket_row = db._conn.execute(
        """SELECT
             SUM(CASE WHEN access_count = 0 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count BETWEEN 1 AND 2 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count BETWEEN 3 AND 9 THEN 1 ELSE 0 END),
             SUM(CASE WHEN access_count >= 10 THEN 1 ELSE 0 END)
           FROM memories"""
    ).fetchone()
    access_buckets = {
        "never": bucket_row[0] or 0,
        "low_1_2": bucket_row[1] or 0,
        "medium_3_9": bucket_row[2] or 0,
        "high_10_plus": bucket_row[3] or 0,
    }

    report["memory_health"] = {
        "total": total,
        "hit_rate_pct": round(100 - rate_stats["never_accessed_pct"], 1),
        "velocity_7d": velocity,
        "velocity_total_7d": week_total,
        "dead_memories": dead_count,
        "dead_pct": round(dead_pct, 1),
        "access_buckets": access_buckets,
        "avg_access_count": rate_stats["avg_access_count"],
    }

    # --- 2. Tool Usage ------------------------------------------------------
    # Tool-usage auditing was a Pro-only feature; unavailable here.
    tool_usage: Dict[str, Any] = {"top_tools": [], "cairn_tools": [], "total_calls": 0, "cairn_calls": 0}
    report["tool_usage"] = tool_usage

    # --- 3. Session Activity ------------------------------------------------
    # Session activity tracking was a Pro-only feature; unavailable here.
    sessions: Dict[str, Any] = {"total": 0, "week": 0, "month": 0}
    report["sessions"] = sessions

    # --- 4. LLM Costs -------------------------------------------------------
    llm_costs: Dict[str, Any] = {}
    try:
        from cairn.usage_tracker import UsageTracker
        tracker = UsageTracker()
        llm_costs = tracker.get_cost_estimate(days=days)
        llm_costs["by_model"] = tracker.get_usage(days=days, group_by="model")
        tracker.close()
    except Exception as e:
        logger.debug("diagnostic: usage_tracker unavailable: %s", e)
    report["llm_costs"] = llm_costs

    # --- 5. Value Assessment ------------------------------------------------
    hit_rate = report["memory_health"]["hit_rate_pct"]
    cairn_calls = tool_usage["cairn_calls"]
    total_calls = tool_usage["total_calls"]

    verdict = "idle"
    if hit_rate > 60 and cairn_calls > 50:
        verdict = "healthy"
    elif hit_rate > 40 and cairn_calls >= 5:
        verdict = "underused"

    report["value_assessment"] = {
        "memory_hit_rate": f"{hit_rate:.0f}%",
        "memory_velocity": f"{week_total} new in 7 days",
        "dead_memory_pct": f"{dead_pct:.0f}%",
        "cairn_tool_calls": cairn_calls,
        "total_tool_calls": total_calls,
        "cairn_usage_pct": f"{cairn_calls / max(total_calls, 1) * 100:.1f}%",
        "verdict": verdict,
    }

    report["period_days"] = days
    return report


# ---------------------------------------------------------------------------
# Public API -- Weekly Knowledge Digest
# ---------------------------------------------------------------------------


def get_weekly_digest(days: int = 7) -> Dict[str, Any]:
    """Generate a weekly knowledge digest with stats, trends, and highlights.

    Returns dict with: summary, type_breakdown, top_topics, growth, highlights.
    """
    db = _bridge._get_store()
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat()
    prev_cutoff = (now - timedelta(days=days * 2)).isoformat()

    total = db.node_count()

    # Delegate all period queries to the store's single-lock method
    stats = db.get_period_stats(cutoff=cutoff, prev_cutoff=prev_cutoff)
    period_count = stats["period_count"]
    type_breakdown = stats["type_breakdown"]
    session_count = stats["session_count"]
    prev_count = stats["prev_period_count"]

    # Top topics: extract most common words from recent content (simple TF)
    _STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "and", "but", "or", "nor",
        "not", "so", "yet", "both", "each", "few", "more", "most", "other",
        "some", "such", "no", "only", "own", "same", "than", "too", "very",
        "just", "because", "if", "when", "while", "how", "what", "which",
        "who", "whom", "this", "that", "these", "those", "it", "its", "my",
        "your", "his", "her", "our", "their", "all", "any", "up", "about",
        "error", "memory", "session", "plan", "decision", "captured",
    }
    top_topics: list[str] = []
    word_counts: Dict[str, int] = {}
    for content in stats["content_samples"]:
        words = re.findall(r'[a-zA-Z_]{4,}', content.lower())
        for w in words:
            if w not in _STOP_WORDS:
                word_counts[w] = word_counts.get(w, 0) + 1
    top_topics = [w for w, _ in sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:8]]

    growth_pct = ((period_count - prev_count) / max(prev_count, 1)) * 100 if prev_count > 0 else 0

    # Oldest memory recalled this week
    oldest_recalled_days = None
    try:
        oldest_recalled_days = db.get_oldest_accessed_since(cutoff)
    except Exception as e:
        logger.debug("get_weekly_digest oldest_recalled failed: %s", e)

    return {
        "period_days": days,
        "total_memories": total,
        "period_new": period_count,
        "session_count": session_count,
        "type_breakdown": type_breakdown,
        "top_topics": top_topics,
        "growth_pct": round(growth_pct, 1),
        "prev_period_count": prev_count,
        "oldest_recalled_days": oldest_recalled_days,
    }


# ---------------------------------------------------------------------------
# Public API -- Activity Summary (CLI)
# ---------------------------------------------------------------------------


def get_activity_summary(days: int = 7) -> Dict[str, Any]:
    """Gather activity data for the CLI activity command.

    Returns: {sessions: [...], tasks: [...], insights: [...], claims: [...]}
    """
    result: Dict[str, Any] = {"sessions": [], "tasks": [], "insights": [], "claims": []}

    # Recent insights from timeline
    try:
        db = _bridge._get_store()
        data = db.get_timeline(days=days, limit_per_day=10)
        if data:
            for day in sorted(data.keys(), reverse=True):
                for m in data[day]:
                    etype = (m.metadata or {}).get("event_type", "memory")
                    preview = m.content[:120].replace("\n", " ")
                    result["insights"].append(
                        {
                            "type": etype,
                            "preview": preview,
                            "created_at": m.created_at.isoformat() if m.created_at else "",
                            "id": m.id[:12] if m.id else "",
                        }
                    )
            # Limit to 15 most recent across all days
            result["insights"] = result["insights"][:15]
    except Exception as e:
        logger.warning(f"Activity summary: insights failed: {e}")

    # Coordination data (sessions, tasks, claims) was a Pro-only feature;
    # unavailable here, so these lists stay empty.

    return result


