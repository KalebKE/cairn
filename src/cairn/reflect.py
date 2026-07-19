"""
Cairn Reflection — query-time memory audit (read-only, no side effects).

Invoked by the cairn_reflect MCP tool. Operates on already-stored memories,
NOT on incoming content. This is the only contradiction path that runs at
query time (the other two run during storage).

Three analysis functions:
- find_contradictions: O(n^2) pairwise audit using contradictions.detect_contradictions()
  as its engine, at lower thresholds (similarity 0.2, confidence 0.3)
- trace_evolution: trace how understanding of a topic changed over time
- find_stale: surface old, never-accessed memories for human triage

All functions take an SQLiteStore instance and return plain dicts
suitable for MCP response formatting.

See also:
- contradictions.py — the pure heuristic engine this module calls
- conflicts.py — pre-storage conflict gate with auto-resolve (Pipeline Phase 2.5)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cairn.reflect")

__all__ = [
    "find_contradictions",
    "trace_evolution",
    "find_stale",
]

# Protected types that should never be flagged as stale
_PROTECTED_TYPES = frozenset(
    {
        "user_preference",
        "constraint",
        "behavioral_pattern",
        "reminder",
    }
)

# Edge types used for evolution tracking.
# _auto_relate stores "evolution"; the tool schema enum says "evolves".
# We check both to be safe.
_EVOLUTION_EDGE_TYPES = ["evolution", "evolves", "supersedes"]


# ---------------------------------------------------------------------------
# 1. Contradiction audit
# ---------------------------------------------------------------------------


def find_contradictions(
    store,
    topic: str,
    limit: int = 20,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Pairwise contradiction audit for existing memories on a topic.

    Unlike the pre-store check_contradictions (which compares *new* content
    against candidates), this audits *existing* memories against each other.

    Args:
        store: SQLiteStore instance.
        topic: Topic to search for relevant memories.
        limit: Max memories to retrieve and compare (caps O(n^2)).
        entity_id: Scope to a specific entity, or None for all.

    Returns:
        Dict with topic, memories_analyzed count, and contradictions list.
    """
    from cairn.contradictions import detect_contradictions

    # Retrieve relevant memories
    results = store.query(
        topic,
        limit=min(limit, 50),
        entity_id=entity_id,
        use_cache=False,
    )

    if len(results) < 2:
        return {
            "topic": topic,
            "memories_analyzed": len(results),
            "contradictions": [],
        }

    # Pairwise comparison: for each memory, check it against all others
    contradictions = []
    seen_pairs = set()

    for i, mem_a in enumerate(results):
        # Build candidate list (all other memories)
        candidates = []
        candidate_ids = []
        for j, mem_b in enumerate(results):
            if i == j:
                continue
            pair_key = tuple(sorted((mem_a.id, mem_b.id)))
            if pair_key in seen_pairs:
                continue
            candidates.append(mem_b.content)
            candidate_ids.append((j, mem_b.id))

        if not candidates:
            continue

        # Run contradiction detection (uses cheap string heuristics)
        detected = detect_contradictions(
            mem_a.content,
            candidates,
            similarity_threshold=0.2,  # Lower gate for existing memories
            contradiction_threshold=0.3,
        )

        for cr in detected:
            idx, mem_b_id = candidate_ids[cr.candidate_index]
            pair_key = tuple(sorted((mem_a.id, mem_b_id)))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            contradictions.append(
                {
                    "memory_a_id": mem_a.id,
                    "memory_a_content": mem_a.content[:200],
                    "memory_b_id": mem_b_id,
                    "memory_b_content": cr.candidate_content[:200],
                    "confidence": cr.confidence,
                    "signals": cr.signals,
                    "reason": cr.reason,
                }
            )

    # Sort by confidence descending
    contradictions.sort(key=lambda c: c["confidence"], reverse=True)

    return {
        "topic": topic,
        "memories_analyzed": len(results),
        "contradictions": contradictions,
    }


# ---------------------------------------------------------------------------
# 2. Evolution tracing
# ---------------------------------------------------------------------------


def trace_evolution(
    store,
    topic: str,
    limit: int = 20,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Trace how understanding of a topic changed over time.

    Uses knowledge graph edges (evolution/supersedes) to build connected
    chains showing how memories evolved. Falls back to chronological
    ordering when no edges exist.

    Args:
        store: SQLiteStore instance.
        topic: Topic to trace evolution for.
        limit: Max seed memories to retrieve.
        entity_id: Scope to a specific entity, or None for all.

    Returns:
        Dict with topic, total_memories count, and chains list.
    """
    # Get seed memories
    results = store.query(
        topic,
        limit=min(limit, 50),
        entity_id=entity_id,
        use_cache=False,
    )

    if not results:
        return {
            "topic": topic,
            "total_memories": 0,
            "chains": [],
        }

    # For each seed, traverse evolution/supersedes edges
    all_chain_nodes: Dict[str, Dict[str, Any]] = {}  # node_id -> info
    edges_found: List[Dict[str, Any]] = []  # edge records for chain building

    seed_ids = {r.id for r in results}

    # Index seed memories
    for r in results:
        all_chain_nodes[r.id] = {
            "node_id": r.id,
            "content": r.content[:200],
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "event_type": r.metadata.get("event_type", ""),
        }

    # Traverse edges from each seed
    for r in results:
        related = store.get_related_chain(
            r.id,
            max_hops=3,
            edge_types=_EVOLUTION_EDGE_TYPES,
        )
        for rel in related:
            nid = rel["node_id"]
            if nid not in all_chain_nodes:
                all_chain_nodes[nid] = {
                    "node_id": nid,
                    "content": rel["content"][:200],
                    "created_at": rel.get("created_at", ""),
                    "event_type": (rel.get("metadata") or {}).get("event_type", ""),
                }
            edges_found.append(
                {
                    "from": r.id,
                    "to": nid,
                    "edge_type": rel["edge_type"],
                    "weight": rel["weight"],
                }
            )

    # Build connected chains using union-find
    parent: Dict[str, str] = {nid: nid for nid in all_chain_nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for edge in edges_found:
        if edge["from"] in parent and edge["to"] in parent:
            union(edge["from"], edge["to"])

    # Also union seed memories that share no edges but are in the same topic
    # (they form a single "ungrouped" chain)
    # -- skip this: let ungrouped seeds be singleton chains

    # Group by root
    groups: Dict[str, List[str]] = {}
    for nid in all_chain_nodes:
        root = find(nid)
        groups.setdefault(root, []).append(nid)

    # Build chains sorted chronologically within each group
    chains = []
    for group_ids in groups.values():
        if len(group_ids) < 2:
            continue  # Skip singletons (no evolution to show)

        chain_memories = []
        for nid in group_ids:
            node = all_chain_nodes[nid]
            chain_memories.append(node)

        # Sort by created_at chronologically
        chain_memories.sort(key=lambda m: m["created_at"] or "")

        # Collect edges within this chain
        chain_node_ids = set(group_ids)
        chain_edges = [e for e in edges_found if e["from"] in chain_node_ids and e["to"] in chain_node_ids]

        chains.append(
            {
                "memories": chain_memories,
                "edges": chain_edges,
                "length": len(chain_memories),
            }
        )

    # Sort chains by length descending (most interesting first)
    chains.sort(key=lambda c: c["length"], reverse=True)

    return {
        "topic": topic,
        "total_memories": len(all_chain_nodes),
        "chains": chains,
    }


# ---------------------------------------------------------------------------
# 3. Stale memory detection
# ---------------------------------------------------------------------------


def find_stale(
    store,
    days: int = 30,
    min_age_days: int = 14,
    limit: int = 30,
    entity_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Surface old, never-accessed memories for human triage.

    Unlike consolidate (which silently prunes), this returns candidates
    for the user to review and decide what to keep or remove.

    Args:
        store: SQLiteStore instance.
        days: Look-back window in days (how far back to search).
        min_age_days: Minimum age in days to be considered stale.
        limit: Max memories to return.
        entity_id: Scope to a specific entity, or None for all.

    Returns:
        Dict with total_candidates count and stale_memories list.
    """
    now = datetime.now(timezone.utc)

    # Build SQL query for stale candidates
    # Criteria: older than min_age_days, access_count = 0, not protected types
    placeholders = ",".join("?" for _ in _PROTECTED_TYPES)
    params: list = [min_age_days, min_age_days + days]

    sql = f"""
        SELECT node_id, content, metadata, created_at,
               access_count, last_accessed, ttl_seconds
        FROM memories
        WHERE julianday('now') - julianday(created_at) >= ?
          AND julianday('now') - julianday(created_at) <= ?
          AND access_count = 0
          AND json_extract(metadata, '$.event_type') NOT IN ({placeholders})
    """
    params.extend(list(_PROTECTED_TYPES))

    if entity_id:
        sql += " AND json_extract(metadata, '$.entity_id') = ?"
        params.append(entity_id)

    sql += " ORDER BY created_at ASC"

    try:
        rows = store._conn.execute(sql, params).fetchall()
    except Exception as e:
        logger.error("find_stale SQL failed: %s", e)
        return {"total_candidates": 0, "stale_memories": []}

    # Score and format results
    stale_memories = []
    for row in rows:
        result = store._row_to_result(row)
        event_type = result.metadata.get("event_type", "memory")

        # Skip protected types (belt and suspenders with SQL filter)
        if event_type in _PROTECTED_TYPES:
            continue

        # Staleness score (0-1)
        age_days = (now - result.created_at).total_seconds() / 86400 if result.created_at else 0
        score = 0.0
        reasons = []

        # Age factor (0.4 weight): older = more stale
        age_factor = min(1.0, age_days / 90)  # Maxes at 90 days
        score += age_factor * 0.4
        if age_days > 30:
            reasons.append(f"old ({int(age_days)}d)")

        # Access factor (0.3 weight): zero access = full score
        score += 0.3  # Always 0 access (SQL filter guarantees this)
        reasons.append("never accessed")

        # Priority factor (0.2 weight): low priority = more stale
        priority = result.metadata.get("priority", 3)
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            priority = 3
        priority_factor = max(0.0, 1.0 - (priority / 5))
        score += priority_factor * 0.2
        if priority <= 2:
            reasons.append(f"low priority ({priority})")

        # Superseded factor (0.1 weight): check if superseded by another memory
        # Quick check via edges table
        try:
            superseded_count = store._conn.execute(
                "SELECT COUNT(*) FROM edges WHERE target_id = ? AND edge_type = 'supersedes'",
                (result.id,),
            ).fetchone()[0]
            if superseded_count > 0:
                score += 0.1
                reasons.append("superseded")
        except Exception as e:
            logger.debug("Superseded edge check failed for %s: %s", result.id, e)

        stale_memories.append(
            {
                "id": result.id,
                "content_preview": result.content[:150],
                "created_at": result.created_at.isoformat() if result.created_at else "",
                "access_count": result.access_count,
                "staleness_score": round(score, 3),
                "event_type": event_type,
                "reasons": reasons,
            }
        )

    # Sort by staleness score descending
    stale_memories.sort(key=lambda m: m["staleness_score"], reverse=True)
    stale_memories = stale_memories[:limit]

    return {
        "total_candidates": len(rows),
        "stale_memories": stale_memories,
    }
