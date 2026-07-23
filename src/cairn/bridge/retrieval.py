"""Retrieval reads for the Cairn bridge (peeled from __init__, Wave 7).

Read-only lookups over the store: similarity, single-memory hydration,
timeline, forgetting audit, graph traversal, and phrase/regex search.
Late-binds the store singleton and shared helpers through the package
module so test monkeypatches on ``cairn.bridge._get_store`` resolve.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge

logger = logging.getLogger("cairn.bridge.retrieval")


# ---------------------------------------------------------------------------
# Public API -- Similar memories
# ---------------------------------------------------------------------------


def find_similar_memories(memory_id: str, limit: int = 5) -> str:
    """Find memories similar to a given memory via vector search."""
    db = _bridge._get_store()
    node = db.get_node(memory_id)
    if node is None:
        return f"Memory `{memory_id}` not found."
    embedding = db.get_embedding(memory_id)
    if embedding is None:
        # A node can be embeddingless when a transient encode failure hit the
        # degraded-embedding discard path at store time. Regenerate from
        # content instead of failing, and heal the store so the node is
        # vector-searchable again.
        from cairn.embedding import generate_embedding, is_embedding_degraded

        embedding = generate_embedding(node.content)
        if embedding is None or is_embedding_degraded():
            return f"No embedding found for `{memory_id[:12]}`. Vector search unavailable."
        try:
            # Same content → cache-hit re-embed inside update_node's guarded path.
            db.update_node(memory_id, content=node.content)
        except Exception as e:
            logger.debug("Failed to persist regenerated embedding for %s: %s", memory_id[:12], e)
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
    db = _bridge._get_store()
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
        "ttl": _bridge._human_ttl(node.ttl_seconds),
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
    db = _bridge._get_store()
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
# Public API -- Forgetting Audit Trail
# ---------------------------------------------------------------------------


def forgetting_log(limit: int = 50, reason: Optional[str] = None) -> str:
    """Retrieve the forgetting audit log as formatted markdown."""
    db = _bridge._get_store()
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
    db = _bridge._get_store()
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
    db = _bridge._get_store()
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
    db = _bridge._get_store()
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
