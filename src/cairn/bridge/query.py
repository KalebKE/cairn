"""Query entry points for the Cairn bridge (peeled from __init__, Wave 8).

The two public retrieval front-doors (`query`, `query_structured`) plus
`semantic_search`. These orchestrate the store's hybrid-retrieval pipeline
and format results; the store singleton and the temporal-range inference
helper are late-bound through the package module so test monkeypatches on
``cairn.bridge._get_store`` resolve.
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn.sqlite_store._types import SurfacingContext  # noqa: F401 (annotation)

logger = logging.getLogger("cairn.bridge.query")


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
    if not _bridge._bridge_enable_vector_search:
        raise RuntimeError(
            "semantic_search requires enable_vector_search=True passed to "
            "initialize_bridge() before the store is created."
        )
    store = _bridge._get_store()
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
    db = _bridge._get_store()
    query_text = unicodedata.normalize("NFC", query_text)

    try:
        # Auto-infer temporal range from query text if not explicitly provided
        effective_temporal = temporal_range or _bridge._infer_temporal_range(query_text)
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
    db = _bridge._get_store()

    try:
        effective_temporal = temporal_range or _bridge._infer_temporal_range(query_text)
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
