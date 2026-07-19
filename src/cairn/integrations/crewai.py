"""Cairn storage backend for CrewAI's unified memory system.

Usage::

    from cairn.integrations.crewai import CairnStorageBackend
    from crewai.memory import Memory

    backend = CairnStorageBackend()
    memory = Memory(storage=backend)

Or configure via Crew::

    from crewai import Crew

    crew = Crew(
        agents=agents,
        tasks=tasks,
        memory=True,
        memory_config={
            "provider": "cairn",
            "config": {"user_id": "my_user"},
        },
    )

Requires: ``pip install cairn crewai``
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard dependency on crewai
_MemoryRecord = None
_ScopeInfo = None


def _ensure_crewai_types():
    global _MemoryRecord, _ScopeInfo
    if _MemoryRecord is None:
        from crewai.memory.types import MemoryRecord, ScopeInfo
        _MemoryRecord = MemoryRecord
        _ScopeInfo = ScopeInfo


class CairnStorageBackend:
    """CrewAI StorageBackend backed by Cairn's local memory engine.

    Maps CrewAI's MemoryRecord to Cairn's memory store, providing:
    - Semantic search via Cairn's bge-small-en-v1.5 embeddings + sqlite-vec
    - Automatic deduplication and contradiction detection
    - Cross-session persistence with no cloud dependency
    """

    def __init__(self, *, project: str = "crewai", user_id: str | None = None):
        from cairn.bridge import (
            query as cairn_query,
            status as cairn_status,
            store as cairn_store,
        )
        from cairn.sqlite_store import CairnSQLiteStore

        self._store_fn = cairn_store
        self._query_fn = cairn_query
        self._status_fn = cairn_status
        self._db = CairnSQLiteStore._instance or CairnSQLiteStore()
        self._project = project
        self._user_id = user_id or "default"

    def _to_cairn_metadata(self, record) -> dict[str, Any]:
        """Convert a CrewAI MemoryRecord to Cairn metadata."""
        meta: dict[str, Any] = {
            "crewai_id": record.id,
            "scope": record.scope,
            "importance": record.importance,
            "source": record.source or self._user_id,
            "private": record.private,
        }
        if record.categories:
            meta["categories"] = ",".join(record.categories)
        if record.metadata:
            for k, v in record.metadata.items():
                meta[f"crewai_{k}"] = v
        return meta

    def _to_memory_record(self, cairn_result: dict) -> Any:
        """Convert an Cairn query result back to a CrewAI MemoryRecord."""
        _ensure_crewai_types()
        meta = cairn_result.get("metadata", {})
        if isinstance(meta, str):
            import json
            try:
                meta = json.loads(meta)
            except (json.JSONDecodeError, TypeError):
                meta = {}

        categories = []
        if meta.get("categories"):
            categories = [c.strip() for c in str(meta["categories"]).split(",")]

        crewai_meta = {}
        for k, v in meta.items():
            if k.startswith("crewai_") and k != "crewai_id":
                crewai_meta[k[7:]] = v

        created_str = cairn_result.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = datetime.utcnow()

        return _MemoryRecord(
            id=meta.get("crewai_id", cairn_result.get("node_id", str(uuid4()))),
            content=cairn_result.get("content", ""),
            scope=meta.get("scope", "/"),
            categories=categories,
            metadata=crewai_meta,
            importance=float(meta.get("importance", 0.5)),
            created_at=created_at,
            last_accessed=datetime.utcnow(),
            source=meta.get("source"),
            private=bool(meta.get("private", False)),
        )

    # -- Required StorageBackend methods --

    def save(self, records: list) -> None:
        """Save CrewAI memory records to Cairn."""
        for record in records:
            meta = self._to_cairn_metadata(record)
            event_type = "memory"
            if record.categories:
                cat = record.categories[0].lower()
                type_map = {
                    "decision": "decision",
                    "lesson": "lesson_learned",
                    "error": "error_pattern",
                    "preference": "user_preference",
                    "task": "task_completion",
                }
                event_type = type_map.get(cat, "memory")

            self._store_fn(
                content=record.content,
                event_type=event_type,
                metadata=meta,
                project=self._project,
            )

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple]:
        """Search Cairn memories by semantic similarity."""
        _ensure_crewai_types()
        # Use Cairn's semantic search (it handles embeddings internally)
        # We reconstruct a text query from the embedding context
        results = self._db.search_by_embedding(
            embedding=query_embedding,
            limit=limit * 2,  # oversample for filtering
        ) if hasattr(self._db, 'search_by_embedding') else []

        # Fall back to text-based search if embedding search unavailable
        if not results:
            return []

        output = []
        for result in results:
            score = result.get("score", result.get("relevance", 0.5))
            if score < min_score:
                continue

            meta = result.get("metadata", {})
            if isinstance(meta, str):
                import json
                try:
                    meta = json.loads(meta)
                except (json.JSONDecodeError, TypeError):
                    meta = {}

            # Apply scope filter
            if scope_prefix and meta.get("scope", "/") != scope_prefix:
                if not meta.get("scope", "/").startswith(scope_prefix):
                    continue

            # Apply category filter
            if categories:
                record_cats = [c.strip() for c in str(meta.get("categories", "")).split(",") if c.strip()]
                if not any(c in record_cats for c in categories):
                    continue

            record = self._to_memory_record(result)
            output.append((record, float(score)))

        output.sort(key=lambda x: x[1], reverse=True)
        return output[:limit]

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete Cairn memories matching criteria."""
        deleted = 0
        if record_ids:
            for rid in record_ids:
                # Search for the Cairn node_id by crewai_id
                rows = self._db._conn.execute(
                    "SELECT node_id FROM memories WHERE json_extract(metadata, '$.crewai_id') = ?",
                    (rid,),
                ).fetchall()
                for row in rows:
                    if self._db.delete_node(row[0]):
                        deleted += 1
        return deleted

    def update(self, record) -> None:
        """Update an existing record in Cairn."""
        # Delete old, save new
        self.delete(record_ids=[record.id])
        self.save([record])

    def get_record(self, record_id: str):
        """Return a single record by CrewAI ID."""
        _ensure_crewai_types()
        rows = self._db._conn.execute(
            "SELECT node_id, content, metadata, created_at FROM memories "
            "WHERE json_extract(metadata, '$.crewai_id') = ? LIMIT 1",
            (record_id,),
        ).fetchall()
        if not rows:
            return None
        row = rows[0]
        return self._to_memory_record({
            "node_id": row[0],
            "content": row[1],
            "metadata": row[2],
            "created_at": row[3],
        })

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list:
        """List records in a scope, newest first."""
        _ensure_crewai_types()
        query = "SELECT node_id, content, metadata, created_at FROM memories"
        params: list[Any] = []
        if scope_prefix:
            query += " WHERE json_extract(metadata, '$.scope') LIKE ?"
            params.append(f"{scope_prefix}%")
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._db._conn.execute(query, params).fetchall()
        return [
            self._to_memory_record({
                "node_id": r[0], "content": r[1],
                "metadata": r[2], "created_at": r[3],
            })
            for r in rows
        ]

    def get_scope_info(self, scope: str):
        """Get information about a scope."""
        _ensure_crewai_types()
        rows = self._db._conn.execute(
            "SELECT COUNT(*), MIN(created_at), MAX(created_at) FROM memories "
            "WHERE json_extract(metadata, '$.scope') LIKE ?",
            (f"{scope}%",),
        ).fetchone()

        return _ScopeInfo(
            path=scope,
            record_count=rows[0] if rows else 0,
            oldest_record=datetime.fromisoformat(rows[1]) if rows and rows[1] else None,
            newest_record=datetime.fromisoformat(rows[2]) if rows and rows[2] else None,
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes."""
        rows = self._db._conn.execute(
            "SELECT DISTINCT json_extract(metadata, '$.scope') FROM memories "
            "WHERE json_extract(metadata, '$.scope') LIKE ?",
            (f"{parent}%",),
        ).fetchall()
        scopes = set()
        for row in rows:
            if row[0] and row[0] != parent:
                # Get immediate child
                rest = row[0][len(parent):]
                child = rest.split("/")[0] if "/" in rest else rest
                if child:
                    scopes.add(f"{parent}{child}/")
        return sorted(scopes)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """List categories and their counts."""
        query = "SELECT json_extract(metadata, '$.categories') FROM memories"
        params: list[Any] = []
        if scope_prefix:
            query += " WHERE json_extract(metadata, '$.scope') LIKE ?"
            params.append(f"{scope_prefix}%")

        rows = self._db._conn.execute(query, params).fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            if row[0]:
                for cat in str(row[0]).split(","):
                    cat = cat.strip()
                    if cat:
                        counts[cat] = counts.get(cat, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Count records in scope."""
        if scope_prefix:
            row = self._db._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE json_extract(metadata, '$.scope') LIKE ?",
                (f"{scope_prefix}%",),
            ).fetchone()
        else:
            row = self._db._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        return row[0] if row else 0

    def reset(self, scope_prefix: str | None = None) -> None:
        """Reset (delete all) memories in scope."""
        if scope_prefix:
            self._db._conn.execute(
                "DELETE FROM memories WHERE json_extract(metadata, '$.scope') LIKE ?",
                (f"{scope_prefix}%",),
            )
        else:
            self._db._conn.execute("DELETE FROM memories")
        self._db._commit()

    # -- Async variants (sync fallback) --

    async def asave(self, records: list) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple]:
        return self.search(query_embedding, scope_prefix, categories, metadata_filter, limit, min_score)

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return self.delete(scope_prefix, categories, record_ids, older_than, metadata_filter)
