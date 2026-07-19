"""Cairn SQLiteStore -- Composed class from domain-specific mixins."""

from ._base import SQLiteStoreBase
from ._store import StoreMixin
from ._query import QueryMixin
from ._search import SearchMixin
from ._maintenance import MaintenanceMixin


class SQLiteStore(
    StoreMixin,
    QueryMixin,
    SearchMixin,
    MaintenanceMixin,
    SQLiteStoreBase,
):
    """SQLite-backed memory store with sqlite-vec for vector search.

    Drop-in replacement for CairnMemory in bridge.py. All data lives on disk
    in a single SQLite database file.

    Composed from domain-specific mixins:
    - SQLiteStoreBase: DB lifecycle, schema, WAL management, configuration
    - StoreMixin: CRUD operations (store, get, delete, update, batch)
    - QueryMixin: Multi-phase query pipeline (vec, FTS, fusion, boost, rerank)
    - SearchMixin: Direct search and retrieval (text, temporal, hot cache)
    - MaintenanceMixin: Cleanup, consolidation, health, graph, entity, I/O
    """
    pass
