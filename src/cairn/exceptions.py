"""Cairn Exception Hierarchy.

Provides structured exception types to replace bare ``except Exception``
handlers with specific catches that preserve diagnostic context.
"""


class CairnError(Exception):
    """Base class for all Cairn errors."""


class StorageError(CairnError):
    """Raised when a storage operation (store, query, delete) fails."""


class EmbeddingError(CairnError):
    """Raised when embedding generation or vector search fails."""


class CoordinationError(CairnError):
    """Raised when multi-agent coordination operations fail."""


class CloudSyncError(CairnError):
    """Raised when cloud sync operations fail."""


class HookError(CairnError):
    """Raised when a hook handler encounters an error."""


class ValidationError(CairnError):
    """Raised when input validation fails (session_id, entity_id, etc.)."""
