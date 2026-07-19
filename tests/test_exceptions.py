"""Tests for cairn.exceptions hierarchy."""

import pytest

from cairn.exceptions import (
    CloudSyncError,
    CoordinationError,
    EmbeddingError,
    HookError,
    CairnError,
    StorageError,
    ValidationError,
)

ALL_EXCEPTIONS = [
    StorageError,
    EmbeddingError,
    CoordinationError,
    CloudSyncError,
    HookError,
    ValidationError,
]


class TestExceptionHierarchy:
    def test_cairn_error_is_exception(self):
        assert issubclass(CairnError, Exception)

    @pytest.mark.parametrize("exc_cls", ALL_EXCEPTIONS)
    def test_all_inherit_from_cairn_error(self, exc_cls):
        assert issubclass(exc_cls, CairnError)

    def test_exception_message_preserved(self):
        err = StorageError("disk full")
        assert str(err) == "disk full"
        assert isinstance(err, CairnError)
        assert isinstance(err, Exception)

    def test_exceptions_catchable_as_base(self):
        with pytest.raises(CairnError):
            raise StorageError("test")
        with pytest.raises(CairnError):
            raise EmbeddingError("test")
        with pytest.raises(CairnError):
            raise CoordinationError("test")

    def test_each_exception_is_distinct(self):
        classes = set(ALL_EXCEPTIONS + [CairnError])
        assert len(classes) == len(ALL_EXCEPTIONS) + 1
        # Each should NOT be caught by a sibling
        with pytest.raises(StorageError):
            raise StorageError("x")
        # StorageError should not match EmbeddingError
        with pytest.raises(StorageError):
            try:
                raise StorageError("x")
            except EmbeddingError:
                pytest.fail("StorageError caught as EmbeddingError")
