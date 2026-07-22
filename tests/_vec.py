"""Shared skip-guard for tests that require the sqlite-vec loadable extension.

Some Python builds (notably setup-python's macOS CPython) compile SQLite
without loadable-extension support: Connection has no enable_load_extension,
sqlite-vec cannot load, and the store runs in brute-force fallback mode.
That is a supported runtime configuration, but tests asserting vec-backed
behavior (find_similar, stored embeddings, embedding-driven supersession)
cannot pass there — they skip with this marker instead.
"""
import sqlite3

import pytest


def _vec_ok() -> bool:
    try:
        c = sqlite3.connect(":memory:")
        c.enable_load_extension  # AttributeError on unsupported builds
        import sqlite_vec
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        return True
    except Exception:
        return False


VEC_AVAILABLE = _vec_ok()
requires_vec = pytest.mark.skipif(
    not VEC_AVAILABLE,
    reason="sqlite-vec loadable extension unavailable (vec-less brute-force mode)",
)
