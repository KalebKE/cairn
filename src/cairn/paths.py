"""Centralized filesystem path resolution for Cairn.

Single source of truth for the data home, database file, and model cache. Every
module should resolve paths through here so ``CAIRN_HOME`` is honored uniformly
— previously ``cli.py``/``telemetry.py`` and several server modules hardcoded
``~/.cairn`` and silently ignored the env var, so running with a custom
``CAIRN_HOME`` read/wrote the wrong directory.

These are functions (not module constants) so the env is read fresh on each
call — important for tests, which set ``CAIRN_HOME`` per-case.
"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_HOME = ".cairn"
_DEFAULT_DB = "cairn.db"


def cairn_home() -> Path:
    """The Cairn data directory (``CAIRN_HOME`` env, default ``~/.cairn``)."""
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / _DEFAULT_HOME)))


def db_path() -> Path:
    """The primary SQLite database file."""
    return cairn_home() / _DEFAULT_DB


def hooks_log() -> Path:
    return cairn_home() / "hooks.log"


def cache_dir() -> Path:
    """ONNX model cache (``CAIRN_CACHE`` env, default ``~/.cache/cairn``)."""
    return Path(os.environ.get("CAIRN_CACHE", str(Path.home() / ".cache" / "cairn")))
