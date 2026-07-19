"""Back-compat shims for the OMEGA -> Cairn rename.

Two concerns, kept separate:

* env fallback (`apply_env_fallback`) — pure, side-effect-free beyond
  os.environ; safe to run on import so a user's existing ``OMEGA_*`` shell/CI
  configuration keeps working after the rename.
* data-dir migration (`migrate_home`) — filesystem side effects; only ever run
  explicitly (``cairn migrate-home``) or from the server/CLI startup, never on
  bare import, so tests and library use are unaffected.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

_LEGACY = "OMEGA_"
_NEW = "CAIRN_"


# Home-location vars are handled by migrate_home()/default resolution, not the
# generic fallback: remapping a stray legacy OMEGA_HOME would redirect Cairn
# back to the old dir and silently skip migration.
_ENV_FALLBACK_SKIP = {"OMEGA_HOME", "OMEGA_DIR"}


def apply_env_fallback() -> None:
    """Copy any set ``OMEGA_*`` env var into the matching ``CAIRN_*`` when the
    latter is unset. Lets pre-rename shell/CI configuration keep working."""
    for key, val in list(os.environ.items()):
        if key.startswith(_LEGACY) and key not in _ENV_FALLBACK_SKIP:
            os.environ.setdefault(_NEW + key[len(_LEGACY):], val)


def _legacy_home() -> Path:
    return Path(os.environ.get("OMEGA_HOME", str(Path.home() / ".omega")))


def _cairn_home() -> Path:
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))


def needs_home_migration() -> bool:
    """True when a legacy ~/.omega store exists and no Cairn store has been
    created yet (and the home is not overridden to a custom/test path)."""
    if os.environ.get("CAIRN_HOME"):
        return False
    legacy, cairn = _legacy_home(), _cairn_home()
    return (legacy / "omega.db").exists() and not (cairn / "cairn.db").exists()


def migrate_home(verbose: bool = False) -> bool:
    """Idempotent one-time migration of the legacy ~/.omega data dir to
    ~/.cairn: omega.db -> cairn.db, plus markers, config, backups, logs, and
    the ONNX model cache. Non-destructive — the legacy dir is left in place as
    a backup, and an existing cairn.db is never overwritten.

    Returns True if a migration was performed.
    """
    legacy, cairn = _legacy_home(), _cairn_home()
    if not (legacy / "omega.db").exists() or (cairn / "cairn.db").exists():
        return False

    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    cairn.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Fold any WAL into the main DB file before copying, then copy as cairn.db.
    try:
        import sqlite3
        con = sqlite3.connect(str(legacy / "omega.db"))
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
    except Exception:
        pass
    shutil.copy2(legacy / "omega.db", cairn / "cairn.db")
    _log(f"  migrated omega.db -> {cairn / 'cairn.db'}")

    _SKIP = {"omega.db", "omega.db-wal", "omega.db-shm", "hook.sock"}
    for item in legacy.iterdir():
        if item.name in _SKIP:
            continue
        dst = cairn / item.name
        if dst.exists():
            continue
        try:
            if item.is_dir():
                shutil.copytree(item, dst)
            else:
                shutil.copy2(item, dst)
        except Exception:
            continue

    # ONNX model cache (regenerable, but copying avoids a slow re-download).
    legacy_cache = Path.home() / ".cache" / "omega"
    cairn_cache = Path.home() / ".cache" / "cairn"
    if legacy_cache.exists():
        cairn_cache.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copytree(legacy_cache, cairn_cache, dirs_exist_ok=True)
            _log(f"  migrated model cache -> {cairn_cache}")
        except Exception:
            pass

    _log(f"  legacy dir left intact at {legacy} (safe to delete once verified)")
    return True
