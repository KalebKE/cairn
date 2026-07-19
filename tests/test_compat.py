"""Tests for cairn._compat — the OMEGA→Cairn back-compat shims.

Covers the env-var fallback and the fs-mutating ~/.omega → ~/.cairn data
migration (previously untested despite being data-loss-capable).
"""
import json
import sqlite3
from pathlib import Path

import pytest

import cairn._compat as compat


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point Path.home() at a tmp dir and clear the home env overrides so
    _legacy_home()/_cairn_home() resolve to tmp/.omega and tmp/.cairn."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("CAIRN_HOME", raising=False)
    monkeypatch.delenv("OMEGA_HOME", raising=False)
    return tmp_path


def _make_legacy_store(home: Path, n_memories: int = 3) -> Path:
    """Create a legacy ~/.omega with a real omega.db + markers + config."""
    omega = home / ".omega"
    omega.mkdir(parents=True)
    con = sqlite3.connect(str(omega / "omega.db"))
    con.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT)")
    con.executemany("INSERT INTO memories (content) VALUES (?)",
                    [(f"m{i}",) for i in range(n_memories)])
    con.commit()
    con.close()
    (omega / "config.json").write_text(json.dumps({"storage_path": str(omega)}))
    (omega / "last-rollup").write_text("2026-07-19T00:00:00+00:00")
    (omega / "hook.sock").write_bytes(b"")  # must NOT be copied
    (omega / "backups").mkdir()
    (omega / "backups" / "omega-2026-07-01.json").write_text("{}")
    return omega


# ---------------------------------------------------------------------------
# apply_env_fallback
# ---------------------------------------------------------------------------

def test_env_fallback_maps_unset_cairn_vars(monkeypatch):
    monkeypatch.delenv("CAIRN_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OMEGA_LLM_PROVIDER", "gemini")
    compat.apply_env_fallback()
    import os
    assert os.environ["CAIRN_LLM_PROVIDER"] == "gemini"


def test_env_fallback_does_not_override_set_cairn_var(monkeypatch):
    monkeypatch.setenv("OMEGA_CROSS_ENCODER", "0")
    monkeypatch.setenv("CAIRN_CROSS_ENCODER", "1")
    compat.apply_env_fallback()
    import os
    assert os.environ["CAIRN_CROSS_ENCODER"] == "1"  # not clobbered


def test_env_fallback_skips_home_location_vars(monkeypatch):
    # A stray legacy OMEGA_HOME must NOT redirect CAIRN_HOME (would skip migration).
    monkeypatch.delenv("CAIRN_HOME", raising=False)
    monkeypatch.setenv("OMEGA_HOME", "/legacy/omega")
    compat.apply_env_fallback()
    import os
    assert "CAIRN_HOME" not in os.environ


# ---------------------------------------------------------------------------
# needs_home_migration
# ---------------------------------------------------------------------------

def test_needs_migration_true_when_legacy_present(fake_home):
    _make_legacy_store(fake_home)
    assert compat.needs_home_migration() is True


def test_needs_migration_false_when_cairn_exists(fake_home):
    _make_legacy_store(fake_home)
    cairn = fake_home / ".cairn"
    cairn.mkdir()
    (cairn / "cairn.db").write_bytes(b"")
    assert compat.needs_home_migration() is False


def test_needs_migration_false_without_legacy(fake_home):
    assert compat.needs_home_migration() is False


def test_needs_migration_false_when_cairn_home_overridden(fake_home, monkeypatch):
    _make_legacy_store(fake_home)
    monkeypatch.setenv("CAIRN_HOME", str(fake_home / "custom"))
    assert compat.needs_home_migration() is False


# ---------------------------------------------------------------------------
# migrate_home
# ---------------------------------------------------------------------------

def test_migrate_copies_db_and_markers_renaming_to_cairn_db(fake_home):
    omega = _make_legacy_store(fake_home, n_memories=5)
    assert compat.migrate_home() is True

    cairn = fake_home / ".cairn"
    assert (cairn / "cairn.db").exists()
    # DB content preserved
    con = sqlite3.connect(str(cairn / "cairn.db"))
    assert con.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 5
    con.close()
    # markers/config/backups carried over
    assert (cairn / "config.json").exists()
    assert (cairn / "last-rollup").exists()
    assert (cairn / "backups" / "omega-2026-07-01.json").exists()
    # socket NOT copied; legacy left intact
    assert not (cairn / "hook.sock").exists()
    assert (omega / "omega.db").exists()


def test_migrate_is_idempotent_and_non_destructive(fake_home):
    _make_legacy_store(fake_home)
    assert compat.migrate_home() is True
    # Second run is a no-op and must not overwrite an existing cairn.db
    cairn_db = fake_home / ".cairn" / "cairn.db"
    cairn_db.write_text("SENTINEL-do-not-clobber")
    assert compat.migrate_home() is False
    assert cairn_db.read_text() == "SENTINEL-do-not-clobber"


def test_migrate_returns_false_without_legacy(fake_home):
    assert compat.migrate_home() is False


def test_migrate_copies_model_cache(fake_home):
    _make_legacy_store(fake_home)
    legacy_cache = fake_home / ".cache" / "omega" / "models" / "bge"
    legacy_cache.mkdir(parents=True)
    (legacy_cache / "model.onnx").write_bytes(b"weights")
    compat.migrate_home()
    assert (fake_home / ".cache" / "cairn" / "models" / "bge" / "model.onnx").exists()
