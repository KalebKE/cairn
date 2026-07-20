"""Regression: gate/coord/export dirs must follow CAIRN_HOME at call time.

`_GATE_DIR` and `_SAFE_EXPORT_DIR` used to be module constants resolved from
`cairn_home()` at import time, so once handlers.py was imported they were frozen
to whatever CAIRN_HOME pointed at then — a stale-state bug for any process
(notably the test suite) that changes CAIRN_HOME afterwards.
"""
import pytest

from cairn.server import handlers


@pytest.mark.asyncio
async def test_gate_dir_follows_cairn_home_change(monkeypatch, tmp_path):
    a = tmp_path / "homeA"
    b = tmp_path / "homeB"
    a.mkdir()
    b.mkdir()

    monkeypatch.setenv("CAIRN_HOME", str(a))
    handlers._mark_deploy_gate_cleared("s1")
    assert (a / "gates" / "s1.gate").exists(), "gate did not land under CAIRN_HOME=A"

    # Change home; the next gate write must follow it, not the import-time home.
    monkeypatch.setenv("CAIRN_HOME", str(b))
    handlers._mark_deploy_gate_cleared("s2")
    assert (b / "gates" / "s2.gate").exists(), "gate dir did not follow CAIRN_HOME change"
    assert not (a / "gates" / "s2.gate").exists()


def test_safe_export_dir_follows_cairn_home_change(monkeypatch, tmp_path):
    b = tmp_path / "homeB"
    b.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(b))
    # The lazy resolver must reflect the current CAIRN_HOME, not the import-time one.
    assert handlers._safe_export_dir() == b
