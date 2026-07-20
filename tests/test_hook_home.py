"""Hooks must honor CAIRN_HOME, and the deploy guard must actually be clearable.

Two regressions:
  * hook scripts hardcoded ``~/.cairn`` for gate markers / hooks.log, so under a
    custom CAIRN_HOME the MCP server (writer) and pre_deploy_guard (reader)
    looked in different directories — the gate read as never-cleared and every
    deploy was blocked.
  * pre_deploy_guard._is_gate_cleared required ``coord`` + ``action_claim``
    markers from the removed Pro coordination module (no writers), so the gate
    could never clear even with matching paths.
"""
import json
import time

import pytest

from cairn.hooks import pre_deploy_guard
from cairn.server import handlers


def test_gate_marker_written_by_server_is_read_by_guard(monkeypatch, tmp_path):
    """Cross-process contract: writer (handlers) and reader (guard) agree on dir."""
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(home))

    handlers._mark_deploy_gate_cleared("sess-x")  # writer → $CAIRN_HOME/gates
    assert (home / "gates" / "sess-x.gate").exists(), "writer did not use CAIRN_HOME"
    assert pre_deploy_guard._is_marker_fresh("sess-x", "gate"), "reader did not follow CAIRN_HOME"


def test_gate_cleared_requires_only_decision_marker(monkeypatch, tmp_path):
    """Only the decision-query gate marker is required (coord/action are dead)."""
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(home))
    gates = home / "gates"
    gates.mkdir(parents=True)
    (gates / "sess-y.gate").write_text(str(time.time()))

    assert pre_deploy_guard._is_gate_cleared("sess-y") is True


def test_hooks_log_honors_cairn_home(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(home))

    pre_deploy_guard._log_timing("pre_deploy_guard", 1.0)
    assert (home / "hooks.log").exists(), "hooks.log did not follow CAIRN_HOME"


def test_deploy_blocked_without_gate(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(home))
    monkeypatch.setenv("TOOL_NAME", "Bash")
    monkeypatch.setenv("TOOL_INPUT", json.dumps({"command": "vercel --prod"}))
    monkeypatch.setenv("SESSION_ID", "sess-block")

    with pytest.raises(SystemExit) as ei:
        pre_deploy_guard.main()
    assert ei.value.code == 2


def test_deploy_allowed_after_decision_gate(monkeypatch, tmp_path, capsys):
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("CAIRN_HOME", str(home))
    gates = home / "gates"
    gates.mkdir(parents=True)
    (gates / "sess-ok.gate").write_text(str(time.time()))

    monkeypatch.setenv("TOOL_NAME", "Bash")
    monkeypatch.setenv("TOOL_INPUT", json.dumps({"command": "vercel --prod"}))
    monkeypatch.setenv("SESSION_ID", "sess-ok")

    # No SystemExit == allowed.
    pre_deploy_guard.main()
    assert "Gate cleared" in capsys.readouterr().out
