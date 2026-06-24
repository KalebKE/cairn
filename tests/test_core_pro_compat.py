"""Regression tests for Core/Pro package-layout compatibility."""

import ast
import importlib
from pathlib import Path

# Pro features live under the ``omega_platform.*`` namespace, never ``omega.*``.
# Core source that imports them from ``omega.<module>`` is the issue #64 bug:
# the path does not exist in either wheel, so the import silently fails and the
# feature degrades (CLI prints a bogus upgrade hint; MCP/hook paths no-op).
_PRO_ONLY_TOP_LEVEL = {"cloud", "knowledge", "embedding_daemon"}
_CORE_SRC = Path(__file__).resolve().parent.parent / "src" / "omega"


def test_bridge_submodule_aliases_support_pro_imports():
    core = importlib.import_module("omega.bridge._core")
    ingest = importlib.import_module("omega.bridge._ingest")
    query = importlib.import_module("omega.bridge._query")

    assert hasattr(core, "_get_store")
    assert hasattr(ingest, "store")
    assert hasattr(query, "query")


def test_pid_registry_imports_and_formats_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr("omega.server.pid_registry._REGISTRY_DIR", tmp_path)

    registry = importlib.import_module("omega.server.pid_registry")
    entry = registry.register_pid(transport="stdio")

    assert entry["pid"] > 0
    assert registry.list_registered_pids()
    assert "Registered OMEGA processes" in registry.format_lock_diagnostic()

    registry.unregister_pid()
    assert registry.list_registered_pids() == []


def test_core_never_imports_pro_features_from_omega_namespace():
    """Issue #64 regression: Core must reference Pro code as ``omega_platform.*``.

    Scans every Core source file for ``from omega.<pro> import ...`` or
    ``import omega.<pro>`` where ``<pro>`` is a Pro-only top-level module. Those
    paths exist in neither the public nor the Pro wheel, so any such import is a
    latent silent failure regardless of whether Pro is installed at test time.
    """
    offenders: list[str] = []
    for path in _CORE_SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                head, _, tail = node.module.partition(".")
                if head == "omega" and tail.split(".")[0] in _PRO_ONLY_TOP_LEVEL:
                    offenders.append(f"{path.relative_to(_CORE_SRC)}:{node.lineno} -> from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    head, _, tail = alias.name.partition(".")
                    if head == "omega" and tail.split(".")[0] in _PRO_ONLY_TOP_LEVEL:
                        offenders.append(f"{path.relative_to(_CORE_SRC)}:{node.lineno} -> import {alias.name}")

    assert not offenders, (
        "Core imports Pro features from the non-existent omega.* namespace "
        "(use omega_platform.*):\n  " + "\n  ".join(sorted(offenders))
    )
