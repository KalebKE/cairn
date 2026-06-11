"""Regression tests for Core/Pro package-layout compatibility."""

import importlib


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
