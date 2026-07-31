"""serve migrate-config must migrate every scope a cairn entry can live in.

The user-scope registration (`claude mcp add -s user cairn -- cairn serve`)
lands at the TOP LEVEL of ~/.claude.json (`mcpServers.cairn`), not under
`projects`. The migrator originally walked only `projects`, reported
"No stdio cairn entries found to migrate", and left the fleet on stdio.
"""

import json

from cairn.cli import _serve_migrate_config


def _write_config(tmp_path, config):
    (tmp_path / ".claude.json").write_text(json.dumps(config))


def _read_config(tmp_path):
    return json.loads((tmp_path / ".claude.json").read_text())


def test_migrates_user_scope_entry(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_config(
        tmp_path,
        {
            "mcpServers": {
                "cairn": {"type": "stdio", "command": "/x/bin/cairn", "args": ["serve"], "env": {}}
            }
        },
    )

    _serve_migrate_config(None)

    entry = _read_config(tmp_path)["mcpServers"]["cairn"]
    assert entry["type"] == "http"
    assert entry["url"].endswith("/mcp/")
    assert "Migrated" in capsys.readouterr().out


def test_migrates_project_scope_and_implied_stdio(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    _write_config(
        tmp_path,
        {
            # no "type" key: stdio is the implied default when command is set
            "mcpServers": {"cairn": {"command": "cairn", "args": ["serve"]}},
            "projects": {
                "/proj": {"mcpServers": {"cairn": {"type": "stdio", "command": "cairn", "args": ["serve"]}}}
            },
        },
    )

    _serve_migrate_config(None)

    config = _read_config(tmp_path)
    assert config["mcpServers"]["cairn"]["type"] == "http"
    assert config["projects"]["/proj"]["mcpServers"]["cairn"]["type"] == "http"


def test_leaves_http_entries_alone(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("HOME", str(tmp_path))
    original = {"mcpServers": {"cairn": {"type": "http", "url": "http://127.0.0.1:8377/mcp/"}}}
    _write_config(tmp_path, original)

    _serve_migrate_config(None)

    assert _read_config(tmp_path) == original
    assert "No stdio cairn entries" in capsys.readouterr().out
