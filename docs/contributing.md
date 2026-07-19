# Contributing to Cairn

Thanks for your interest in contributing to Cairn!

## Development Setup

```bash
git clone https://github.com/TracqiTechnology/cairn.git
cd cairn
pip install -e ".[dev]"
cairn setup
```

## Running Tests

```bash
pytest tests/                          # All tests
pytest tests/ --cov=cairn              # With coverage
pytest tests/test_bridge.py -v         # Single file
ruff check src/ tests/                 # Lint
```

## Code Style

- **Linter**: ruff (config in `pyproject.toml`)
- **Line length**: 120 characters
- **Python**: 3.11+
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `test:`, `chore:`)

## Pull Request Process

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Ensure `pytest` and `ruff check` pass
4. Submit a PR with a clear description of the change

## Architecture

- `src/cairn/bridge.py` — Public API (start here for new features)
- `src/cairn/sqlite_store.py` — Storage layer (SQLite + sqlite-vec + FTS5)
- `src/cairn/server/handlers.py` — MCP tool handlers
- `src/cairn/server/hook_server.py` — Daemon hook handlers
- `src/cairn/coordination.py` — Multi-agent coordination
- `src/cairn/server/coord_handlers.py` — Coordination MCP handlers
- `tests/` — 1538+ tests across 39 files

## Reporting Issues

Use [GitHub Issues](https://github.com/TracqiTechnology/cairn/issues). For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
