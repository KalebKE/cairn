---
title: Installation
description: Install Cairn and set up persistent memory for Claude Code
---

# Installation

## Install from PyPI

=== "Core (memory + coordination)"

    ```bash
    pip install cairn
    ```

    Includes all 24 memory tools and 28 coordination tools. This is everything most users need.

=== "With LLM routing"

    ```bash
    pip install cairn[router]
    ```

    Adds 10 routing tools to send prompts to the optimal model across Anthropic, OpenAI, Google, Groq, and xAI.

=== "With entity registry"

    ```bash
    pip install cairn[entity]
    ```

    Adds 8 entity tools for tracking companies, LLCs, and organizational structures. Includes encryption support.

=== "With PDF ingestion"

    ```bash
    pip install cairn[knowledge-pdf]
    ```

    Adds document ingestion with Docling for high-quality PDF extraction with native markdown output.

    For a lighter alternative using pdfplumber only:

    ```bash
    pip install cairn[knowledge-pdf-lite]
    ```

=== "With encryption"

    ```bash
    pip install cairn[encrypt]
    ```

    Adds AES-256 encrypted secure profile storage with macOS Keychain integration.

=== "With cloud sync"

    ```bash
    pip install cairn[cloud]
    ```

    Adds Supabase cloud sync for cross-device memory sharing.

=== "Everything"

    ```bash
    pip install cairn[full]
    ```

    Installs all optional modules: router, entity, knowledge-pdf, encrypt, and cloud.

## Install from source

```bash
git clone https://github.com/TracqiTechnology/cairn.git
cd cairn
pip install -e ".[dev]"
cairn setup
```

The `[dev]` extra includes test dependencies (pytest, ruff, etc.) in addition to core functionality.

## Requirements

| Requirement | Details |
|-------------|---------|
| **Python** | 3.11 or higher |
| **Disk** | ~90MB for the BGE-Small ONNX embedding model |
| **RAM** | ~31MB at startup, ~337MB after first query (CPU-only ONNX inference) |
| **OS** | macOS, Linux (Windows untested) |
| **Claude Code** | Required for MCP integration and hooks |

## Run setup

After installing, run the setup wizard:

```bash
cairn setup
```

This performs 5 steps:

1. **Creates `~/.cairn/`** — The storage directory for your database, profile, secrets, and logs.
2. **Downloads the ONNX embedding model** — Fetches `bge-small-en-v1.5` (~90MB) to `~/.cache/cairn/models/` for local semantic search. No API calls needed.
3. **Registers the MCP server** — Adds an `cairn` entry to `~/.claude.json` so Claude Code can spawn Cairn on demand via stdio transport.
4. **Installs hooks** — Adds 7 hook entries to `~/.claude/settings.json` for automatic memory capture, surfacing, coordination, and guard rails.
5. **Updates CLAUDE.md** — Adds a managed `<!-- Cairn:BEGIN -->` block to `~/.claude/CLAUDE.md` with instructions for using memory and coordination tools.

!!! tip "Setup is idempotent"
    You can run `cairn setup` multiple times safely. It will update existing configuration without duplicating entries.

## Verify the installation

```bash
cairn doctor
```

This checks:

- Python version and Cairn package version
- SQLite database exists and is accessible
- Embedding model is downloaded and loadable
- MCP server entry is registered in `~/.claude.json`
- Hooks are installed in `~/.claude/settings.json`
- CLAUDE.md contains the Cairn block

Example output:

```
Cairn Doctor — v0.6.1
─────────────────────
[OK] Python 3.12.4
[OK] Database: ~/.cairn/cairn.db (254 memories)
[OK] Embedding model: bge-small-en-v1.5-onnx
[OK] MCP server registered in ~/.claude.json
[OK] 7 hooks installed in ~/.claude/settings.json
[OK] CLAUDE.md has Cairn block

All checks passed.
```

!!! warning "If doctor reports issues"
    Run `cairn setup` again to repair missing configuration. If the embedding model download fails, check your internet connection — the model is fetched once from Hugging Face and cached locally.

## Uninstalling

To remove Cairn completely:

```bash
cairn setup --uninstall   # Removes hooks, MCP entry, and CLAUDE.md block
pip uninstall cairn
rm -rf ~/.cairn            # Delete all stored memories (irreversible)
rm -rf ~/.cache/cairn      # Delete cached embedding models
```

## Next steps

- **[Quickstart](quickstart.md)** — Store your first memory and see it come back in a new session.
- **[Configuration](configuration.md)** — Customize storage paths, hooks, and environment variables.
