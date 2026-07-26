# cairn

> Persistent memory for AI coding agents. Local-first, 25 MCP tools, zero cloud dependencies.

## OBJECTIVE

Install Cairn as an MCP server so the AI coding agent has persistent memory across sessions.

## TODO

- [ ] Install the cairn Python package
- [ ] Run cairn setup to configure MCP server and hooks
- [ ] Verify installation with cairn doctor

## Step 1: Install

Requires Python 3.11+.

```bash
pip install cairn-memory[server]
```

If `pip install` fails with a Python version error, try `pip3.11 install cairn-memory[server]` or `python3.11 -m pip install cairn-memory[server]`.

## Step 2: Setup

```bash
cairn setup
```

This will:
1. Create `~/.cairn/` directory
2. Download the ONNX embedding model (~90 MB)
3. Register cairn as an MCP server with the detected client (Claude Code, Cursor, Windsurf, Cline)
4. Install session hooks for auto-capture

For a specific client, pass `--client`:

```bash
cairn setup --client claude-code
cairn setup --client cursor
cairn setup --client windsurf
cairn setup --client cline
cairn setup --client codex
```

## Step 3: Verify

```bash
cairn doctor
```

Expected output:
```
Cairn Doctor
  Python: 3.11.x
  Package: cairn 1.3.0
  Database: ~/.cairn/cairn.db (OK)
  Embedding model: bge-small-en-v1.5-onnx (OK)
  MCP server: registered (OK)
  Hooks: installed (OK)
All checks passed.
```

## DONE WHEN

- `cairn doctor` reports all checks passed
- The MCP server is registered and the agent can call cairn_store and cairn_query tools

## EXECUTE NOW
