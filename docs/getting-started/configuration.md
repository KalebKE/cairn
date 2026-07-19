---
title: Configuration
description: Storage paths, environment variables, hooks, and MCP server settings
---

# Configuration

## Storage paths

Cairn stores all data locally. Here's where everything lives:

| Path | Purpose |
|------|---------|
| `~/.cairn/cairn.db` | SQLite database (memories, coordination, entities) |
| `~/.cairn/profile.json` | User profile (greeting name, timezone, role) |
| `~/.cairn/secrets.json` | Router API keys (chmod 600, never committed) |
| `~/.cairn/hooks.log` | Hook error log (for debugging hook failures) |
| `~/.cairn/documents/` | Drop folder for auto-ingestion (knowledge base) |
| `~/.cairn/backups/` | Automatic weekly backups |
| `~/.cache/cairn/models/bge-small-en-v1.5-onnx/` | Primary ONNX embedding model |
| `~/.cache/cairn/models/all-MiniLM-L6-v2-onnx/` | Fallback embedding model |

!!! tip "Portable storage"
    Set the `CAIRN_HOME` environment variable to move the storage directory anywhere. The cache directory for models stays at `~/.cache/cairn/` regardless.

## Files modified outside `~/.cairn`

`cairn setup` modifies three files in your Claude Code configuration:

### `~/.claude.json` — MCP server registration

Adds an `cairn` entry to the `mcpServers` section:

```json
{
  "mcpServers": {
    "cairn": {
      "command": "python3",
      "args": ["-m", "cairn.server.mcp_server"],
      "env": {},
      "timeout": 3600
    }
  }
}
```

This tells Claude Code how to spawn the Cairn MCP server.

### `~/.claude/settings.json` — Hook entries

Adds 7 hook entries that power automatic memory capture, surfacing, and coordination. See the [Hooks](#hooks) section below for details.

### `~/.claude/CLAUDE.md` — Agent instructions

Adds a managed block between `<!-- Cairn:BEGIN -->` and `<!-- Cairn:END -->` markers with instructions for using memory and coordination tools. This block is updated on each `cairn setup` run.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAIRN_HOME` | `~/.cairn` | Override the storage directory for database, profile, secrets, and logs |
| `CAIRN_IDLE_TIMEOUT` | `3600` | MCP server idle timeout in seconds. Server auto-shuts down after this period of inactivity |
| `NO_COLOR` | (unset) | When set, disables Rich formatting in CLI output |

=== "Set temporarily"

    ```bash
    CAIRN_HOME=/path/to/storage cairn doctor
    ```

=== "Set permanently (zsh)"

    ```bash
    echo 'export CAIRN_HOME=/path/to/storage' >> ~/.zshrc
    source ~/.zshrc
    ```

## Hooks

Cairn uses 7 hook processes (batched from 11 handlers) that run automatically during Claude Code sessions. All hooks are **fail-open** — if a hook errors, it logs to `~/.cairn/hooks.log` and lets the operation proceed.

| # | Hook Event | Trigger | What it does |
|---|-----------|---------|--------------|
| 1 | `SessionStart` | Session opens | Delivers welcome briefing, registers coordination session, syncs git state, resumes checkpointed tasks |
| 2 | `Stop` | Session closes | Captures session summary, deregisters session, releases all file and branch claims |
| 3 | `UserPromptSubmit` | Every user message | Auto-captures decisions and lessons from conversation patterns |
| 4 | `PostToolUse` (Edit/Write) | After file edits | Surfaces relevant memories, sends coordination heartbeat, auto-claims edited file |
| 5 | `PostToolUse` (Bash/Read) | After bash/read | Surfaces relevant memories, sends coordination heartbeat |
| 6 | `PreToolUse` (Bash) | Before bash commands | Guards against git push divergence and unclaimed branch pushes |
| 7 | `PreToolUse` (Edit/Write) | Before file edits | Guards against editing files claimed by other agents, checks task assignment |

### Disabling hooks

To disable a specific hook, remove its entry from `~/.claude/settings.json`. To disable all hooks:

```bash
cairn setup --uninstall-hooks
```

To reinstall them later:

```bash
cairn setup --install-hooks
```

!!! warning "Disabling hooks reduces functionality"
    Without hooks, Cairn still works as an MCP tool server — you can manually query and store memories. But automatic capture, surfacing, coordination guards, and session management won't function.

## MCP server

Cairn runs as a **stdio MCP server** spawned by Claude Code on demand. Key characteristics:

- **Transport**: stdio (stdin/stdout JSON-RPC). No network ports, no HTTP.
- **Lifecycle**: Claude Code spawns the server process when it first needs an Cairn tool. The server stays alive for the duration of the session.
- **Idle timeout**: Auto-shuts down after 1 hour (3600s) of inactivity. Configurable via `CAIRN_IDLE_TIMEOUT`.
- **Memory**: ~31MB at startup, ~337MB after first query (loads ONNX embedding model into RAM).
- **Tool count**: Up to 70 tools depending on installed extras (24 memory + 28 coordination + 10 router + 8 entity).

### Checking server status

```bash
# Verify MCP registration
cairn doctor

# Check if the server process is running
ps aux | grep cairn.server.mcp_server
```

## Router configuration (optional)

If you installed `cairn[router]`, configure API keys in `~/.cairn/secrets.json`:

```json
{
  "anthropic_api_key": "<anthropic-api-key>",
  "openai_api_key": "<openai-api-key>",
  "google_api_key": "<google-api-key>",
  "groq_api_key": "<groq-api-key>",
  "xai_api_key": "<xai-api-key>"
}
```

!!! warning "Protect your secrets"
    `cairn setup` creates `secrets.json` with `chmod 600` (owner read/write only). Never commit this file to version control.

You can also set API keys as environment variables:

```bash
export ANTHROPIC_API_KEY="<anthropic-api-key>"
export OPENAI_API_KEY="<openai-api-key>"
export GOOGLE_API_KEY="<google-api-key>"
export GROQ_API_KEY="<groq-api-key>"
export XAI_API_KEY="<xai-api-key>"
```

## Cloud sync configuration (optional)

If you installed `cairn[cloud]`, configure Supabase credentials:

```bash
cairn cloud setup
```

This prompts for your Supabase URL and anon key, stored in `~/.cairn/secrets.json`. Sync runs automatically:

- **Pull**: Once per day at session start
- **Push**: At session end after `sync_all`

## Auto-maintenance

Cairn runs background maintenance tasks on a schedule, tracked by marker files in `~/.cairn/`:

| Task | Cadence | Marker file | What it does |
|------|---------|-------------|--------------|
| Consolidate | 7 days | `last-consolidate` | Prunes stale low-value memories, caps session summaries |
| Compact | 14 days | `last-compact` | Merges similar memories into consolidated nodes |
| Backup | 7 days | `last-backup` | Exports full database to `~/.cairn/backups/` |
| Doctor | 7 days | `last-doctor` | Runs health checks, logs warnings |
| Cloud pull | 1 day | `last-cloud-pull` | Syncs from Supabase (if configured) |
| Cloud push | per session | `last-cloud-push` | Syncs to Supabase at session end |

All maintenance runs at session start and is non-blocking.

## Next steps

- **[Quickstart](quickstart.md)** — Try Cairn hands-on with your first memories.
- **[MCP Tools Reference](../reference/mcp-tools.md)** — Full documentation for all 70 MCP tools.
