# Cairn CLI Reference

Complete reference for all `cairn` CLI commands.

## Setup & Diagnostics

### `cairn setup`

Set up Cairn: download embedding model, initialize database, and configure your editor.

```bash
cairn setup                          # auto-detect Claude Code
cairn setup --client cursor          # configure Cursor
cairn setup --client windsurf        # configure Windsurf
cairn setup --client zed             # configure Zed
cairn setup --client codex           # configure OpenAI Codex CLI
cairn setup --client antigravity     # configure Antigravity IDE
```

| Flag | Description |
|------|-------------|
| `--client` | Target editor: `claude-code`, `cursor`, `windsurf`, `zed`, `codex`, `antigravity` |
| `--download-model` | Download bge-small-en-v1.5 ONNX model (upgrade from all-MiniLM-L6-v2) |
| `--skip-model` | Skip embedding model download (text-only search, no semantic search) |
| `--hooks-only` | Configure hooks and CLAUDE.md without MCP server (saves ~600 MB RAM) |

### `cairn doctor`

Verify installation health: checks Python imports, embedding model, database, and optionally client-specific config.

```bash
cairn doctor                         # basic checks
cairn doctor --client claude-code    # include Claude Code-specific checks (MCP, hooks)
cairn doctor --fix                   # auto-fix issues by running missing setup steps
```

| Flag | Description |
|------|-------------|
| `--client` | Include client-specific checks (currently: `claude-code`) |
| `--fix` | Attempt to automatically fix detected issues |

**What each check does:**

1. **Import check** — verifies `import cairn` succeeds
2. **Model check** — verifies bge-small-en-v1.5 ONNX model is downloaded and loadable
3. **Database check** — verifies `~/.cairn/cairn.db` exists and has a valid schema
4. **MCP check** (with `--client`) — verifies MCP server configuration is registered
5. **Hook check** (with `--client`) — verifies hook entries in `~/.claude/settings.json`

### `cairn status`

Show memory count, database size, model status, and version info.

```bash
cairn status
cairn status --json
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON (also: `CAIRN_JSON=1` env var) |

### `cairn activate`

Activate a Pro license key.

```bash
cairn activate <your-license-key>
```

### `cairn license`

Show Pro license status.

```bash
cairn license
cairn license --deactivate           # remove local license
```

| Flag | Description |
|------|-------------|
| `--deactivate` | Remove the local license |

---

## Memory Operations

### `cairn store`

Store a memory with a specified type.

```bash
cairn store "We chose PostgreSQL for ACID compliance" -t decision
cairn store "Docker volume mount shadows node_modules" -t error
cairn store "Always use early returns" -t preference
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Memory type: `memory` (default), `lesson`, `decision`, `error`, `task`, `preference` |
| `--json` | Output as JSON |

### `cairn query`

Search memories by semantic similarity or exact phrase.

```bash
cairn query database choice           # semantic search
cairn query "PostgreSQL" --exact       # exact phrase (FTS5)
cairn query auth --limit 5 --json     # limit results, JSON output
```

| Flag | Description |
|------|-------------|
| `--exact` | Use FTS5 exact phrase search instead of semantic |
| `--limit` | Max results (default: 10) |
| `--json` | Output as JSON |

### `cairn remember`

Store a permanent user preference (shorthand for `cairn store -t preference`).

```bash
cairn remember "Always use TypeScript strict mode"
cairn remember "Prefer composition over inheritance"
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

---

## Analysis & Insights

### `cairn stats`

Show memory type distribution and health summary.

```bash
cairn stats
cairn stats --json
cairn stats --card                    # formatted stats card with Rich styling
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |
| `--card` | Display a formatted stats card |

### `cairn timeline`

Show memory timeline grouped by day.

```bash
cairn timeline                        # last 7 days
cairn timeline --days 30              # last 30 days
```

| Flag | Description |
|------|-------------|
| `--days` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

### `cairn activity`

Show recent session activity overview.

```bash
cairn activity                        # last 7 days
cairn activity --days 14 --json
```

| Flag | Description |
|------|-------------|
| `--days` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

---

## Maintenance

### `cairn consolidate`

Deduplicate, prune, and optimize the memory store.

```bash
cairn consolidate                     # prune entries older than 30 days with 0 access
cairn consolidate --prune-days 60     # custom prune threshold
```

| Flag | Description |
|------|-------------|
| `--prune-days` | Prune entries older than N days with 0 access (default: 30) |

### `cairn compact`

Cluster and summarize related memories of the same type.

```bash
cairn compact                         # compact lesson_learned entries
cairn compact -t decision             # compact decisions
cairn compact --threshold 0.75        # higher similarity threshold
cairn compact --dry-run               # preview without changing data
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Event type: `lesson_learned` (default), `decision`, `error_pattern`, `task_completion` |
| `--threshold` | Similarity threshold (default: 0.60) |
| `--dry-run` | Show what would be compacted without changing data |

### `cairn validate`

Validate database integrity (SQLite + FTS5 index).

```bash
cairn validate
cairn validate --repair               # attempt to rebuild FTS5 index if corrupted
```

| Flag | Description |
|------|-------------|
| `--repair` | Attempt to repair FTS5 index if corrupted |

### `cairn backup`

Back up `cairn.db` to `~/.cairn/backups/`. Keeps the last 5 backups.

```bash
cairn backup
```

---

## Export & Import

### `cairn export`

Export memories to a JSON file.

```bash
cairn export memories.json
cairn export decisions.json -t decision   # export only decisions
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Export only this type: `memory`, `decision`, `lesson_learned`, `error_pattern`, `user_preference`, `task_completion` |

### `cairn import`

Import memories from a JSON file.

```bash
cairn import memories.json
cairn import backup.json --clear          # clear existing memories before import
```

| Flag | Description |
|------|-------------|
| `--clear` | Clear existing memories before import |

### `cairn export-obsidian`

Export memories as Obsidian-compatible markdown files.

```bash
cairn export-obsidian
cairn export-obsidian --output-dir ~/vault --project myapp --limit 100
```

| Flag | Description |
|------|-------------|
| `--output-dir` | Output directory (default: `./cairn-vault`) |
| `--project` | Only export memories for this project |
| `--limit` | Max number of memories to export (default: all) |

---

## Server

### `cairn serve`

Run the MCP server (stdio or HTTP transport).

```bash
cairn serve                           # stdio (default, used by editors)
cairn serve --http --port 8787        # HTTP transport
cairn serve --no-condensed            # expose all tools individually
```

| Flag | Description |
|------|-------------|
| `--http` | Run as HTTP server (Streamable HTTP transport) |
| `--port` | HTTP port (default: 8787) |
| `--host` | Bind address (default: 127.0.0.1) |
| `--no-auth` | Disable API key authentication |
| `--no-condensed` | Disable condensed mode (expose all tools individually instead of meta-tools) |

---

## Hooks

### `cairn hooks`

Manage Claude Code hooks configuration.

```bash
cairn hooks setup                     # configure hooks in ~/.claude/settings.json
cairn hooks path                      # print the hooks directory path
cairn hooks doctor                    # check hook configuration health
```

### `cairn logs`

Show recent hook errors from `~/.cairn/hooks.log`.

```bash
cairn logs                            # last 50 lines
cairn logs -n 200                     # last 200 lines
```

| Flag | Description |
|------|-------------|
| `-n, --lines` | Number of lines to show (default: 50) |

---

## Reminders (Experimental)

### `cairn remind`

Manage time-based reminders.

```bash
cairn remind set "Review PR feedback" -d 2h
cairn remind set "Deploy to staging" -d 1d --context "After QA sign-off"
cairn remind list
cairn remind list --status all
cairn remind check --notify           # check due + send macOS notification
cairn remind dismiss <reminder_id>
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `set` | Set a new reminder. Requires `-d/--duration` (e.g., `1h`, `30m`, `2d`, `1w`) |
| `list` | List reminders. `--status`: `pending`, `fired`, `dismissed`, `all` |
| `check` | Check for due reminders. `--notify` sends macOS notification |
| `dismiss` | Dismiss a reminder by ID |

---

## Knowledge Base

### `cairn knowledge` (alias: `cairn kb`)

Manage the document knowledge base.

```bash
cairn kb scan                         # scan ~/.cairn/documents/ for new files
cairn kb scan --dir ~/papers          # scan custom directory
cairn kb list                         # list all ingested documents
cairn kb search "transformer architecture" --limit 10
cairn kb sync-kb --batch-size 20      # sync from cloud KB queue
```

**Subcommands:**

| Subcommand | Description |
|------------|-------------|
| `scan` | Scan for new/changed files. `--dir` to specify a custom directory |
| `list` | List all ingested documents |
| `search` | Search documents. `--limit` (default: 5) |
| `sync-kb` | Sync pending files from cloud KB queue. `--batch-size` (default: 10) |

---

## Cloud & Mobile

### `cairn cloud`

Cloud sync and mobile access via Supabase.

```bash
cairn cloud setup --url <url> --key <anon_key>
cairn cloud sync                      # push local data to Supabase
cairn cloud pull                      # pull memories from Supabase
cairn cloud status                    # show sync status
cairn cloud verify                    # verify Supabase connection
cairn cloud schema                    # print Supabase SQL schema
```

### `cairn mobile`

Mobile access via mcp-proxy + Tailscale.

```bash
cairn mobile setup                    # print setup instructions
cairn mobile serve --port 8089        # start HTTP proxy for mobile
```

| Flag | Description |
|------|-------------|
| `--port` | HTTP port (default: 8089) |
| `--host` | Bind address (default: 127.0.0.1) |

---

## Database Migration

### `cairn migrate-db`

Migrate legacy JSON graphs to SQLite backend.

```bash
cairn migrate-db
cairn migrate-db --force              # overwrite existing SQLite database
```

### `cairn reingest`

Reload `store.jsonl` entries into the graph system.

```bash
cairn reingest
```

---

## Global Options

Most commands support these flags:

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON (also available as `CAIRN_JSON=1` environment variable) |
