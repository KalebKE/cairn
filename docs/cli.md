# OMEGA CLI Reference

Complete reference for all `omega` CLI commands.

## Setup & Diagnostics

### `omega setup`

Set up OMEGA: download embedding model, initialize database, and configure your editor.

```bash
omega setup                          # auto-detect Claude Code
omega setup --client cursor          # configure Cursor
omega setup --client windsurf        # configure Windsurf
omega setup --client zed             # configure Zed
omega setup --client codex           # configure OpenAI Codex CLI
omega setup --client antigravity     # configure Antigravity IDE
```

| Flag | Description |
|------|-------------|
| `--client` | Target editor: `claude-code`, `cursor`, `windsurf`, `zed`, `codex`, `antigravity` |
| `--download-model` | Download bge-small-en-v1.5 ONNX model (upgrade from all-MiniLM-L6-v2) |
| `--skip-model` | Skip embedding model download (text-only search, no semantic search) |
| `--hooks-only` | Configure hooks and CLAUDE.md without MCP server (saves ~600 MB RAM) |

### `omega doctor`

Verify installation health: checks Python imports, embedding model, database, and optionally client-specific config.

```bash
omega doctor                         # basic checks
omega doctor --client claude-code    # include Claude Code-specific checks (MCP, hooks)
omega doctor --fix                   # auto-fix issues by running missing setup steps
```

| Flag | Description |
|------|-------------|
| `--client` | Include client-specific checks (currently: `claude-code`) |
| `--fix` | Attempt to automatically fix detected issues |

**What each check does:**

1. **Import check** — verifies `import omega` succeeds
2. **Model check** — verifies bge-small-en-v1.5 ONNX model is downloaded and loadable
3. **Database check** — verifies `~/.omega/omega.db` exists and has a valid schema
4. **MCP check** (with `--client`) — verifies MCP server configuration is registered
5. **Hook check** (with `--client`) — verifies hook entries in `~/.claude/settings.json`

### `omega status`

Show memory count, database size, model status, and version info.

```bash
omega status
omega status --json
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON (also: `OMEGA_JSON=1` env var) |

### `omega activate`

Activate a Pro license key.

```bash
omega activate <your-license-key>
```

### `omega license`

Show Pro license status.

```bash
omega license
omega license --deactivate           # remove local license
```

| Flag | Description |
|------|-------------|
| `--deactivate` | Remove the local license |

---

## Memory Operations

### `omega store`

Store a memory with a specified type.

```bash
omega store "We chose PostgreSQL for ACID compliance" -t decision
omega store "Docker volume mount shadows node_modules" -t error
omega store "Always use early returns" -t preference
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Memory type: `memory` (default), `lesson`, `decision`, `error`, `task`, `preference` |
| `--json` | Output as JSON |

### `omega query`

Search memories by semantic similarity or exact phrase.

```bash
omega query database choice           # semantic search
omega query "PostgreSQL" --exact       # exact phrase (FTS5)
omega query auth --limit 5 --json     # limit results, JSON output
```

| Flag | Description |
|------|-------------|
| `--exact` | Use FTS5 exact phrase search instead of semantic |
| `--limit` | Max results (default: 10) |
| `--json` | Output as JSON |

### `omega remember`

Store a permanent user preference (shorthand for `omega store -t preference`).

```bash
omega remember "Always use TypeScript strict mode"
omega remember "Prefer composition over inheritance"
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |

---

## Analysis & Insights

### `omega stats`

Show memory type distribution and health summary.

```bash
omega stats
omega stats --json
omega stats --card                    # formatted stats card with Rich styling
```

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON |
| `--card` | Display a formatted stats card |

### `omega timeline`

Show memory timeline grouped by day.

```bash
omega timeline                        # last 7 days
omega timeline --days 30              # last 30 days
```

| Flag | Description |
|------|-------------|
| `--days` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

### `omega activity`

Show recent session activity overview.

```bash
omega activity                        # last 7 days
omega activity --days 14 --json
```

| Flag | Description |
|------|-------------|
| `--days` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

---

## Maintenance

### `omega consolidate`

Deduplicate, prune, and optimize the memory store.

```bash
omega consolidate                     # prune entries older than 30 days with 0 access
omega consolidate --prune-days 60     # custom prune threshold
```

| Flag | Description |
|------|-------------|
| `--prune-days` | Prune entries older than N days with 0 access (default: 30) |

### `omega compact`

Cluster and summarize related memories of the same type.

```bash
omega compact                         # compact lesson_learned entries
omega compact -t decision             # compact decisions
omega compact --threshold 0.75        # higher similarity threshold
omega compact --dry-run               # preview without changing data
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Event type: `lesson_learned` (default), `decision`, `error_pattern`, `task_completion` |
| `--threshold` | Similarity threshold (default: 0.60) |
| `--dry-run` | Show what would be compacted without changing data |

### `omega validate`

Validate database integrity (SQLite + FTS5 index).

```bash
omega validate
omega validate --repair               # attempt to rebuild FTS5 index if corrupted
```

| Flag | Description |
|------|-------------|
| `--repair` | Attempt to repair FTS5 index if corrupted |

### `omega backup`

Back up `omega.db` to `~/.omega/backups/`. Keeps the last 5 backups.

```bash
omega backup
```

---

## Export & Import

### `omega export`

Export memories to a JSON file.

```bash
omega export memories.json
omega export decisions.json -t decision   # export only decisions
```

| Flag | Description |
|------|-------------|
| `-t, --type` | Export only this type: `memory`, `decision`, `lesson_learned`, `error_pattern`, `user_preference`, `task_completion` |

### `omega import`

Import memories from a JSON file.

```bash
omega import memories.json
omega import backup.json --clear          # clear existing memories before import
```

| Flag | Description |
|------|-------------|
| `--clear` | Clear existing memories before import |

### `omega export-obsidian`

Export memories as Obsidian-compatible markdown files.

```bash
omega export-obsidian
omega export-obsidian --output-dir ~/vault --project myapp --limit 100
```

| Flag | Description |
|------|-------------|
| `--output-dir` | Output directory (default: `./omega-vault`) |
| `--project` | Only export memories for this project |
| `--limit` | Max number of memories to export (default: all) |

---

## Server

### `omega serve`

Run the MCP server (stdio or HTTP transport).

```bash
omega serve                           # stdio (default, used by editors)
omega serve --http --port 8787        # HTTP transport
omega serve --no-condensed            # expose all tools individually
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

### `omega hooks`

Manage Claude Code hooks configuration.

```bash
omega hooks setup                     # configure hooks in ~/.claude/settings.json
omega hooks path                      # print the hooks directory path
omega hooks doctor                    # check hook configuration health
```

### `omega logs`

Show recent hook errors from `~/.omega/hooks.log`.

```bash
omega logs                            # last 50 lines
omega logs -n 200                     # last 200 lines
```

| Flag | Description |
|------|-------------|
| `-n, --lines` | Number of lines to show (default: 50) |

---

## Reminders (Experimental)

### `omega remind`

Manage time-based reminders.

```bash
omega remind set "Review PR feedback" -d 2h
omega remind set "Deploy to staging" -d 1d --context "After QA sign-off"
omega remind list
omega remind list --status all
omega remind check --notify           # check due + send macOS notification
omega remind dismiss <reminder_id>
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

### `omega knowledge` (alias: `omega kb`)

Manage the document knowledge base.

```bash
omega kb scan                         # scan ~/.omega/documents/ for new files
omega kb scan --dir ~/papers          # scan custom directory
omega kb list                         # list all ingested documents
omega kb search "transformer architecture" --limit 10
omega kb sync-kb --batch-size 20      # sync from cloud KB queue
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

### `omega cloud`

Cloud sync and mobile access via Supabase.

```bash
omega cloud setup --url <url> --key <anon_key>
omega cloud sync                      # push local data to Supabase
omega cloud pull                      # pull memories from Supabase
omega cloud status                    # show sync status
omega cloud verify                    # verify Supabase connection
omega cloud schema                    # print Supabase SQL schema
```

### `omega mobile`

Mobile access via mcp-proxy + Tailscale.

```bash
omega mobile setup                    # print setup instructions
omega mobile serve --port 8089        # start HTTP proxy for mobile
```

| Flag | Description |
|------|-------------|
| `--port` | HTTP port (default: 8089) |
| `--host` | Bind address (default: 127.0.0.1) |

---

## Database Migration

### `omega migrate-db`

Migrate legacy JSON graphs to SQLite backend.

```bash
omega migrate-db
omega migrate-db --force              # overwrite existing SQLite database
```

### `omega reingest`

Reload `store.jsonl` entries into the graph system.

```bash
omega reingest
```

---

## Global Options

Most commands support these flags:

| Flag | Description |
|------|-------------|
| `--json` | Output as JSON (also available as `OMEGA_JSON=1` environment variable) |
