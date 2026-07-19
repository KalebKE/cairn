# CLI Commands

All commands are invoked as `cairn <command>`. The CLI is implemented in `src/cairn/cli.py`.

---

## Core Commands

### `cairn setup`

Set up Cairn: create directories, download embedding model, initialize database, register MCP server, install hooks, update CLAUDE.md.

```
cairn setup [--download-model] [--client {claude-code}]
```

| Option | Description |
|--------|-------------|
| `--download-model` | Download bge-small-en-v1.5 ONNX model (upgrade from all-MiniLM-L6-v2) |
| `--client {claude-code}` | Configure a specific client (MCP registration, hooks) |

### `cairn doctor`

Verify installation health: imports, embedding model, database, MCP registration, hooks.

```
cairn doctor [--client {claude-code}]
```

| Option | Description |
|--------|-------------|
| `--client {claude-code}` | Include client-specific checks (MCP registration, hooks) |

### `cairn status`

Show memory count, database size, model status, edge count.

```
cairn status
```

### `cairn serve`

Run the MCP server in stdio mode. Used by Claude Code internally -- not normally called directly.

```
cairn serve
```

---

## Memory Commands

### `cairn query`

Search memories by semantic similarity or exact phrase match.

```
cairn query <text> [--exact] [--limit N] [--json]
```

| Option | Description |
|--------|-------------|
| `<text>` | Search text (positional, one or more words) |
| `--exact` | Use FTS5 exact phrase search instead of semantic |
| `--limit N` | Max results (default: 10) |
| `--json` | Output as JSON |

Example:

```
cairn query "database migration pattern" --limit 5
cairn query "threading deadlock" --exact
```

### `cairn store`

Store a memory with a specified type.

```
cairn store <content> [-t TYPE]
```

| Option | Description |
|--------|-------------|
| `<content>` | Memory content (positional, one or more words) |
| `-t`, `--type` | Memory type: `memory` (default), `lesson`, `decision`, `error`, `task`, `preference` |

Example:

```
cairn store "Always use absolute paths in hooks" -t lesson
cairn store "Switched from PyPDF2 to Docling for PDF extraction" -t decision
```

### `cairn remember`

Store a permanent user preference.

```
cairn remember <text>
```

Example:

```
cairn remember "I prefer tabs over spaces"
```

### `cairn timeline`

Show memory timeline grouped by day.

```
cairn timeline [--days N] [--json]
```

| Option | Description |
|--------|-------------|
| `--days N` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

---

## Maintenance Commands

### `cairn consolidate`

Deduplicate, prune stale memories, cap session summaries, clean orphaned edges.

```
cairn consolidate [--prune-days N]
```

| Option | Description |
|--------|-------------|
| `--prune-days N` | Prune entries older than N days with zero access (default: 30) |

### `cairn compact`

Cluster and summarize related memories to reduce noise.

```
cairn compact [-t TYPE] [--threshold FLOAT] [--dry-run]
```

| Option | Description |
|--------|-------------|
| `-t`, `--type` | Event type to compact: `lesson_learned` (default), `decision`, `error_pattern`, `task_completion` |
| `--threshold` | Similarity threshold for clustering (default: 0.60) |
| `--dry-run` | Preview clusters without compacting |

### `cairn backup`

Back up cairn.db to ~/.cairn/backups/ (keeps last 5).

```
cairn backup
```

### `cairn validate`

Validate cairn.db integrity (SQLite + FTS5 + vec index).

```
cairn validate [--repair]
```

| Option | Description |
|--------|-------------|
| `--repair` | Attempt to repair FTS5 index if corrupted |

### `cairn stats`

Show memory type distribution and health summary.

```
cairn stats [--json]
```

| Option | Description |
|--------|-------------|
| `--json` | Output as JSON |

### `cairn activity`

Show recent session activity overview.

```
cairn activity [--days N] [--json]
```

| Option | Description |
|--------|-------------|
| `--days N` | Number of days to show (default: 7) |
| `--json` | Output as JSON |

### `cairn logs`

Show recent hook errors from hooks.log.

```
cairn logs [-n LINES]
```

| Option | Description |
|--------|-------------|
| `-n`, `--lines` | Number of lines to show (default: 50) |

---

## Knowledge Commands

### `cairn knowledge scan`

Scan ~/.cairn/documents/ for new or changed files and auto-ingest. Alias: `cairn kb scan`.

```
cairn knowledge scan [--dir PATH]
```

| Option | Description |
|--------|-------------|
| `--dir` | Custom directory to scan (default: ~/.cairn/documents/) |

### `cairn knowledge list`

List all ingested documents with chunk counts and metadata.

```
cairn knowledge list
```

### `cairn knowledge search`

Search across ingested documents using vector similarity.

```
cairn knowledge search <query> [--limit N]
```

| Option | Description |
|--------|-------------|
| `<query>` | Search query (positional) |
| `--limit N` | Max results (default: 5) |

---

## Cloud Commands

### `cairn cloud setup`

Configure Supabase connection for cloud sync.

```
cairn cloud setup [--url URL] [--key KEY] [--service-key KEY]
```

| Option | Description |
|--------|-------------|
| `--url` | Supabase project URL |
| `--key` | Supabase anon key |
| `--service-key` | Supabase service role key (optional) |

### `cairn cloud sync`

Sync local data to Supabase cloud.

```
cairn cloud sync
```

### `cairn cloud pull`

Pull memories and documents from Supabase cloud.

```
cairn cloud pull
```

### `cairn cloud status`

Show cloud sync status.

```
cairn cloud status
```

### `cairn cloud schema`

Print the Supabase SQL schema for manual setup.

```
cairn cloud schema
```

### `cairn cloud verify`

Verify the Supabase connection is working.

```
cairn cloud verify
```

---

## Mobile Commands

### `cairn mobile setup`

Print setup instructions for mobile access via mcp-proxy + Tailscale.

```
cairn mobile setup
```

### `cairn mobile serve`

Start an mcp-proxy HTTP server for mobile access.

```
cairn mobile serve [--port PORT] [--host HOST]
```

| Option | Description |
|--------|-------------|
| `--port` | HTTP port (default: 8089) |
| `--host` | Bind address (default: 127.0.0.1) |

---

## Migration Commands

### `cairn migrate`

Copy MAGMA data to Cairn (non-destructive legacy migration).

```
cairn migrate
```

### `cairn migrate-db`

Migrate legacy JSON graphs to the SQLite backend.

```
cairn migrate-db [--force]
```

| Option | Description |
|--------|-------------|
| `--force` | Overwrite existing SQLite database |

### `cairn reingest`

Reload store.jsonl entries into the graph system.

```
cairn reingest
```
