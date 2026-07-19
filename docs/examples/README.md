# Cairn Usage Examples

Practical examples for common workflows using the CLI and Python API.

## Table of Contents

- [Storing Memories](#storing-memories)
- [Querying Context](#querying-context)
- [Checkpoint and Resume](#checkpoint-and-resume)
- [Maintenance](#maintenance)
- [Reminders](#reminders)
- [Import and Export](#import-and-export)
- [Scripting and Automation](#scripting-and-automation)

---

## Storing Memories

### CLI

```bash
# Store a decision
cairn store "We chose PostgreSQL over MongoDB for ACID transaction support" --type decision

# Store a user preference
cairn store "Always use early returns, never nest more than 2 levels" --type user_preference

# Store a lesson learned from debugging
cairn store "Docker node_modules volume mount shadows container deps -- use anonymous volume" --type lesson

# Store with tags for better retrieval
cairn store "API rate limit is 100 req/min per user" --type decision --tags api,rate-limit
```

### Python API

```python
from cairn import store, remember

# Store a decision
store("We chose PostgreSQL over MongoDB for ACID transaction support", "decision")

# Store a preference (shorthand -- auto-tags as user_preference)
remember("Always use early returns, never nest more than 2 levels")

# Store with metadata
store(
    "API uses JWT tokens with 15-minute expiry, refresh tokens last 7 days",
    "decision",
    metadata={"tags": ["auth", "jwt"], "project": "backend-api"},
)

# Batch store multiple memories at once
from cairn import batch_store

batch_store([
    {"content": "Use pnpm, not npm", "event_type": "user_preference"},
    {"content": "CI runs on GitHub Actions", "event_type": "decision"},
    {"content": "Flaky test: retry network calls in test_sync.py", "event_type": "lesson"},
])
```

---

## Querying Context

### CLI

```bash
# Semantic search -- finds relevant memories even with different wording
cairn query "database choice for orders"

# Filter by type
cairn query "auth" --type decision

# View recent memory timeline
cairn timeline

# View timeline for last 14 days
cairn timeline --days 14

# Check what Cairn knows about a topic
cairn query "Docker deployment gotchas"
```

### Python API

```python
from cairn import query, timeline, find_similar_memories

# Basic semantic search
results = query("database choice for orders")
print(results)

# Filter by type and limit results
results = query("deployment", event_type="lesson", limit=5)

# Search with tag filter -- only memories tagged "auth"
results = query("token expiry", filter_tags=["auth"])

# View recent timeline
print(timeline(days=7))

# Find memories similar to an existing one
similar = find_similar_memories("memory-id-here", limit=3)
```

---

## Checkpoint and Resume

Use checkpoints to save task state mid-work and resume in a later session.

### In Claude Code (via MCP tools)

During a session, tell Claude:

> "Checkpoint this -- I'm halfway through migrating auth to the new middleware pattern. Files changed: auth.py, middleware.py. Still need to update tests and the login route."

Claude calls `cairn_checkpoint` automatically. In your next session:

> "Resume the auth middleware migration."

Claude calls `cairn_resume_task` and picks up with full context.

### CLI

```bash
# List recent activity to find checkpointed tasks
cairn activity

# Query for checkpointed tasks
cairn query "auth middleware migration" --type checkpoint
```

### Python API

```python
from cairn import store, query

# Store a checkpoint manually
store(
    "Migrating auth middleware: auth.py and middleware.py updated. "
    "TODO: update tests in test_auth.py and login route in routes/auth.py",
    "checkpoint",
    metadata={"task": "auth-middleware-migration", "progress": "50%"},
)

# Resume by querying for the checkpoint
results = query("auth middleware migration", event_type="checkpoint", limit=1)
print(results)
```

---

## Maintenance

### CLI

```bash
# Check installation health
cairn doctor

# View memory stats
cairn stats

# Deduplicate and prune old session summaries
cairn consolidate

# Cluster and summarize related memories
cairn compact

# Back up the database (keeps last 5 backups)
cairn backup

# Validate database integrity
cairn validate

# View hook errors
cairn logs
```

### Python API

```python
from cairn import check_health, consolidate, compact, status, type_stats

# Quick status check
print(status())
# => {'node_count': 142, 'db_size_mb': 4.2, 'backend': 'sqlite', ...}

# Health check
print(check_health())

# Memory type breakdown
print(type_stats())

# Consolidate (deduplicate + prune)
print(consolidate(prune_days=30))

# Compact (cluster + summarize)
print(compact())
```

---

## Reminders

### CLI

```bash
# Reminders are managed through Claude via the cairn_remind MCP tool.
# Ask Claude: "Remind me to update the API docs before the release next Friday"
```

### Python API

```python
from cairn import create_reminder, list_reminders, get_due_reminders, dismiss_reminder

# Create a reminder
create_reminder(
    content="Update API docs before release",
    due_at="2026-03-15T09:00:00Z",
)

# List all active reminders
reminders = list_reminders()
for r in reminders:
    print(f"{r['due_at']}: {r['content']}")

# Check what's due now
due = get_due_reminders(mark_fired=True)

# Dismiss a reminder
dismiss_reminder(reminder_id="reminder-id-here")
```

---

## Import and Export

### CLI

```bash
# Export all memories to JSON
cairn export memories.json

# Import memories from JSON (replaces existing)
cairn import memories.json
```

### Python API

```python
from cairn import export_memories, import_memories

# Export
export_memories("/tmp/cairn-backup.json")

# Import (clears existing memories first by default)
import_memories("/tmp/cairn-backup.json")

# Import without clearing existing
import_memories("/tmp/cairn-backup.json", clear_existing=False)
```

---

## Scripting and Automation

### CI/CD: Store deployment context

```python
#!/usr/bin/env python3
"""Post-deploy hook: store deployment context in Cairn."""
import os
from cairn import store

store(
    f"Deployed {os.environ['GIT_SHA'][:8]} to {os.environ['DEPLOY_ENV']}. "
    f"Branch: {os.environ.get('GIT_BRANCH', 'unknown')}",
    "decision",
    metadata={
        "tags": ["deploy", os.environ["DEPLOY_ENV"]],
        "project": os.environ.get("PROJECT_NAME", "unknown"),
    },
)
```

### Pre-commit: Auto-capture decisions from commit messages

```bash
#!/bin/sh
# .git/hooks/post-commit
MSG=$(git log -1 --pretty=%B)
# Store architectural decisions (commits starting with "decision:" or "ADR:")
case "$MSG" in
  decision:*|ADR:*)
    cairn store "$MSG" --type decision
    ;;
esac
```

### Session bootstrap script

```python
#!/usr/bin/env python3
"""Print a quick context briefing for the current project."""
from cairn import query, status

s = status()
print(f"Cairn: {s['node_count']} memories, {s['db_size_mb']:.1f} MB")

# Surface key decisions for this project
results = query("key decisions and preferences", event_type="decision", limit=5)
print(results)
```
