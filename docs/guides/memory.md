# Working with Memory

## Overview

Cairn stores memories as semantically embedded nodes in a graph, enabling cross-session recall for AI coding agents. Each memory has:

- **Event type**: `decision`, `lesson_learned`, `error_pattern`, `task_completion`, `session_summary`, `user_preference`, `checkpoint`
- **Priority**: 1-5 (5 = highest). Auto-set from event type if omitted.
- **Tags**: Auto-extracted from content for filtering and boost scoring.
- **TTL**: Time-to-live varies by type (session summaries expire in 1 day; lessons and preferences are permanent).
- **Entity scope**: Optionally scope memories to a corporate entity (e.g., `entity_id="acme"`).

Memories are stored in SQLite with FTS5 full-text search and sqlite-vec vector embeddings (bge-small-en-v1.5, ONNX CPU-only). Retrieval blends semantic similarity (70%) with BM25 text search (30%), boosted by word overlap, tag matching, and feedback signals.

Use memory when you need to persist decisions, capture lessons from debugging sessions, save task progress across context windows, or recall prior context before starting work.

## Quick Example

```
# Store a decision
cairn_store(content="Use PostgreSQL for the analytics service — need window functions and JSONB", event_type="decision")

# Query later
cairn_query(query="database choice for analytics")

# User says "remember this"
cairn_remember(text="Deploy window is Tuesdays 2-4pm PST")
```

## Tools Reference

| Tool | Purpose |
|------|---------|
| `cairn_remember` | Store permanent memory (use when user says "remember this") |
| `cairn_store` | Store typed memory with event_type, priority, entity_id, metadata |
| `cairn_query` | Semantic search with filters (entity_id, event_type, filter_tags, temporal_range, context_file, context_tags) |
| `cairn_phrase_search` | Exact substring/phrase match via FTS5 (case-insensitive by default) |
| `cairn_lessons` | Cross-session/project lessons ranked by verification count and access frequency |
| `cairn_similar` | Find memories similar to a given memory ID |
| `cairn_traverse` | Walk the relationship graph from a memory (1-5 hops, min_weight filter) |
| `cairn_timeline` | View memories grouped by day (configurable lookback and limit per day) |
| `cairn_checkpoint` | Save task state: plan, progress, files touched, decisions, key context, next steps |
| `cairn_resume_task` | Resume a checkpointed task with full/summary/minimal verbosity |
| `cairn_compact` | Cluster similar memories and create summary nodes (Jaccard similarity threshold) |
| `cairn_consolidate` | Prune stale zero-access memories, cap session summaries, clean orphaned edges |
| `cairn_feedback` | Rate a memory as helpful, unhelpful, or outdated (affects future retrieval scoring) |
| `cairn_delete_memory` | Delete a specific memory by ID |
| `cairn_edit_memory` | Edit the content of a specific memory by ID |
| `cairn_type_stats` | Memory counts grouped by event type |
| `cairn_session_stats` | Memory counts grouped by session (top 20) |
| `cairn_welcome` | Session briefing with recent relevant memories and user profile |
| `cairn_health` | Detailed health check: memory usage, node counts, cache stats, warnings |
| `cairn_backup` | Export or import memories for backup/restore |
| `cairn_list_preferences` | List all stored user preferences |
| `cairn_save_profile` | Save or update user profile (name, timezone, role, preferences) |
| `cairn_profile` | Show the user profile built from memory patterns |
| `cairn_clear_session` | Clear all memories for a specific session |

## Common Workflows

### Storing Memories

There are three ways memories enter the system:

**1. Explicit remember** --- when the user says "remember this":
```
cairn_remember(text="The staging environment uses port 8443")
```

**2. Typed store** --- for structured capture with metadata:
```
cairn_store(
    content="Switched from REST to gRPC for inter-service calls — 3x latency improvement",
    event_type="decision",
    priority=4,
    entity_id="acme"
)
```

**3. Auto-capture** --- hooks detect decisions and lessons in conversation and store them automatically via the `UserPromptSubmit` hook. No manual action needed.

### Querying Memories

**Semantic search** --- finds conceptually related memories:
```
cairn_query(query="authentication flow for mobile app")
```

**Filtered search** --- narrow by type, tags, entity, or time:
```
cairn_query(
    query="database migration",
    event_type="decision",
    filter_tags=["postgres"],
    entity_id="acme",
    temporal_range=["2026-01-01", "2026-02-01"]
)
```

**Context-aware search** --- boost results relevant to what you are editing:
```
cairn_query(
    query="error handling patterns",
    context_file="src/api/handler.py",
    context_tags=["python", "fastapi"]
)
```

**Exact phrase search** --- for specific strings, error messages, or known terms:
```
cairn_phrase_search(phrase="ECONNREFUSED", event_type="error_pattern")
```

**Cross-session lessons** --- ranked by how often they have been verified:
```
cairn_lessons(task="setting up CI pipeline", limit=5, cross_project=True)
```

### Memory Types and TTLs

| Event Type | Default TTL | Priority | Use Case |
|------------|-------------|----------|----------|
| `session_summary` | 1 day | 2 | Auto-generated session recaps |
| `task_completion` | 7 days | 3 | Completed task records |
| `checkpoint` | 7 days | 5 | Context virtualization snapshots |
| `error_pattern` | 30 days | 3 | Recurring error signatures |
| `decision` | Permanent | 4 | Architectural and technical decisions |
| `lesson_learned` | Permanent | 4 | Debugging insights and best practices |
| `user_preference` | Permanent | 5 | User preferences and conventions |

### Context Virtualization

When the context window is getting full (>70% capacity), checkpoint your work:

```
cairn_checkpoint(
    task_title="API redesign Phase 2",
    plan="Migrate all endpoints to v2 schema",
    progress="Completed users and orders endpoints. Auth endpoints remain.",
    files_touched={"src/api/users.py": "Migrated to v2 schema", "src/api/orders.py": "Migrated to v2 schema"},
    decisions=["Keep backward compat for v1 until March", "Use Pydantic v2 model_validator"],
    key_context="V2 schema uses camelCase keys. Auth endpoints depend on the new JWT middleware in src/middleware/auth.py.",
    next_steps="Migrate src/api/auth.py and src/api/billing.py, then update integration tests."
)
```

In a new session, resume where you left off:

```
cairn_resume_task(task_title="API redesign", verbosity="full")
```

Verbosity levels:
- `full` --- plan + progress + files + decisions + key context + next steps
- `summary` --- plan + progress + next steps
- `minimal` --- just next steps

### Graph Traversal

Memories form a graph through automatic relationship edges. Traverse to discover related context:

```
cairn_traverse(memory_id="mem_abc123", max_hops=2, min_weight=0.3)
```

Find memories similar to a known one:

```
cairn_similar(memory_id="mem_abc123", limit=5)
```

### Maintenance

**Consolidate** --- prune stale memories and clean up (run weekly, auto-triggered by hooks):
```
cairn_consolidate(prune_days=30, max_summaries=50)
```

**Compact** --- cluster similar memories into summary nodes (run biweekly):
```
cairn_compact(event_type="lesson_learned", similarity_threshold=0.6, min_cluster_size=3)
```

Preview clusters without compacting:
```
cairn_compact(dry_run=True)
```

**Timeline** --- review what was captured recently:
```
cairn_timeline(days=7, limit_per_day=10)
```

**Feedback** --- improve retrieval quality over time:
```
cairn_feedback(memory_id="mem_abc123", rating="helpful")
cairn_feedback(memory_id="mem_xyz789", rating="outdated", reason="We switched to Redis")
```

## Tips

- **Query before you work.** Run `cairn_query` at the start of non-trivial tasks to surface prior decisions, gotchas, and lessons. This prevents re-discovering known issues.
- **Check Cairn on errors.** Before debugging from scratch, search for prior solutions with `cairn_phrase_search` on the error message.
- **Let hooks do the work.** Auto-capture detects decisions and lessons in conversation. You do not need to manually store everything.
- **Checkpoint early.** Do not wait until the context window is completely full. Checkpoint at major milestones or when you notice things slowing down.
- **Use entity scoping.** If you work across multiple organizations or projects, scope memories with `entity_id` to keep contexts separate.
- **Rate memories.** Using `cairn_feedback` with `helpful` or `unhelpful` directly affects future retrieval scoring. Marking memories `outdated` dampens them.
- **Compact periodically.** If you accumulate many similar lessons (e.g., repeated debugging insights for the same system), `cairn_compact` clusters them into clean summaries.
- **Phrase search for exact matches.** Semantic search is fuzzy by design. When you need an exact error message or specific identifier, use `cairn_phrase_search`.
- **Temporal ranges narrow results.** If you know roughly when something happened, add `temporal_range` to your query to avoid pulling in old, irrelevant memories.
- **Backup regularly.** `cairn_backup(filepath="~/cairn-backup.json")` exports everything. The auto-maintenance hooks back up weekly, but manual backups before risky operations are wise.
