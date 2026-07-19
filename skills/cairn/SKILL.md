---
name: cairn
description: "Persistent memory for AI coding agents. Teaches agents how to use Cairn's MCP tools for storing decisions, querying context, coordinating multi-agent workflows, and resuming tasks across sessions."
license: Apache-2.0
compatibility: "Python 3.11+, Claude Code, Cursor, Windsurf, Zed"
metadata:
  category: memory
  pypi: cairn
  github: cairn/cairn
---

# Cairn

Persistent memory for AI coding agents. Cairn gives your agent a knowledge graph it can query, learn from, and coordinate through across sessions.

This skill teaches you how to use Cairn's MCP tools effectively.

## Setup

```bash
pip3 install cairn[server]
cairn setup        # auto-configures your editor + downloads embedding model
cairn doctor       # verify everything works
```

Works with Claude Code, Cursor, Windsurf, Zed, and any MCP client.

## Core Tools

Cairn provides 12 MCP tools. Here's when to use each one.

### Storing Memories

**`cairn_store(content, event_type, metadata?, entity_id?)`**

Store decisions, lessons, and context that should persist across sessions.

| Event Type | When to Use | TTL |
|------------|------------|-----|
| `decision` | Architectural choices, technology selections | 90 days |
| `lesson_learned` | Debugging insights, patterns that worked/failed | 90 days |
| `user_preference` | Code style, workflow preferences, tool choices | Permanent |
| `error_pattern` | Recurring errors and their fixes | 30 days |
| `task_completion` | Completed work with outcomes | 14 days |
| `checkpoint` | Mid-task state for resumption | 7 days |

```
cairn_store("Switched from REST to GraphQL for the dashboard API — reduces N+1 queries", "decision")
cairn_store("User prefers early returns, max 2 levels of nesting", "user_preference")
cairn_store("pytest fixtures with db cleanup must use function scope, not session scope", "lesson_learned")
```

**Don't store:** Raw code output, tool results, transient status updates, anything shorter than a sentence.

### Querying Memories

**`cairn_query(query, mode?, limit?, entity_id?)`**

Search memories by meaning, not just keywords. Uses hybrid retrieval: vector similarity + full-text search + cross-encoder reranking.

| Mode | When to Use |
|------|------------|
| `semantic` (default) | Find memories by meaning — "how did we handle auth?" |
| `phrase` | Exact substring match — find a specific term or identifier |
| `timeline` | Recent memories grouped by day — "what happened this week?" |
| `browse` | List by type, session, or recency — explore what's stored |

```
cairn_query("database migration strategy")
cairn_query("what decisions were made about the API", mode="timeline", days=7)
cairn_query("pytest", mode="phrase")
cairn_query(mode="browse", browse_by="type")
```

**Pro tip:** Query before starting work. Prior decisions and lessons save time and prevent repeating mistakes.

### Session Management

**`cairn_welcome(project?)`** — Call at session start. Returns recent context, active reminders, and project state. This is how your agent picks up where it left off.

**`cairn_checkpoint()`** — Save current task state mid-session. If the session ends unexpectedly, the next `cairn_welcome` restores this context.

**`cairn_resume_task(task_id)`** — Resume a previously checkpointed task with full context.

### Memory Maintenance

**`cairn_reflect()`** — Analyze memory quality: duplicates, contradictions, coverage gaps.

**`cairn_maintain(action)`** — Run maintenance operations: consolidation, compaction, health checks.

## Retrieval Architecture

Cairn's query pipeline runs 7 phases to find the most relevant memories:

1. **Vector similarity** — Embedding search (bge-small-en-v1.5, 384-dim) via sqlite-vec
2. **Full-text search** — FTS5 with BM25 scoring
3. **Strong signal short-circuit** — Skip expensive phases when FTS5 finds an exact match
4. **Score fusion** — Reciprocal Rank Fusion combines vector + text scores
5. **Contextual boosting** — Boost results matching current file, project, or tags
6. **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 rescores top candidates
7. **Assembly** — Dedup, normalize, apply minimum relevance threshold

This hybrid approach achieves 95.4% on LongMemEval (500-question benchmark).

## Best Practices

### What to Store

- Architectural decisions with reasoning ("chose X because Y")
- Debugging insights that took effort to discover
- User preferences stated explicitly ("always use..." / "never...")
- Cross-session context that future sessions need

### What NOT to Store

- Information already in the codebase (read the code instead)
- Transient state (build output, test results)
- Anything shorter than a meaningful sentence
- Speculative conclusions from reading a single file

### Query Patterns That Work

- **Before starting a task:** `cairn_query("prior decisions about [feature area]")`
- **Before modifying a file:** `cairn_query(context_file="/path/to/file.py")`
- **After debugging:** `cairn_store("[root cause and fix]", "lesson_learned")`
- **When user says "remember":** `cairn_store("[what they said]", "user_preference")`

### Anti-Patterns

| Don't | Do Instead |
|-------|-----------|
| Store every tool result | Store only insights and decisions |
| Query with single words | Use natural language questions |
| Skip `cairn_welcome` at session start | Always call it — it loads critical context |
| Store without `event_type` | Always specify type for proper TTL and dedup |
| Guess from stale memory | Query Cairn to verify current state |

## How It Works Under the Hood

- **Storage:** SQLite with WAL mode. Single file at `~/.cairn/cairn.db`.
- **Embeddings:** bge-small-en-v1.5 via ONNX Runtime (~90MB RAM). LRU cache (512 entries).
- **Vector search:** sqlite-vec extension for ANN similarity search.
- **Text search:** FTS5 with BM25 ranking.
- **Dedup:** Jaccard similarity with per-type thresholds (0.70-0.90). Content-level and embedding-level.
- **Memory evolution:** Similar memories merge (Zettelkasten-style) instead of creating duplicates.
- **TTL:** Automatic expiry based on event type. Permanent for preferences, 7-90 days for others.
- **Privacy:** Everything stays local. No cloud, no telemetry. Apache-2.0 licensed.

## Links

- **PyPI:** [cairn](https://pypi.org/project/cairn/)
- **GitHub:** [cairn/cairn](https://github.com/TracqiTechnology/cairn)
- **Docs:** [tracqi.com/docs](https://tracqi.com/docs)
- **Benchmarks:** [tracqi.com/benchmarks](https://tracqi.com/benchmarks)
