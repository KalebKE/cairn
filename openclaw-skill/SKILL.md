---
name: cairn
description: Persistent memory for AI coding agents. Semantic search, auto-capture, checkpoint/resume across sessions.
version: 1.0.0
requires_binaries: ["python3", "pip3"]
requires_env: []
---

# Cairn

Persistent memory for AI coding agents. Your agent remembers decisions, learns from mistakes, and picks up where it left off.

## Installation

```bash
pip3 install cairn[server]
cairn setup
```

The `cairn setup` command auto-configures your MCP client (Claude Code, Cursor, Windsurf, or Zed). No API keys needed — runs fully local with CPU-only embeddings.

## What It Does

Cairn gives your agent a persistent memory layer across coding sessions:

- **Decisions & context** carry forward — no re-explaining
- **Lessons learned** from errors are recalled before you repeat them
- **Checkpoint/resume** lets you pause complex tasks and pick up later
- **Semantic search** over all stored memories with contextual re-ranking

## MCP Tools (12 tools)

| Tool | Purpose |
|------|---------|
| `cairn_welcome` | Session briefing with recent memories and profile |
| `cairn_protocol` | Retrieve operating rules and behavioral guidelines |
| `cairn_store` | Store typed memory (decision, lesson, error, preference, summary) |
| `cairn_query` | Semantic or phrase search with tag filters and re-ranking |
| `cairn_lessons` | Cross-session lessons ranked by access count |
| `cairn_profile` | Read or update the user profile |
| `cairn_checkpoint` | Save task state for cross-session continuity |
| `cairn_resume_task` | Resume a previously checkpointed task |
| `cairn_memory` | Manage a specific memory (edit, delete, feedback, similar, traverse) |
| `cairn_remind` | Set, list, or dismiss time-based reminders |
| `cairn_maintain` | System housekeeping (health, consolidate, compact, backup, restore) |
| `cairn_stats` | Analytics: type breakdown, session stats, weekly digest, access rates |

## Usage Pattern

At the start of every session:
1. Call `cairn_welcome()` for context briefing
2. Call `cairn_protocol()` for operating instructions
3. Follow the protocol it returns

During work:
- Before non-trivial tasks: `cairn_query()` to check for prior context
- After completing tasks: `cairn_store(content, "decision")` to save key outcomes
- When context is getting full: `cairn_checkpoint()` to save state

## Links

- PyPI: https://pypi.org/project/cairn/
- GitHub: https://github.com/TracqiTechnology/cairn
- Website: https://tracqi.com
