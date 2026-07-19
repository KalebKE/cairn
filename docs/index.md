---
title: Cairn
description: Persistent memory for AI coding agents
---

# Cairn — Persistent memory for AI coding agents

**Stop losing context. Stop repeating mistakes. Stop colliding with teammates.**

Cairn gives AI coding agents a durable memory layer that survives across sessions, projects, and teams. Every decision, lesson, and preference is captured, indexed, and surfaced exactly when it matters.

## The problem

Without persistent memory, AI coding agents:

- **Lose context between sessions** — every conversation starts from zero, re-discovering architecture decisions, coding conventions, and project history.
- **Repeat the same mistakes** — debugging a tricky issue today teaches nothing to tomorrow's session. The same pitfalls get hit over and over.
- **Collide in multi-agent workflows** — two agents editing the same file, pushing to the same branch, or duplicating work with no awareness of each other.

## The solution

Cairn is a local-first, SQLite-backed memory system that integrates with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via the Model Context Protocol (MCP). It runs entirely on your machine — no cloud required, no data leaves your device unless you opt in.

## Quick install

```bash
pip install cairn
cairn setup
```

That's it. Cairn registers itself as an MCP server, installs hooks into Claude Code, and starts capturing memories automatically.

## Key features

- **Persistent Memory** (24 tools) — Store, query, and traverse decisions, lessons, preferences, and session summaries with semantic search powered by BGE-Small embeddings.
- **Multi-Agent Coordination** (28 tools) — File claims, branch ownership, intent broadcasting, task management, and messaging so agents work together without conflicts.
- **LLM Routing** (10 tools, optional) — Route prompts to the optimal model across 5 providers based on intent classification, priority mode, and context size.
- **Knowledge Base** (optional) — Ingest PDFs, web pages, and documents into a chunked vector store for RAG-style retrieval over your own files.
- **Entity Registry** (optional) — Track companies, LLCs, and organizational structures with typed relationships and entity-scoped memories.
- **Secure Profile** (optional) — AES-256 encrypted storage for sensitive personal data with macOS Keychain integration.

## How it compares

| Feature | Cairn | Mem0 | Zep | Copilot Memory |
|---------|-------|------|-----|----------------|
| **Local-first** | Yes — SQLite on your machine | Cloud-hosted | Cloud or self-hosted | Cloud-only |
| **Multi-agent coordination** | 28 tools (claims, tasks, messaging) | No | No | No |
| **LLM routing** | 10 tools, 5 providers | No | No | No |
| **Document RAG** | PDF + web + markdown ingestion | No | Yes | No |
| **Privacy** | Nothing leaves your machine | Data on their servers | Depends on deployment | Data on GitHub servers |
| **MCP native** | Yes — 70 tools via MCP | REST API | REST API | Proprietary |
| **Open source** | Yes (MIT) | Partial | Partial | No |

!!! tip "Local-first means private by default"
    Cairn stores everything in `~/.cairn/cairn.db` on your machine. Cloud sync to Supabase is available but entirely opt-in.

## Architecture at a glance

```
Claude Code <──MCP (stdio)──> Cairn Server
                                  │
                    ┌──────────────┼──────────────┐
                    │              │              │
              SQLite + FTS5   sqlite-vec    ONNX Embeddings
              (memories,      (vector       (bge-small-en-v1.5,
               coordination)   search)       CPU-only, ~337MB)
```

Cairn runs as a stdio MCP server spawned by Claude Code on demand. Seven hook processes handle automatic memory capture, surfacing, and coordination — all fail-open so they never block your workflow.

## Next steps

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Getting Started](getting-started/installation.md)**

    Install Cairn, run setup, and verify everything works in under 2 minutes.

-   :material-lightning-bolt: **[Quickstart](getting-started/quickstart.md)**

    Store your first memory, close the session, and watch it come back.

-   :material-cog: **[Configuration](getting-started/configuration.md)**

    Storage paths, environment variables, hook configuration, and MCP settings.

</div>
