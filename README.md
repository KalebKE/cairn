# Cairn

**Self-hosted, local-first memory for AI coding agents.** One SQLite store, shared across every repo, worktree, and session — served over MCP to Claude Code or any MCP client. Your agent's memory lives on your machine, not someone else's server.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1765%20passing-brightgreen.svg)]()

## What it does

- **Capture**: decisions, lessons, errors, and preferences — stored explicitly (`cairn_store`) or auto-captured by session hooks, with Jaccard dedup, Zettelkasten-style memory *evolution*, contradiction supersession, and noise blocklists.
- **Retrieve**: hybrid search fusing four channels (vector · BM25/FTS5 · temporal · entity) with canonical Reciprocal Rank Fusion, then a confidence-gated cross-encoder rerank (`bge-reranker-v2-m3`). Embeddings are local ONNX (`bge-small-en-v1.5`, 384-d, cosine).
- **Forget**: per-type exponential decay (ACT-R style, access-aware), TTLs, strength floors, and an audited `forgetting_log` — protected types (decisions, preferences, constraints) never decay.
- **Measure**: a built-in eval harness with **non-self-referential judged probe sets**, paired A/B on any config knob, sign tests, and an MRR trend that `cairn doctor` watches for silent degradation. Current baseline on a 2,300-memory live store: **MRR 0.842, nDCG@5 0.71, hit-rate 0.92**.

Every scoring change in this fork was either proven equivalent or measured on judged probes before it shipped.

## Quick start

```bash
git clone git@github.com:KalebKE/cairn.git && cd cairn
pip install -e .
cairn doctor                                   # model / db / embeddings health
claude mcp add -s user cairn -- cairn serve    # register (stdio, per-session)
```

Data lives in `~/.cairn` (override with `CAIRN_HOME`). The MCP surface is 15 composite tools (`cairn_store`, `cairn_query`, `cairn_welcome`, `cairn_memory`, `cairn_maintain`, `cairn_stats`, `cairn_reflect`, …) defined in a single `ToolSpec` registry; condensed mode exposes 5 and routes the rest through `cairn_call`.

## Evaluating & tuning

```bash
cairn eval-retrieval --build-probes --sample-size 30   # LLM-judged probe set (frozen, reusable)
cairn eval-retrieval --probes <set.json>               # provider-free scored run
cairn eval-retrieval --probes <set.json> --ab CAIRN_CE_MODE=boost|hybrid
cairn eval-retrieval --build-probes --from-query-log   # replay real logged queries
```

Evals always run against a snapshot copy — never the live store (queries mutate access counts, which feed decay).

## Concurrency model

Per-session stdio servers over one WAL-mode SQLite store: N sessions across N repos/worktrees share memory safely (single-writer with busy-timeout retries; cross-session dedup keeps parallel sessions from storing the same fact twice). There is deliberately **no** inter-agent coordination layer — one agent per worktree is the intended pattern; git is the isolation.

## Provenance

Cairn began as a fork of [omega-memory](https://github.com/omega-memory/omega-memory) v1.5.5 (Apache-2.0 — see `LICENSE` and `NOTICE`, both retained). It is independently maintained and has diverged substantially: the commercial/coordination/cloud/freemium layers were removed (~40% of the upstream tree), the integration layer was rebuilt (modular bridge, derived tool registry, `RetrievalContext` pipeline), and the eval/tuning infrastructure is original. Roughly a third of the current code is post-fork; the retrieval core's bones are upstream's, kept because they audit well against the literature.

## License

Apache-2.0. Upstream attribution preserved in `NOTICE`.
