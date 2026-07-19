#!/usr/bin/env python3.11
"""Precision@K audit for Cairn's query pipeline.

Measures Precision@1, Precision@3, and Precision@5 against a curated set of
test queries with ground-truth keyword relevance judgments.

Inspired by arxiv 2603.02473 -- Precision@5 correlates with accuracy at r=0.98.

Usage:
    python3.11 scripts/precision_audit.py
    python3.11 scripts/precision_audit.py --verbose
    python3.11 scripts/precision_audit.py --limit 3   # top-3 only
    python3.11 scripts/precision_audit.py --json       # machine-readable output
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Bootstrap: ensure src/ is importable
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from cairn.sqlite_store import SQLiteStore  # noqa: E402


# ---------------------------------------------------------------------------
# Test queries with ground-truth relevance keywords
# ---------------------------------------------------------------------------
# Each entry: (category, query_text, [ground_truth_keywords])
#
# A result is "relevant" if its content (case-insensitive) contains ANY of the
# ground-truth keywords/phrases.  Keywords are intentionally loose to account
# for paraphrasing -- e.g. "reranker" matches a result about cross-encoders.
#
# Categories: factual, conceptual, navigational, temporal, preference
# ---------------------------------------------------------------------------

TEST_QUERIES: List[Tuple[str, str, List[str]]] = [
    # --- Factual: specific facts about Cairn ---
    (
        "factual",
        "What embedding model does Cairn use?",
        ["embedding", "onnx", "all-minilm", "sentence-transformer", "384"],
    ),
    (
        "factual",
        "What Python version does Cairn require?",
        ["python", "3.11", "python3.11"],
    ),
    (
        "factual",
        "How many MCP tools does Cairn expose?",
        ["tool", "schema", "handler", "mcp"],
    ),
    (
        "factual",
        "What reranker model does Cairn support?",
        ["reranker", "cross-encoder", "bge", "ms-marco", "minilm", "onnx"],
    ),
    # --- Conceptual: architecture and design decisions ---
    (
        "conceptual",
        "How does Cairn's query pipeline combine vector and text search?",
        ["vector", "fts", "fusion", "rrf", "reciprocal rank", "composite", "rerank"],
    ),
    (
        "conceptual",
        "What is the memory decay model in Cairn?",
        ["decay", "lambda", "exponential", "floor", "access_count", "ttl"],
    ),
    (
        "conceptual",
        "How does Cairn handle multi-agent coordination?",
        ["coordination", "agent", "session", "worktree", "peer", "lock", "claim"],
    ),
    (
        "conceptual",
        "What deduplication strategies does Cairn use for memories?",
        ["dedup", "jaccard", "embedding", "cosine", "threshold", "similarity", "duplicate"],
    ),
    (
        "conceptual",
        "How does the hook system dispatch events in Cairn?",
        ["hook", "fast_hook", "dispatch", "event", "pre_", "post_"],
    ),
    # --- Navigational: finding specific code or config ---
    (
        "navigational",
        "Where is the protocol definition for Cairn sessions?",
        ["protocol", "protocol.py", "session", "operating instruction"],
    ),
    (
        "navigational",
        "Which file handles the cairn_store MCP tool?",
        ["handler", "store", "server", "tool_schema", "mcp_server"],
    ),
    (
        "navigational",
        "How is the Cairn website deployed?",
        ["vercel", "cairnmax", "website", "deploy", "next.js"],
    ),
    # --- Temporal: recent events and changes ---
    (
        "temporal",
        "What recent changes were made to the Cairn website?",
        ["website", "cairnmax", "vercel", "next.js", "deploy"],
    ),
    (
        "temporal",
        "What was the last benchmark result for Cairn?",
        ["benchmark", "longmemeval", "precision", "accuracy", "score", "f1"],
    ),
    # --- User preference: stored user decisions ---
    (
        "preference",
        "What are the user's rules about sending emails?",
        ["email", "pre-flight", "approval", "duplicate", "never auto-send", "send_email"],
    ),
    (
        "preference",
        "What are the tweet posting rules for @cairn_memory?",
        ["tweet", "approval", "cairn_memory", "post", "admin", "403", "gate"],
    ),
    (
        "preference",
        "Where is the user based? What timezone?",
        ["singapore", "timezone", "asia", "sgt", "utc+8"],
    ),
    (
        "preference",
        "How should images be generated?",
        ["gemini", "imagen", "replicate", "image", "banana"],
    ),
    # --- Conversational/vague: realistic user queries ---
    (
        "conversational",
        "that thing about deployments",
        ["deploy", "vercel", "push", "production", "build"],
    ),
    (
        "conversational",
        "the convergence decision",
        ["convergence", "migration", "merge", "consolidate", "absorb"],
    ),
    (
        "conversational",
        "what we decided about the reranker",
        ["reranker", "cross-encoder", "bge", "ms-marco", "onnx"],
    ),
]


def is_relevant(content: str, ground_truth: List[str]) -> bool:
    """Check if content is relevant based on ground-truth keywords."""
    content_lower = content.lower()
    return any(kw.lower() in content_lower for kw in ground_truth)


def precision_at_k(results, ground_truth: List[str], k: int) -> float:
    """Compute Precision@K for a single query."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if is_relevant(r.content, ground_truth))
    return relevant / k


def run_audit(
    store: SQLiteStore,
    verbose: bool = False,
    json_output: bool = False,
    limit: int = 5,
) -> Dict:
    """Run the full precision audit and return results."""

    k_values = sorted(set([1, 3, limit]))  # Always measure P@1, P@3, P@limit
    per_query_results = []
    category_scores: Dict[str, List[float]] = {}
    total_latencies = []

    for category, query_text, ground_truth in TEST_QUERIES:
        t0 = time.monotonic()
        results = store.query(
            query_text=query_text,
            limit=max(limit, 5),  # Always fetch at least 5
            entity_id="cairn",
            use_cache=False,  # Disable cache for honest measurement
            include_infrastructure=False,
            scope="global",
        )
        latency_ms = (time.monotonic() - t0) * 1000

        total_latencies.append(latency_ms)

        # Compute precision at each K
        precisions = {}
        relevance_flags = []
        for k in k_values:
            precisions[f"P@{k}"] = precision_at_k(results, ground_truth, k)

        # Track per-result relevance for top-5
        for i, r in enumerate(results[:limit]):
            rel = is_relevant(r.content, ground_truth)
            relevance_flags.append(rel)

        entry = {
            "category": category,
            "query": query_text,
            "ground_truth": ground_truth,
            "precisions": precisions,
            "latency_ms": round(latency_ms, 1),
            "num_results": len(results),
            "relevance": relevance_flags,
        }
        per_query_results.append(entry)

        # Accumulate by category
        p5_key = f"P@{limit}"
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(precisions.get(p5_key, 0.0))

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------
    overall = {}
    for k in k_values:
        key = f"P@{k}"
        scores = [e["precisions"][key] for e in per_query_results]
        overall[key] = round(sum(scores) / len(scores), 4) if scores else 0.0

    category_avg = {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in category_scores.items()
    }

    summary = {
        "num_queries": len(TEST_QUERIES),
        "limit": limit,
        "overall": overall,
        "by_category": category_avg,
        "latency": {
            "mean_ms": round(sum(total_latencies) / len(total_latencies), 1),
            "min_ms": round(min(total_latencies), 1),
            "max_ms": round(max(total_latencies), 1),
            "p50_ms": round(sorted(total_latencies)[len(total_latencies) // 2], 1),
        },
        "queries": per_query_results,
    }

    return summary


def print_report(summary: Dict, verbose: bool = False) -> None:
    """Print a human-readable precision audit report."""

    print("=" * 72)
    print("  Cairn Precision@K Audit")
    print("=" * 72)
    print()
    print(f"  Queries: {summary['num_queries']}    Limit: {summary['limit']}")
    print()

    # Overall scores
    print("  OVERALL SCORES")
    print("  " + "-" * 40)
    for key, val in summary["overall"].items():
        bar = "#" * int(val * 30)
        print(f"  {key:>6s}:  {val:.4f}  |{bar:<30s}|")
    print()

    # By category
    print("  BY CATEGORY (P@{})".format(summary["limit"]))
    print("  " + "-" * 40)
    for cat, val in sorted(summary["by_category"].items()):
        bar = "#" * int(val * 30)
        print(f"  {cat:>14s}:  {val:.4f}  |{bar:<30s}|")
    print()

    # Latency
    lat = summary["latency"]
    print("  LATENCY")
    print("  " + "-" * 40)
    print(f"  Mean: {lat['mean_ms']:.1f}ms  Min: {lat['min_ms']:.1f}ms  "
          f"Max: {lat['max_ms']:.1f}ms  P50: {lat['p50_ms']:.1f}ms")
    print()

    # Per-query breakdown
    print("  PER-QUERY BREAKDOWN")
    print("  " + "-" * 70)
    print(f"  {'Category':>14s}  {'P@5':>6s}  {'Lat':>7s}  {'Rel':>9s}  Query")
    print("  " + "-" * 70)

    for entry in summary["queries"]:
        cat = entry["category"]
        p5 = entry["precisions"].get(f"P@{summary['limit']}", 0.0)
        lat_ms = entry["latency_ms"]
        # Show relevance as checkmarks/crosses
        rel_str = "".join("+" if r else "-" for r in entry["relevance"])
        query_short = entry["query"][:42]
        print(f"  {cat:>14s}  {p5:6.2f}  {lat_ms:6.1f}ms  [{rel_str:<5s}]  {query_short}")

        if verbose:
            print(f"  {'':>14s}  ground_truth: {entry['ground_truth']}")
            print()

    print()
    print("  Legend: [+] = relevant result, [-] = irrelevant result")
    print("=" * 72)

    # Diagnostic: flag weak queries
    weak = [e for e in summary["queries"]
            if e["precisions"].get(f"P@{summary['limit']}", 0) < 0.4]
    if weak:
        print()
        print(f"  WARNING: {len(weak)} queries below P@{summary['limit']} < 0.40:")
        for e in weak:
            print(f"    - [{e['category']}] {e['query'][:60]}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Cairn Precision@K Audit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show ground-truth keywords per query")
    parser.add_argument("--json", "-j", action="store_true", dest="json_output",
                        help="Output machine-readable JSON")
    parser.add_argument("--limit", "-k", type=int, default=5,
                        help="K value for primary Precision@K (default: 5)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to Cairn SQLite database (default: ~/.cairn/cairn.db)")
    args = parser.parse_args()

    # Instantiate store
    db_path = args.db if args.db else None
    store = SQLiteStore(db_path=db_path)

    try:
        # Quick sanity check: how many memories exist?
        count = store.node_count()
        if count == 0:
            print("ERROR: Store is empty (0 memories). Nothing to audit.", file=sys.stderr)
            sys.exit(1)

        if not args.json_output:
            print(f"  Store: {store.db_path}  ({count} memories)")
            print()

        summary = run_audit(
            store,
            verbose=args.verbose,
            json_output=args.json_output,
            limit=args.limit,
        )

        if args.json_output:
            # Strip verbose per-result data for cleaner JSON
            slim = {
                "num_queries": summary["num_queries"],
                "limit": summary["limit"],
                "overall": summary["overall"],
                "by_category": summary["by_category"],
                "latency": summary["latency"],
                "queries": [
                    {
                        "category": q["category"],
                        "query": q["query"],
                        "precisions": q["precisions"],
                        "latency_ms": q["latency_ms"],
                        "relevance": q["relevance"],
                    }
                    for q in summary["queries"]
                ],
            }
            print(json.dumps(slim, indent=2))
        else:
            print_report(summary, verbose=args.verbose)
    finally:
        store.close()


if __name__ == "__main__":
    main()
