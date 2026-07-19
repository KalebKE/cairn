#!/usr/bin/env python3.11
"""Latency profiler for Cairn query pipeline.

Monkey-patches the query pipeline to add per-phase timing,
then runs representative queries and outputs a breakdown.
"""

import os
import sys
import time
import functools

# Ensure cairn is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Phase timing instrumentation ──

_phase_timings: dict[str, list[float]] = {}


def _wrap_phase(cls, method_name, phase_label):
    """Wrap a phase method to record its execution time."""
    original = getattr(cls, method_name)

    @functools.wraps(original)
    def timed(self, *args, **kwargs):
        t0 = time.monotonic()
        result = original(self, *args, **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _phase_timings.setdefault(phase_label, []).append(elapsed_ms)
        return result

    setattr(cls, method_name, timed)


def _wrap_function(module, func_name, phase_label):
    """Wrap a module-level function to record its execution time."""
    original = getattr(module, func_name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        t0 = time.monotonic()
        result = original(*args, **kwargs)
        elapsed_ms = (time.monotonic() - t0) * 1000
        _phase_timings.setdefault(phase_label, []).append(elapsed_ms)
        return result

    setattr(module, func_name, timed)


def instrument():
    """Apply all timing instrumentation."""
    from cairn.sqlite_store._query import QueryMixin

    phases = [
        ("_query_phase_vec", "1_vec_search"),
        ("_query_phase_fts", "2_fts_search"),
        ("_query_phase_expand", "2.7_expansion"),
        ("_query_phase_fusion", "3_fusion"),
        ("_query_phase_filter", "4_filter"),
        ("_query_phase_boost", "5_boost"),
        ("_query_phase_rerank", "6_rerank"),
        ("_query_phase_assemble", "7_assemble"),
    ]
    for method_name, label in phases:
        if hasattr(QueryMixin, method_name):
            _wrap_phase(QueryMixin, method_name, label)

    # Instrument embedding generation
    import cairn.embedding as emb_mod
    _wrap_function(emb_mod, "generate_embedding", "embed_gen")

    # Instrument LLM call for query expansion
    import cairn.llm as llm_mod
    _wrap_function(llm_mod, "llm_complete", "llm_complete")

    # Instrument cross-encoder scoring
    import cairn.reranker as reranker_mod
    _wrap_function(reranker_mod, "cross_encoder_score", "cross_encoder")

    # Instrument query expansion function
    import cairn.query_expansion as qe_mod
    _wrap_function(qe_mod, "expand_query", "qe_expand_query")

    # Instrument adaptive retry
    if hasattr(QueryMixin, "_adaptive_retry_query"):
        _wrap_phase(QueryMixin, "_adaptive_retry_query", "7.5_adaptive_retry")

    # Instrument decomposition check
    if hasattr(QueryMixin, "_decompose_query"):
        _wrap_phase(QueryMixin, "_decompose_query", "0_decompose")


def run_queries(store, queries):
    """Run queries and collect per-phase timings."""
    results = []
    for label, query_text in queries:
        _phase_timings.clear()
        t0 = time.monotonic()
        res = store.query(
            query_text,
            limit=10,
            use_cache=False,
            expand_query=True,
            entity_id="cairn",
        )
        total_ms = (time.monotonic() - t0) * 1000
        # Snapshot timings
        timings = {k: list(v) for k, v in _phase_timings.items()}
        results.append({
            "label": label,
            "query": query_text,
            "total_ms": total_ms,
            "n_results": len(res),
            "phases": timings,
        })
    return results


def print_report(results):
    """Print formatted latency report."""
    print("\n" + "=" * 80)
    print("Cairn QUERY PIPELINE LATENCY PROFILE")
    print("=" * 80)

    for r in results:
        print(f"\n{'─' * 70}")
        print(f"Query: {r['query'][:60]}")
        print(f"Label: {r['label']}  |  Total: {r['total_ms']:.1f}ms  |  Results: {r['n_results']}")
        print(f"{'─' * 70}")

        # Sort phases by their label prefix
        sorted_phases = sorted(r["phases"].items(), key=lambda x: x[0])
        accounted = 0.0
        for phase_name, durations in sorted_phases:
            total_phase = sum(durations)
            count = len(durations)
            pct = (total_phase / r["total_ms"] * 100) if r["total_ms"] > 0 else 0
            avg = total_phase / count if count else 0
            accounted += total_phase
            bar = "#" * int(pct / 2)
            print(f"  {phase_name:25s} {total_phase:8.1f}ms ({count:2d}x, avg {avg:6.1f}ms) [{pct:5.1f}%] {bar}")

        unaccounted = r["total_ms"] - accounted
        if unaccounted > 1:
            pct = (unaccounted / r["total_ms"] * 100) if r["total_ms"] > 0 else 0
            print(f"  {'(unaccounted)':25s} {unaccounted:8.1f}ms                       [{pct:5.1f}%]")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")

    totals = [r["total_ms"] for r in results]
    print(f"  Mean total latency:  {sum(totals)/len(totals):.1f}ms")
    print(f"  Max total latency:   {max(totals):.1f}ms")
    print(f"  Min total latency:   {min(totals):.1f}ms")

    # Aggregate phase totals across all queries
    phase_totals: dict[str, float] = {}
    phase_counts: dict[str, int] = {}
    for r in results:
        for phase, durations in r["phases"].items():
            phase_totals[phase] = phase_totals.get(phase, 0) + sum(durations)
            phase_counts[phase] = phase_counts.get(phase, 0) + len(durations)

    grand_total = sum(totals)
    print(f"\n  Aggregate phase breakdown (across all {len(results)} queries):")
    for phase, total in sorted(phase_totals.items(), key=lambda x: -x[1]):
        pct = (total / grand_total * 100) if grand_total > 0 else 0
        print(f"    {phase:25s} {total:8.1f}ms  [{pct:5.1f}%]  (called {phase_counts[phase]}x)")


def main():
    # Instrument before importing the store
    instrument()

    from cairn.sqlite_store import SQLiteStore

    # Find the user's actual store
    store_path = os.path.expanduser("~/.cairn/cairn.db")
    if not os.path.exists(store_path):
        print(f"Store not found at {store_path}")
        sys.exit(1)

    print(f"Opening store: {store_path}")
    store = SQLiteStore(store_path)

    # Representative queries (mix of expected fast and slow)
    queries = [
        ("navigational", "what is the Cairn version number"),
        ("factual", "project migration convergence status"),
        ("conceptual", "how does the memory system handle conflicts between agents"),
        ("vague", "what happened recently"),
        ("keyword", "pyproject.toml license field"),
    ]

    # Warmup: pre-load models (don't count cold start)
    print("Warming up models...")
    from cairn.embedding import preload_embedding_model
    preload_embedding_model()
    from cairn.reranker import preload_reranker_model
    preload_reranker_model()
    _phase_timings.clear()
    print("Models loaded. Running profiling queries...\n")

    results = run_queries(store, queries)
    print_report(results)

    store.close()


if __name__ == "__main__":
    main()
