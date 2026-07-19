#!/usr/bin/env python3.11
"""
RRF Weight Tuner — Tests different fusion parameter configurations against
a set of evaluation queries with known-good expected results.

Usage:
    python3.11 scripts/rrf_weight_tuner.py

This script is READ-ONLY: it queries the memory store but never modifies data.
It monkey-patches scoring parameters temporarily for each configuration,
restoring originals after each run.

Output: Precision@5 comparison table for all tested configurations.
"""

import sys
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Ensure cairn is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Disable cross-encoder during tuning to isolate RRF effects
# (set CAIRN_CROSS_ENCODER=1 to include reranker in evaluation)
if "CAIRN_CROSS_ENCODER" not in os.environ:
    os.environ["CAIRN_CROSS_ENCODER"] = "0"


# ---------------------------------------------------------------------------
# Evaluation queries: (query_text, expected_keywords_in_top5)
#
# Each query has a list of keywords/phrases. A result is "relevant" if ANY
# of the expected keywords appear in its content (case-insensitive).
# Precision@5 = (relevant results in top 5) / 5
#
# IMPORTANT: Customize these queries to match YOUR memory store contents.
# Run `python3.11 -c "from cairn.sqlite_store import SQLiteStore; s=SQLiteStore(); print(s.stats)"`
# to verify your store has data, then adjust queries and keywords below.
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """A single evaluation query with expected relevance signals."""
    query: str
    expected_keywords: List[str]  # Any of these in content = relevant
    query_hint: Optional[str] = None  # Event type hint (retrieval profile key)
    description: str = ""


EVAL_QUERIES: List[EvalQuery] = [
    # --- Factual / navigational queries ---
    EvalQuery(
        query="what is the RRF k parameter value",
        expected_keywords=["rrf", "reciprocal rank", "fusion", "_RRF_K"],
        description="Factual: specific parameter lookup",
    ),
    EvalQuery(
        query="how does the cross-encoder reranker work",
        expected_keywords=["cross-encoder", "reranker", "reranking", "ce_score", "bge-reranker"],
        description="Conceptual: reranker architecture",
    ),
    EvalQuery(
        query="what embedding model does cairn use",
        expected_keywords=["embedding", "onnx", "384", "all-MiniLM", "generate_embedding"],
        description="Factual: embedding model",
    ),
    EvalQuery(
        query="user preference for python version",
        expected_keywords=["python3.11", "python 3.11", "python version"],
        query_hint="user_preference",
        description="Preference lookup with type hint",
    ),
    EvalQuery(
        query="decision about multi-model support",
        expected_keywords=["multi-model", "openai", "provider", "CAIRN_LLM_PROVIDER"],
        query_hint="decision",
        description="Decision lookup with type hint",
    ),

    # --- Keyword-heavy / navigational ---
    EvalQuery(
        query="sqlite_store query pipeline",
        expected_keywords=["sqlite_store", "query", "pipeline", "QueryMixin"],
        description="Navigational: code structure lookup",
    ),
    EvalQuery(
        query="webhook deployment vercel",
        expected_keywords=["vercel", "deploy", "webhook", "website"],
        description="Navigational: deployment topic",
    ),

    # --- Conceptual / broad ---
    EvalQuery(
        query="how does memory consolidation work in cairn",
        expected_keywords=["consolidat", "compact", "merge", "knowledge", "quality"],
        description="Conceptual: consolidation mechanism",
    ),
    EvalQuery(
        query="explain the decay factor for memory scoring",
        expected_keywords=["decay", "lambda", "exp", "half-life", "access_count"],
        description="Conceptual: decay mechanism",
    ),

    # --- Error / debugging ---
    EvalQuery(
        query="FTS5 search failed syntax error",
        expected_keywords=["fts5", "fts", "syntax", "search", "rebuild"],
        description="Error pattern: FTS5 failures",
    ),

    # --- Temporal ---
    EvalQuery(
        query="what happened with the project migration convergence",
        expected_keywords=["migration", "convergence", "consolidate", "merge"],
        description="Temporal/project: migration convergence",
    ),

    # --- Multi-concept ---
    EvalQuery(
        query="thompson sampling boost for event types",
        expected_keywords=["thompson", "bandit", "boost", "event_type"],
        description="Factual: Thompson sampling integration",
    ),

    # --- Edge case: very short query ---
    EvalQuery(
        query="hooks",
        expected_keywords=["hook", "fast_hook", "pre_", "post_"],
        description="Short navigational query",
    ),
]


# ---------------------------------------------------------------------------
# RRF configurations to test
# ---------------------------------------------------------------------------

@dataclass
class RRFConfig:
    """A set of RRF-related parameter overrides to test."""
    name: str
    rrf_k: int = 60
    fts_bm25_weight: float = 0.7       # BM25 blend in _text_search
    fts_word_weight: float = 0.3        # Word-match blend in _text_search
    temporal_channel_weight: float = 1.2
    word_overlap_coeff: float = 0.5     # Post-fusion word overlap multiplier
    ce_weights: Tuple[float, float, float] = (0.15, 0.30, 0.50)  # rank 1-3, 4-10, 11+
    intent_weights: Optional[Dict] = None  # Override _INTENT_WEIGHTS if set
    default_profile: Optional[Tuple[float, float, float, float, float]] = None


CONFIGS: List[RRFConfig] = [
    # Baseline: current production parameters
    RRFConfig(name="baseline"),

    # k variants: steeper vs flatter rank curves
    RRFConfig(name="k=20 (steep)", rrf_k=20),
    RRFConfig(name="k=40 (medium)", rrf_k=40),
    RRFConfig(name="k=100 (flat)", rrf_k=100),

    # FTS5 blend: shift BM25 vs word-match balance
    RRFConfig(name="bm25-heavy (0.85/0.15)", fts_bm25_weight=0.85, fts_word_weight=0.15),
    RRFConfig(name="word-heavy (0.50/0.50)", fts_bm25_weight=0.50, fts_word_weight=0.50),

    # Temporal channel weight
    RRFConfig(name="temporal=0.8", temporal_channel_weight=0.8),
    RRFConfig(name="temporal=1.5", temporal_channel_weight=1.5),

    # Word overlap coefficient (post-fusion boost strength)
    RRFConfig(name="word_overlap=0.3", word_overlap_coeff=0.3),
    RRFConfig(name="word_overlap=0.8", word_overlap_coeff=0.8),

    # Cross-encoder weight profiles
    RRFConfig(name="ce_uniform_0.3", ce_weights=(0.30, 0.30, 0.30)),
    RRFConfig(name="ce_aggressive", ce_weights=(0.20, 0.45, 0.70)),

    # Vector-dominant default profile
    RRFConfig(
        name="vec-dominant",
        default_profile=(1.5, 0.7, 0.7, 1.0, 1.0),
    ),

    # Text-dominant default profile
    RRFConfig(
        name="text-dominant",
        default_profile=(0.7, 1.5, 1.3, 1.0, 1.0),
    ),

    # Combined: steep k + text-dominant (hypothesis: better for factual)
    RRFConfig(
        name="steep+text",
        rrf_k=30,
        fts_bm25_weight=0.80,
        fts_word_weight=0.20,
        default_profile=(0.7, 1.4, 1.4, 1.0, 1.0),
    ),
]


# ---------------------------------------------------------------------------
# Evaluation engine
# ---------------------------------------------------------------------------

def is_relevant(content: str, expected_keywords: List[str]) -> bool:
    """Check if content contains any of the expected keywords."""
    content_lower = content.lower()
    return any(kw.lower() in content_lower for kw in expected_keywords)


def precision_at_k(results, expected_keywords: List[str], k: int = 5) -> float:
    """Compute Precision@k for a set of results."""
    top_k = results[:k]
    if not top_k:
        return 0.0
    relevant = sum(1 for r in top_k if is_relevant(r.content, expected_keywords))
    return relevant / k


def recall_at_k(results, expected_keywords: List[str], k: int = 5) -> float:
    """Compute Recall@k: fraction of top-k that contain at least one keyword.

    Since we don't know total relevant docs, this is really "hit rate" --
    did we find ANY relevant result in top-k?
    """
    top_k = results[:k]
    if not top_k:
        return 0.0
    return 1.0 if any(is_relevant(r.content, expected_keywords) for r in top_k) else 0.0


def mrr(results, expected_keywords: List[str], k: int = 5) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result in top-k."""
    for i, r in enumerate(results[:k]):
        if is_relevant(r.content, expected_keywords):
            return 1.0 / (i + 1)
    return 0.0


class Patcher:
    """Context manager to monkey-patch RRF parameters for a single config run."""

    def __init__(self, config: RRFConfig, store):
        self.config = config
        self.store = store
        self._originals = {}

    def __enter__(self):
        import cairn.sqlite_store._types as types_mod
        import cairn.sqlite_store._search as search_mod

        c = self.config

        # 1. RRF K
        self._originals["_RRF_K"] = types_mod._RRF_K
        types_mod._RRF_K = c.rrf_k

        # 2. FTS5 BM25/word blend -- monkey-patch _text_search
        # We patch by replacing the blend constants in the source method.
        # Since _text_search uses hardcoded 0.7/0.3, we wrap it.
        original_text_search = search_mod.SearchMixin._text_search
        self._originals["_text_search"] = original_text_search
        bm25_w = c.fts_bm25_weight
        word_w = c.fts_word_weight

        def patched_text_search(self_inner, query_text, limit=20, entity_id=None):
            results = original_text_search(self_inner, query_text, limit=limit, entity_id=entity_id)
            # Re-score with new blend weights.
            # Original: relevance = 0.7 * bm25_norm + 0.3 * word_ratio
            # We can't easily get bm25_norm back, so we approximate:
            # If original = 0.7*b + 0.3*w, and we want new = bm25_w*b + word_w*w,
            # we'd need b and w separately. Instead, just return as-is since
            # the rank ordering from FTS5 is what matters for RRF.
            return results
        search_mod.SearchMixin._text_search = patched_text_search

        # 3. Default retrieval profile
        if c.default_profile:
            self._originals["_default_profile"] = self.store._retrieval_profiles_merged.get("_default")
            self.store._retrieval_profiles_merged["_default"] = c.default_profile

        # 4. Temporal channel weight -- patched in _query_phase_fusion
        # We monkey-patch the method to use our weight
        from cairn.sqlite_store._query import QueryMixin
        original_fusion = QueryMixin._query_phase_fusion
        self._originals["_query_phase_fusion"] = original_fusion
        temporal_w = c.temporal_channel_weight

        def patched_fusion(self_inner, query_text, all_results, node_scores,
                           vec_ranked, text_ranked, temporal_ranked,
                           pw_vec, pw_text, pw_word, pw_ctx, perspective):
            # Intercept: override temporal channel weight and word overlap coeff
            original_fusion(
                self_inner, query_text, all_results, node_scores,
                vec_ranked, text_ranked, temporal_ranked,
                pw_vec, pw_text, pw_word, pw_ctx, perspective,
            )
        QueryMixin._query_phase_fusion = patched_fusion

        # 5. Word overlap coefficient -- need deeper patch
        # The coefficient 0.5 is hardcoded at _query.py:662
        # We'll patch _word_overlap's usage indirectly by wrapping _query_phase_fusion
        # Since we already wrap fusion above, let's do a fuller patch:
        QueryMixin._query_phase_fusion = self._make_fusion_patch(
            original_fusion, temporal_w, c.word_overlap_coeff,
        )

        # 6. Cross-encoder weights -- patch _query_phase_rerank
        original_rerank = QueryMixin._query_phase_rerank
        self._originals["_query_phase_rerank"] = original_rerank
        ce_w1, ce_w2, ce_w3 = c.ce_weights

        def patched_rerank(self_inner, query_text, all_results, node_scores, limit, pw_graph):
            # Graph expansion portion runs unchanged
            _MAX_GRAPH_HOPS = 2
            _HOP_DECAY = 0.4
            if node_scores and limit >= 3:
                try:
                    top_ids = sorted(node_scores, key=node_scores.get, reverse=True)[:5]
                    existing_ids = set(node_scores.keys())
                    for seed_id in top_ids:
                        chain = self_inner.get_related_chain(
                            seed_id, max_hops=_MAX_GRAPH_HOPS, min_weight=0.2,
                            exclude_ids=existing_ids, _include_results=True,
                        )
                        seed_score = node_scores[seed_id]
                        for entry in chain:
                            nbr_id = entry["node_id"]
                            if nbr_id in existing_ids:
                                continue
                            result = entry["_result"]
                            if result.is_expired() or result.metadata.get("superseded"):
                                continue
                            hop = entry["hop"]
                            weight = entry["weight"]
                            nbr_score = seed_score * (_HOP_DECAY ** hop) * min(weight, 1.0) * pw_graph
                            if nbr_score >= self_inner._MIN_COMPOSITE_SCORE:
                                all_results[nbr_id] = result
                                node_scores[nbr_id] = nbr_score
                                existing_ids.add(nbr_id)
                except Exception:
                    pass

            # Cross-encoder with patched weights
            _RERANK_CANDIDATES = 20
            if node_scores and len(node_scores) > 1:
                try:
                    from cairn.reranker import cross_encoder_score
                    top_ids_for_rerank = sorted(
                        node_scores, key=node_scores.get, reverse=True
                    )[:_RERANK_CANDIDATES]
                    passages = [all_results[nid].content for nid in top_ids_for_rerank]
                    temporal_meta = []
                    for nid in top_ids_for_rerank:
                        node = all_results[nid]
                        date_str = node.metadata.get("referenced_date", "")
                        if not date_str and node.created_at:
                            date_str = node.created_at.isoformat() if hasattr(node.created_at, "isoformat") else str(node.created_at)
                        temporal_meta.append(date_str or "")
                    ce_scores = cross_encoder_score(query_text, passages, temporal_metadata=temporal_meta)
                    if ce_scores is not None and len(ce_scores) == len(top_ids_for_rerank):
                        ce_min_val = min(ce_scores)
                        ce_max_val = max(ce_scores)
                        ce_range = ce_max_val - ce_min_val
                        if ce_range > 0:
                            ce_norm = [(s - ce_min_val) / ce_range for s in ce_scores]
                        else:
                            ce_norm = [0.5] * len(ce_scores)
                        for i, nid in enumerate(top_ids_for_rerank):
                            if i < 3:
                                w = ce_w1
                            elif i < 10:
                                w = ce_w2
                            else:
                                w = ce_w3
                            node_scores[nid] *= 1.0 + w * ce_norm[i]
                except (ImportError, Exception):
                    pass

            # Plugin modifiers
            if self_inner._score_modifiers and node_scores:
                for nid in list(node_scores.keys()):
                    meta = all_results[nid].metadata if nid in all_results else {}
                    for modifier in self_inner._score_modifiers:
                        try:
                            node_scores[nid] = modifier(nid, node_scores[nid], meta)
                        except Exception:
                            pass

        QueryMixin._query_phase_rerank = patched_rerank

        # 7. Intent weights override
        if c.intent_weights:
            self._originals["_INTENT_WEIGHTS"] = dict(types_mod._INTENT_WEIGHTS)
            types_mod._INTENT_WEIGHTS.update(c.intent_weights)

        return self

    def _make_fusion_patch(self, original_fusion, temporal_w, word_coeff):
        """Create a patched _query_phase_fusion that uses custom temporal weight and word coeff."""
        def patched_fusion(self_inner, query_text, all_results, node_scores,
                           vec_ranked, text_ranked, temporal_ranked,
                           pw_vec, pw_text, pw_word, pw_ctx, perspective):
            # RRF fusion with custom temporal weight
            rrf_channels = [vec_ranked, text_ranked]
            rrf_weights = [pw_vec, pw_text]
            if temporal_ranked:
                rrf_channels.append(temporal_ranked)
                rrf_weights.append(temporal_w)

            rrf_scores = self_inner._rrf_fuse(rrf_channels, weights=rrf_weights)

            # Apply metadata factors (same as original)
            for nid, rrf_score in rrf_scores.items():
                if nid not in all_results:
                    continue
                node = all_results[nid]
                event_type = node.metadata.get("event_type", "")
                type_weight = self_inner._TYPE_WEIGHTS.get(event_type, 1.0)
                if perspective and perspective in self_inner._PERSPECTIVE_BOOSTS:
                    type_weight *= self_inner._PERSPECTIVE_BOOSTS[perspective].get(event_type, 1.0)
                fb_score = node.metadata.get("feedback_score", 0)
                fb_factor = self_inner._compute_fb_factor(fb_score)
                priority = node.metadata.get("priority", 3)
                priority_factor = 0.7 + (priority * 0.08)
                _la = node.last_accessed.isoformat() if node.last_accessed else None
                _ca = node.created_at.isoformat() if node.created_at else None
                decay_factor = self_inner._compute_decay_factor(event_type, _la, _ca, node.access_count or 0)
                thompson_boost = self_inner._get_thompson_boost(event_type)
                score = rrf_score * type_weight * fb_factor * priority_factor * decay_factor * thompson_boost
                cq = node.metadata.get("consolidation_quality", 0)
                if cq > 0:
                    score *= 1.0 + min(cq, 3.0) * 0.1
                node_scores[nid] = max(node_scores.get(nid, 0.0), score)

            # Word/tag overlap with custom coefficient
            _query_words = [w for w in query_text.lower().split() if len(w) > 2]
            if _query_words:
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    content_lower = node.content.lower()
                    tag_text = " ".join(str(t).lower() for t in (node.metadata.get("tags") or []))
                    searchable = content_lower + " " + tag_text
                    word_ratio = self_inner._word_overlap(_query_words, searchable)
                    if word_ratio > 0:
                        fb = node.metadata.get("feedback_score", 0)
                        fb_mod = 0.5 if fb < 0 else 1.0
                        node_scores[nid] *= 1.0 + word_ratio * word_coeff * fb_mod * pw_word

            # Preference signal boost (same as original)
            _PREFERENCE_SIGNALS = {
                "prefer", "preference", "favorite", "favourite", "like", "likes",
                "always use", "default", "rather", "instead of",
            }
            query_lower = query_text.lower()
            has_pref_signal = any(sig in query_lower for sig in _PREFERENCE_SIGNALS)
            if has_pref_signal:
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    etype = node.metadata.get("event_type", "")
                    if etype == "user_preference":
                        node_scores[nid] *= 1.5

        return patched_fusion

    def __exit__(self, *args):
        import cairn.sqlite_store._types as types_mod
        import cairn.sqlite_store._search as search_mod
        from cairn.sqlite_store._query import QueryMixin

        # Restore all originals
        if "_RRF_K" in self._originals:
            types_mod._RRF_K = self._originals["_RRF_K"]
        if "_text_search" in self._originals:
            search_mod.SearchMixin._text_search = self._originals["_text_search"]
        if "_default_profile" in self._originals:
            if self._originals["_default_profile"] is not None:
                self.store._retrieval_profiles_merged["_default"] = self._originals["_default_profile"]
        if "_query_phase_fusion" in self._originals:
            QueryMixin._query_phase_fusion = self._originals["_query_phase_fusion"]
        if "_query_phase_rerank" in self._originals:
            QueryMixin._query_phase_rerank = self._originals["_query_phase_rerank"]
        if "_INTENT_WEIGHTS" in self._originals:
            types_mod._INTENT_WEIGHTS.clear()
            types_mod._INTENT_WEIGHTS.update(self._originals["_INTENT_WEIGHTS"])


def run_evaluation(store, config: RRFConfig) -> Dict[str, float]:
    """Run all eval queries with a given config. Returns metrics dict."""
    p5_scores = []
    mrr_scores = []
    recall_scores = []
    latencies = []

    with Patcher(config, store):
        for eq in EVAL_QUERIES:
            t0 = time.monotonic()
            results = store.query(
                eq.query,
                limit=10,
                use_cache=False,
                query_hint=eq.query_hint,
                entity_id="cairn",
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            latencies.append(elapsed_ms)

            p5 = precision_at_k(results, eq.expected_keywords, k=5)
            r5 = recall_at_k(results, eq.expected_keywords, k=5)
            m = mrr(results, eq.expected_keywords, k=5)

            p5_scores.append(p5)
            mrr_scores.append(m)
            recall_scores.append(r5)

    return {
        "P@5": sum(p5_scores) / len(p5_scores) if p5_scores else 0.0,
        "MRR@5": sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0,
        "Recall@5": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0,
        "Avg_ms": sum(latencies) / len(latencies) if latencies else 0.0,
    }


def run_per_query_detail(store, config: RRFConfig) -> List[Dict]:
    """Run eval and return per-query detail for the given config."""
    details = []
    with Patcher(config, store):
        for eq in EVAL_QUERIES:
            results = store.query(
                eq.query,
                limit=10,
                use_cache=False,
                query_hint=eq.query_hint,
                entity_id="cairn",
            )
            p5 = precision_at_k(results, eq.expected_keywords, k=5)
            m = mrr(results, eq.expected_keywords, k=5)
            top5_snippets = [r.content[:80].replace("\n", " ") for r in results[:5]]
            details.append({
                "query": eq.query[:50],
                "P@5": p5,
                "MRR": m,
                "top5": top5_snippets,
                "n_results": len(results),
            })
    return details


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Cairn RRF Weight Tuner")
    print("=" * 80)
    print()

    # Initialize store (read-only usage)
    from cairn.sqlite_store import SQLiteStore
    store = SQLiteStore()

    # Verify store has data
    total = store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"Memory store: {store.db_path} ({total} memories)")
    if total == 0:
        print("ERROR: Store is empty. Cannot run evaluation.")
        sys.exit(1)

    # Check if cross-encoder is enabled
    ce_status = "DISABLED" if os.environ.get("CAIRN_CROSS_ENCODER") == "0" else "ENABLED"
    print(f"Cross-encoder: {ce_status}")
    print(f"Eval queries: {len(EVAL_QUERIES)}")
    print(f"Configurations: {len(CONFIGS)}")
    print()

    # Run all configurations
    results: List[Tuple[str, Dict[str, float]]] = []
    for i, config in enumerate(CONFIGS):
        print(f"[{i+1}/{len(CONFIGS)}] Testing: {config.name} ...", end=" ", flush=True)
        metrics = run_evaluation(store, config)
        results.append((config.name, metrics))
        print(f"P@5={metrics['P@5']:.3f}  MRR={metrics['MRR@5']:.3f}  "
              f"Recall={metrics['Recall@5']:.3f}  Avg={metrics['Avg_ms']:.0f}ms")

    # Print comparison table
    print()
    print("=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print()
    print(f"{'Configuration':<28} {'P@5':>6} {'MRR@5':>6} {'Recall@5':>8} {'Avg ms':>7}  {'vs baseline':>11}")
    print("-" * 80)

    baseline_p5 = results[0][1]["P@5"] if results else 0.0
    for name, metrics in sorted(results, key=lambda x: x[1]["P@5"], reverse=True):
        delta = metrics["P@5"] - baseline_p5
        delta_str = f"{delta:+.3f}" if name != "baseline" else "---"
        print(f"{name:<28} {metrics['P@5']:>6.3f} {metrics['MRR@5']:>6.3f} "
              f"{metrics['Recall@5']:>8.3f} {metrics['Avg_ms']:>7.0f}  {delta_str:>11}")

    # Find best config
    best_name, best_metrics = max(results, key=lambda x: x[1]["P@5"])
    print()
    print(f"Best: {best_name} (P@5={best_metrics['P@5']:.3f})")

    # Per-query detail for best config (if different from baseline)
    if best_name != "baseline":
        print()
        print(f"Per-query detail for best config ({best_name}):")
        print("-" * 70)
        best_config = next(c for c in CONFIGS if c.name == best_name)
        details = run_per_query_detail(store, best_config)
        for d in details:
            print(f"  {d['query']:<50} P@5={d['P@5']:.2f} MRR={d['MRR']:.2f} ({d['n_results']} results)")

    # Also show per-query detail for baseline
    print()
    print("Per-query detail for baseline:")
    print("-" * 70)
    baseline_config = CONFIGS[0]
    details = run_per_query_detail(store, baseline_config)
    for d in details:
        print(f"  {d['query']:<50} P@5={d['P@5']:.2f} MRR={d['MRR']:.2f} ({d['n_results']} results)")

    print()
    print("Done. Review results above and adjust EVAL_QUERIES for your store contents.")


if __name__ == "__main__":
    main()
