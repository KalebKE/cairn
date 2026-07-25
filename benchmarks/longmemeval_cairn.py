#!/usr/bin/env python3
"""LongMemEval benchmark with Cairn as the retrieval backend.

Protocol is a faithful port of MemPalace's ``benchmarks/longmemeval_bench.py``
raw mode (https://github.com/MemPalace/mempalace) so numbers are directly
comparable:

- one document per haystack session (all user turns joined with newlines)
- fresh index per question, single query = the benchmark question
- Recall-any / Recall-all / NDCG at k in {1, 3, 5, 10, 30, 50}
- documents the retriever does not return are appended to the ranking in
  corpus order (same fill rule as MemPalace's harness)

The retriever here is Cairn's full production query pipeline on a fresh
``SQLiteStore`` per question: local bge-small embeddings + FTS5/BM25 fused
with RRF, metadata weighting, and the confidence-gated cross-encoder rerank
(controlled by CAIRN_CE_MODE, default hybrid).

Ingestion uses ``store(..., skip_inference=True)``: embedding-level dedup and
contradiction supersession are Cairn features for accreting knowledge, but on
a benchmark corpus they would merge distinct haystack sessions and corrupt
the answer key.

Usage:
    python benchmarks/longmemeval_cairn.py /tmp/longmemeval-data/longmemeval_s_cleaned.json --limit 20
    python benchmarks/longmemeval_cairn.py /tmp/longmemeval-data/longmemeval_s_cleaned.json
    CAIRN_CE_MODE=off python benchmarks/longmemeval_cairn.py ... --tag no-ce

Dataset:
    curl -fsSL -o /tmp/longmemeval-data/longmemeval_s_cleaned.json \
      https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
"""

import os

# Must be set before importing cairn: read at class-definition / module time.
os.environ.setdefault("CAIRN_MAX_CONTENT_SIZE", "2000000")
os.environ.setdefault("CAIRN_QUERY_LOG", "0")
# Hermetic by default: LLM query expansion samples a cloud model at
# temperature 0.3 inside query(), so with a provider key in the shell the
# SAME benchmark run gives different rankings run-to-run (±1-3 questions
# per 30 measured). Pin it off so results are reproducible and keyless
# reproductions run the same pipeline. Explicitly set
# CAIRN_QUERY_EXPANSION=1 to A/B expansion.
os.environ.setdefault("CAIRN_QUERY_EXPANSION", "0")

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from cairn.sqlite_store import SQLiteStore

KS = [1, 3, 5, 10, 30, 50]


# ── Metrics (identical to MemPalace's harness) ──────────────────────────────


def dcg(relevances, k):
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg(rankings, correct_ids, corpus_ids, k):
    relevances = [1.0 if corpus_ids[idx] in correct_ids else 0.0 for idx in rankings[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(relevances, k) / idcg if idcg else 0.0


def evaluate_retrieval(rankings, correct_ids, corpus_ids, k):
    top_k_ids = set(corpus_ids[idx] for idx in rankings[:k])
    recall_any = float(any(cid in top_k_ids for cid in correct_ids))
    recall_all = float(all(cid in top_k_ids for cid in correct_ids))
    return recall_any, recall_all, ndcg(rankings, correct_ids, corpus_ids, k)


# ── Cairn retriever ─────────────────────────────────────────────────────────


def build_corpus(entry):
    """Session granularity: one doc per haystack session = joined user turns."""
    corpus, corpus_ids = [], []
    for session, sess_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if user_turns:
            corpus.append("\n".join(user_turns))
            corpus_ids.append(sess_id)
    return corpus, corpus_ids


def cairn_retrieve(entry, tmpdir, n_results=50, relaxed=False):
    corpus, corpus_ids = build_corpus(entry)
    if not corpus:
        return [], corpus, corpus_ids

    db_path = Path(tmpdir) / f"{entry['question_id']}.db"
    store = SQLiteStore(db_path=db_path)
    try:
        node_to_idx = {}
        for i, doc in enumerate(corpus):
            node_id = store.store(
                content=doc,
                session_id="lme",
                metadata={"event_type": "memory", "corpus_id": corpus_ids[i]},
                skip_inference=True,
            )
            node_to_idx[node_id] = i

        results = store.query(
            entry["question"],
            limit=min(n_results, len(corpus)),
            use_cache=False,
            # --relaxed: run with the production adaptive-retry thresholds.
            # Cairn's abstention floors are a feature in agent use (an empty
            # answer beats a plausible wrong one) but under this benchmark's
            # fill rule every unreturned doc is ranked by corpus order, so
            # truncation directly costs deep recall.
            _is_retry=relaxed,
        )
        ranked = [node_to_idx[r.id] for r in results if r.id in node_to_idx]
    finally:
        store.close()
        for p in Path(tmpdir).glob(f"{entry['question_id']}.db*"):
            p.unlink(missing_ok=True)

    # Fill rule: unreturned docs appended in corpus order (MemPalace-identical).
    seen = set(ranked)
    ranked.extend(i for i in range(len(corpus)) if i not in seen)
    return ranked, corpus, corpus_ids


# ── Benchmark loop ──────────────────────────────────────────────────────────


def run(data_file, limit=0, tag="", out_dir=None, relaxed=False):
    with open(data_file) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]

    ce_mode = os.environ.get("CAIRN_CE_MODE", "hybrid (default)")
    expansion_on = os.environ.get("CAIRN_QUERY_EXPANSION", "0") != "0"
    print(f"\n{'=' * 62}")
    print("  Cairn × LongMemEval  (MemPalace raw-mode protocol, session)")
    print(f"{'=' * 62}")
    print(f"  Data:      {Path(data_file).name}")
    print(f"  Questions: {len(data)}")
    print(f"  CE mode:   {ce_mode}")
    print(f"  Abstain:   {'relaxed (adaptive-retry thresholds)' if relaxed else 'default'}")
    print(f"  Expansion: {'ON (LLM, non-hermetic!)' if expansion_on else 'off (hermetic)'}")
    print(f"{'─' * 62}\n")

    metrics = {f"recall_any@{k}": [] for k in KS}
    metrics.update({f"recall_all@{k}": [] for k in KS})
    metrics.update({f"ndcg_any@{k}": [] for k in KS})
    per_type = defaultdict(lambda: defaultdict(list))
    results_log = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="cairn-lme-") as tmpdir:
        for i, entry in enumerate(data):
            qid = entry["question_id"]
            qtype = entry["question_type"]
            answer_sids = set(entry["answer_session_ids"])

            rankings, corpus, corpus_ids = cairn_retrieve(entry, tmpdir, relaxed=relaxed)
            if not rankings:
                print(f"  [{i + 1:4}/{len(data)}] {qid[:32]:32} SKIP (empty corpus)")
                continue
            for k in KS:
                r_any, r_all, nd = evaluate_retrieval(rankings, answer_sids, corpus_ids, k)
                metrics[f"recall_any@{k}"].append(r_any)
                metrics[f"recall_all@{k}"].append(r_all)
                metrics[f"ndcg_any@{k}"].append(nd)
                per_type[qtype][f"recall_any@{k}"].append(r_any)

            r5 = metrics["recall_any@5"][-1]
            run_r5 = sum(metrics["recall_any@5"]) / len(metrics["recall_any@5"])
            mark = "✓" if r5 else "✗"
            print(
                f"  [{i + 1:4}/{len(data)}] {qid[:32]:32} {mark}  "
                f"R@5={run_r5:.3f}  ({qtype})"
            )
            results_log.append(
                {
                    "question_id": qid,
                    "question_type": qtype,
                    "recall_any@5": r5,
                    "recall_any@10": metrics["recall_any@10"][-1],
                    "ndcg@10": metrics["ndcg_any@10"][-1],
                    "top10_ids": [corpus_ids[idx] for idx in rankings[:10]],
                    "answer_ids": sorted(answer_sids),
                }
            )

    elapsed = time.monotonic() - t_start
    n = len(metrics["recall_any@5"])
    print(f"\n{'=' * 62}")
    print(f"  RESULTS — {n} questions in {elapsed / 60:.1f} min")
    print(f"{'=' * 62}")
    for k in KS:
        ra = sum(metrics[f"recall_any@{k}"]) / n
        rall = sum(metrics[f"recall_all@{k}"]) / n
        nd = sum(metrics[f"ndcg_any@{k}"]) / n
        print(f"  Recall@{k:<3} any={ra:.3f}  all={rall:.3f}  NDCG={nd:.3f}")
    print("\n  By question type (Recall_any@5):")
    for qtype, m in sorted(per_type.items()):
        vals = m["recall_any@5"]
        print(f"    {qtype:<28} {sum(vals) / len(vals):.3f}  (n={len(vals)})")

    out_dir = Path(out_dir or Path.home() / ".cairn" / "eval" / "longmemeval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"_{tag}" if tag else ""
    out_file = out_dir / f"results_cairn_session_top10{suffix}_{stamp}.json"
    # Model provenance — resolved AFTER the run so lazy loads are reflected.
    try:
        from cairn.embedding import get_embedding_model_info
        from cairn.reranker import _RERANKER_MODEL_NAME
        from cairn.sqlite_store import EMBEDDING_DIM

        _mi = get_embedding_model_info()
        provenance = {
            "embedding_model": _mi.get("model_name"),
            "embedding_backend": _mi.get("backend"),
            "embedding_dim": EMBEDDING_DIM,
            "reranker_model": _RERANKER_MODEL_NAME,
        }
    except Exception:
        provenance = {}

    with open(out_file, "w") as f:
        json.dump(
            {
                "harness": "longmemeval_cairn",
                "protocol": "mempalace-raw-session",
                "ce_mode": ce_mode,
                "query_expansion": expansion_on,
                "relaxed_abstention": relaxed,
                "provenance": provenance,
                "n": n,
                "elapsed_s": round(elapsed, 1),
                "summary": {
                    k2: round(sum(v) / n, 4) for k2, v in metrics.items() if v
                },
                "per_type_recall_any_at_5": {
                    t: round(sum(m["recall_any@5"]) / len(m["recall_any@5"]), 4)
                    for t, m in per_type.items()
                },
                "results": results_log,
            },
            f,
            indent=1,
        )
    if provenance:
        print(
            f"\n  Models: embed={provenance.get('embedding_model')} "
            f"({provenance.get('embedding_dim')}d, {provenance.get('embedding_backend')}) "
            f"rerank={provenance.get('reranker_model')}"
        )
    print(f"\n  Results → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data_file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="suffix for the results filename")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--relaxed",
        action="store_true",
        help="query with adaptive-retry (relaxed abstention) thresholds",
    )
    args = ap.parse_args()
    if not Path(args.data_file).exists():
        print(f"ERROR: {args.data_file} not found — see module docstring for download.")
        sys.exit(1)
    run(args.data_file, limit=args.limit, tag=args.tag, out_dir=args.out_dir, relaxed=args.relaxed)
