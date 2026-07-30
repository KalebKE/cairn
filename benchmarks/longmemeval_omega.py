#!/usr/bin/env python3
"""LongMemEval benchmark with upstream omega-memory as the retrieval backend.

Protocol is the same faithful port of MemPalace's ``longmemeval_bench.py``
raw mode used by ``longmemeval_cairn.py``, so all three systems (MemPalace,
omega-memory, Cairn) are directly comparable:

- one document per haystack session (all user turns joined with newlines)
- fresh store per question, single query = the benchmark question
- Recall-any / Recall-all / NDCG at k in {1, 3, 5, 10, 30, 50}
- documents the retriever does not return are appended to the ranking in
  corpus order (same fill rule as MemPalace's harness)

The retriever is omega-memory v1.5.5 (the project Cairn was forked from) on
its shipped defaults: bge-small-en-v1.5 ONNX embeddings, its own fusion and
reranking. Run this under a venv with omega-memory installed, NOT the cairn
venv:

    python3 -m venv /tmp/omega-venv
    /tmp/omega-venv/bin/pip install -e <omega-memory checkout>
    /tmp/omega-venv/bin/python benchmarks/longmemeval_omega.py \
        /tmp/longmemeval-data/longmemeval_s_cleaned.json --limit 20

Hermetic like the Cairn run: provider keys are stripped from the environment
before omega imports, so LLM query expansion no-ops and a keyless clone
reproduces the same digits. OMEGA_HOME is pointed at a temp dir so the run
cannot touch a real ~/.omega store. Ingestion uses skip_inference=True for
the same reason as the Cairn harness: write-time dedup would merge distinct
haystack sessions and corrupt the answer key.
"""

import os

# Hermetic + isolated: must be set before importing omega.
os.environ.setdefault("OMEGA_MAX_CONTENT_SIZE", "2000000")
for _key in (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GROQ_API_KEY",
    "DEEPINFRA_API_KEY",
    "TOGETHER_API_KEY",
    "FIREWORKS_API_KEY",
    "OPENROUTER_API_KEY",
    "XAI_API_KEY",
    "MISTRAL_API_KEY",
    "OMEGA_LLM_API_KEY",
):
    os.environ.pop(_key, None)

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

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


# ── omega-memory retriever ──────────────────────────────────────────────────


def build_corpus(entry):
    """Session granularity: one doc per haystack session = joined user turns."""
    corpus, corpus_ids = [], []
    for session, sess_id in zip(entry["haystack_sessions"], entry["haystack_session_ids"]):
        user_turns = [t["content"] for t in session if t["role"] == "user"]
        if user_turns:
            corpus.append("\n".join(user_turns))
            corpus_ids.append(sess_id)
    return corpus, corpus_ids


def omega_retrieve(entry, tmpdir, n_results=50):
    from omega.sqlite_store import SQLiteStore

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


def run(data_file, limit=0, tag="", out_dir=None):
    with open(data_file) as f:
        data = json.load(f)
    if limit > 0:
        data = data[:limit]

    print(f"\n{'=' * 62}")
    print("  omega-memory × LongMemEval  (MemPalace raw-mode protocol)")
    print(f"{'=' * 62}")
    print(f"  Data:      {Path(data_file).name}")
    print(f"  Questions: {len(data)}")
    print("  Backend:   omega-memory shipped defaults, keyless (hermetic)")
    print(f"{'─' * 62}\n")

    metrics = {f"recall_any@{k}": [] for k in KS}
    metrics.update({f"recall_all@{k}": [] for k in KS})
    metrics.update({f"ndcg_any@{k}": [] for k in KS})
    per_type = defaultdict(lambda: defaultdict(list))
    results_log = []
    t_start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="omega-lme-") as tmpdir:
        os.environ["OMEGA_HOME"] = str(Path(tmpdir) / "omega-home")
        for i, entry in enumerate(data):
            qid = entry["question_id"]
            qtype = entry["question_type"]
            answer_sids = set(entry["answer_session_ids"])

            rankings, corpus, corpus_ids = omega_retrieve(entry, tmpdir)
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
    out_path = out_dir / f"omega{suffix}_{stamp}.json"
    summary = {
        "backend": "omega-memory 1.5.5, shipped defaults, keyless",
        "protocol": "MemPalace longmemeval_bench.py raw mode, session granularity",
        "questions": n,
        "elapsed_min": round(elapsed / 60, 1),
        "metrics": {
            k: round(sum(v) / n, 4) for k, v in metrics.items() if v
        },
        "per_type_recall_any_at_5": {
            qtype: round(sum(m["recall_any@5"]) / len(m["recall_any@5"]), 4)
            for qtype, m in per_type.items()
        },
        "results": results_log,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    sys.exit(run(args.data_file, limit=args.limit, tag=args.tag, out_dir=args.out_dir))
