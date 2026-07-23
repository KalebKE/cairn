#!/usr/bin/env python3
"""Phase 3: reranker × margin sweep on the judged probes.

One OS process per (embedder, reranker) pair — reranker model selection is
module-global. Within the process, CAIRN_CE_MARGIN is call-time, so all
margins share one rebuilt snapshot (cheap, and identical corpus per margin).

Usage:
    python scripts/reranker_sweep.py gte msmarco
    python scripts/reranker_sweep.py gte bge-int8
    python scripts/reranker_sweep.py baseline msmarco
    python scripts/reranker_sweep.py baseline bge-int8
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODELS = Path(os.path.expanduser("~/.cache/cairn/models"))

EMBEDDERS = {
    "baseline": {"dir": None, "dim": 384},
    "gte": {"dir": MODELS / "gte-modernbert-base-onnx", "dim": 768},
}
RERANKERS = {
    "msmarco": {"env": {"CAIRN_RERANKER_MODEL": "ms-marco-MiniLM-L-6-v2"}},
    "bge-int8": {
        "env": {
            "CAIRN_RERANKER_MODEL": "bge-reranker-v2-m3",
            # Staged OUTSIDE the registry default dir so live servers never
            # auto-flip; this env points the loader at the quarantined copy.
            "CAIRN_CROSS_ENCODER_DIR": str(MODELS / "sweep-bge-reranker-v2-m3-int8"),
        }
    },
}
MARGINS = ["0.05", "0.10", "0.15", "0.20"]

DEFAULT_PROBES = os.path.expanduser("~/.cairn/eval/probes-v2-20260721-final.json")
LIVE_DB = os.path.expanduser("~/.cairn/cairn.db")
OUT_DIR = Path(os.path.expanduser("~/.cairn/eval/model-sweep"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("embedder", choices=sorted(EMBEDDERS))
    ap.add_argument("reranker", choices=sorted(RERANKERS))
    ap.add_argument("--probes", default=DEFAULT_PROBES)
    args = ap.parse_args()
    emb = EMBEDDERS[args.embedder]
    rr = RERANKERS[args.reranker]

    # ---- env BEFORE any cairn import -------------------------------------
    if emb["dir"] is not None:
        os.environ["CAIRN_ONNX_MODEL_DIR"] = str(emb["dir"])
    os.environ["CAIRN_EMBEDDING_DIM"] = str(emb["dim"])
    os.environ.update(rr["env"])
    os.environ["CAIRN_QUERY_LOG"] = "0"

    # ---- fail-fast: embedder AND reranker must actually load --------------
    from cairn.embedding import generate_embedding, get_active_backend

    e = generate_embedding("reranker sweep fail-fast probe")
    if get_active_backend() != "onnx" or len(e) != emb["dim"]:
        print(f"FATAL: embedder failed (backend={get_active_backend()}, dim={len(e)})")
        sys.exit(2)

    from cairn.reranker import _RERANKER_MODEL_NAME, cross_encoder_score

    t0 = time.monotonic()
    scores = cross_encoder_score("what changed in the sync layer?",
                                 ["The sync retry policy was rewritten.",
                                  "Lunch options near the office."] * 10)
    rerank_ms = (time.monotonic() - t0) * 1000
    if not scores or len(scores) != 20:
        print(f"FATAL: reranker {_RERANKER_MODEL_NAME} did not score (got {scores!r:.80})")
        sys.exit(2)
    if scores[0] <= scores[1]:
        print("FATAL: reranker scored the irrelevant passage above the relevant one")
        sys.exit(2)
    print(f"[{args.embedder}×{args.reranker}] reranker={_RERANKER_MODEL_NAME} "
          f"20-pair score={rerank_ms:.0f}ms (load included)")

    # ---- one snapshot, four margins ---------------------------------------
    from cairn.evaluation.retrieval_eval import rebuild_snapshot, run_evaluation_v2

    snap = rebuild_snapshot(LIVE_DB)
    results = {}
    for m in MARGINS:
        report = run_evaluation_v2(
            args.probes, db_path=snap, variant=f"{args.embedder}×{args.reranker}@m{m}",
            env={"CAIRN_CE_MARGIN": m, "CAIRN_CE_MODE": "hybrid"},
        )
        results[m] = {
            "mrr": round(report.mrr, 4),
            "ndcg": round(report.ndcg_at_k, 4),
            "hit_rate": round(report.hit_rate, 4),
            "duration_s": report.duration_seconds,
        }
        print(f"  margin {m}: MRR={report.mrr:.4f} nDCG={report.ndcg_at_k:.4f} "
              f"hit={report.hit_rate:.3f} ({report.duration_seconds:.0f}s)")

    out = OUT_DIR / f"reranker_{args.embedder}_{args.reranker}.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {"embedder": args.embedder, "reranker": _RERANKER_MODEL_NAME,
             "rerank_20pair_ms_cold": round(rerank_ms, 0), "margins": results},
            f, indent=1,
        )
    print(f"[{args.embedder}×{args.reranker}] DONE → {out}")


if __name__ == "__main__":
    main()
