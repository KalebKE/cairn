#!/usr/bin/env python3
"""Run one model-benchmark arm end to end, in its own process.

Model selection is module-global in cairn (resolved at import/first use), so
each arm MUST be a fresh OS process. This driver sets the arm's environment
BEFORE importing cairn, fail-fast-verifies the model actually loaded (a hash
fallback would produce plausible-garbage numbers), then runs:

  1. LongMemEval full-500 (benchmarks/longmemeval_cairn.py protocol)
  2. Judged frozen probes against a rebuilt snapshot of the live store
     (re-embedded with the arm's model at the arm's dimension)
  3. Latency microbench (run this arm exclusively — no parallel arms)

Usage:
    python scripts/sweep_arm.py baseline            # current stack, re-anchored
    python scripts/sweep_arm.py arctic
    python scripts/sweep_arm.py gte
    python scripts/sweep_arm.py gemma768
    python scripts/sweep_arm.py gemma512
    python scripts/sweep_arm.py <arm> --skip-lme --skip-probes --skip-latency
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

MODELS = Path(os.path.expanduser("~/.cache/cairn/models"))

ARMS = {
    "baseline": {"dir": None, "dim": 384, "model": "bge-small-en-v1.5"},
    "arctic": {"dir": MODELS / "arctic-embed-s-onnx", "dim": 384, "model": "arctic-embed-s"},
    "gte": {"dir": MODELS / "gte-modernbert-base-onnx", "dim": 768, "model": "gte-modernbert-base"},
    "gemma768": {"dir": MODELS / "embeddinggemma-300m-onnx", "dim": 768, "model": "embeddinggemma-300m"},
    "gemma512": {"dir": MODELS / "embeddinggemma-300m-512-onnx", "dim": 512, "model": "embeddinggemma-300m-512"},
}

DEFAULT_PROBES = os.path.expanduser("~/.cairn/eval/probes-v2-20260721-final.json")
DEFAULT_LME = "/tmp/longmemeval-data/longmemeval_s_cleaned.json"
LIVE_DB = os.path.expanduser("~/.cairn/cairn.db")
OUT_DIR = Path(os.path.expanduser("~/.cairn/eval/model-sweep"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arm", choices=sorted(ARMS))
    ap.add_argument("--probes", default=DEFAULT_PROBES)
    ap.add_argument("--lme-data", default=DEFAULT_LME)
    ap.add_argument("--lme-limit", type=int, default=0)
    ap.add_argument("--skip-lme", action="store_true")
    ap.add_argument("--skip-probes", action="store_true")
    ap.add_argument("--skip-latency", action="store_true")
    args = ap.parse_args()
    arm = ARMS[args.arm]

    # ---- environment BEFORE any cairn import -----------------------------
    if arm["dir"] is not None:
        os.environ["CAIRN_ONNX_MODEL_DIR"] = str(arm["dir"])
    os.environ["CAIRN_EMBEDDING_DIM"] = str(arm["dim"])
    # Pin the reranker so embedder arms are not confounded by reranker
    # auto-detection, and keep benchmark queries out of the query log.
    os.environ.setdefault("CAIRN_RERANKER_MODEL", "ms-marco-MiniLM-L-6-v2")
    os.environ["CAIRN_QUERY_LOG"] = "0"

    # ---- fail-fast model probe -------------------------------------------
    from cairn.embedding import (
        generate_embedding,
        get_active_backend,
        get_embedding_model_info,
        is_embedding_degraded,
    )

    emb = generate_embedding("fail-fast model probe: retrieval quality benchmark")
    backend = get_active_backend()
    info = get_embedding_model_info()
    if is_embedding_degraded() or backend != "onnx":
        print(f"FATAL: arm {args.arm} did not load its ONNX model (backend={backend})")
        sys.exit(2)
    if len(emb) != arm["dim"]:
        print(f"FATAL: arm {args.arm} produced {len(emb)}-dim embeddings, expected {arm['dim']}")
        sys.exit(2)
    print(
        f"[arm {args.arm}] model={info['model_name']} dim={len(emb)} backend={backend} "
        f"dir={info.get('model_dir')}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {"arm": args.arm, "model": info["model_name"], "dim": arm["dim"]}

    # ---- 1. LongMemEval ---------------------------------------------------
    if not args.skip_lme:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))
        from longmemeval_cairn import run as lme_run

        lme_run(args.lme_data, limit=args.lme_limit, tag=f"arm-{args.arm}",
                out_dir=str(OUT_DIR))

    # ---- 2. Judged probes on a rebuilt snapshot ---------------------------
    if not args.skip_probes:
        from cairn.evaluation.retrieval_eval import (
            format_report_v2,
            rebuild_snapshot,
            run_evaluation_v2,
        )

        t0 = time.monotonic()
        snap = rebuild_snapshot(LIVE_DB)
        print(f"[arm {args.arm}] snapshot rebuilt in {time.monotonic() - t0:.0f}s: {snap}")
        report = run_evaluation_v2(args.probes, db_path=snap, variant=f"arm-{args.arm}")
        print(format_report_v2(report))
        out = OUT_DIR / f"probes_{args.arm}.json"
        from dataclasses import asdict

        with open(out, "w") as f:
            json.dump(asdict(report), f, indent=1)
        summary["probes"] = {
            "mrr": report.mrr, "ndcg": report.ndcg_at_k, "hit_rate": report.hit_rate,
            "abstentions": report.abstentions,
        }
        print(f"[arm {args.arm}] probe report → {out}")

    # ---- 3. Latency microbench (run exclusive) ----------------------------
    if not args.skip_latency:
        import resource

        from cairn.embedding import reset_embedding_state

        reset_embedding_state()
        t0 = time.monotonic()
        generate_embedding("model load timing probe")
        load_s = time.monotonic() - t0

        texts = [
            f"latency sample {i}: decision about retry logic in the sync layer "
            f"module number {i} with backoff and jitter parameters" for i in range(200)
        ]
        times = []
        for t in texts:
            t1 = time.monotonic()
            generate_embedding(t)
            times.append((time.monotonic() - t1) * 1000)
        times.sort()
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        summary["latency"] = {
            "load_s": round(load_s, 2), "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1), "rss_mb": round(rss_mb, 0),
        }
        print(
            f"[arm {args.arm}] load={load_s:.2f}s p50={p50:.1f}ms p95={p95:.1f}ms "
            f"rss={rss_mb:.0f}MB"
        )

    with open(OUT_DIR / f"summary_{args.arm}.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[arm {args.arm}] DONE → {OUT_DIR}/summary_{args.arm}.json")


if __name__ == "__main__":
    main()
