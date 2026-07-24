#!/usr/bin/env python3
"""Rank low-cost LLM providers on Cairn's judge task (relevance grading 0-3).

Subcommands:
  sample  Extract a stratified set of (query, memory) pairs + content from a
          frozen probe set, for hand-labeling into a gold set.
  run     Grade the gold pairs with one provider/model (via cairn's real
          judge_grade path), twice, recording grade + latency + tokens.
  score   Rank all run results against the gold labels.

The gold labels are drafted by a human (or LLM-drafted then human-reviewed) in
~/.cairn/eval/llm-bench/gold.jsonl — {id, query, content, gold}. Everything is
scored against THOSE, never the probe set's own silver grades.

    python scripts/llm_provider_bench.py sample --n-per-grade 15
    # (hand-label candidates.jsonl -> gold.jsonl)
    CAIRN_LLM_PROVIDER=openrouter python scripts/llm_provider_bench.py run \
        --label gemini-2.5-flash --model google/gemini-2.5-flash
    python scripts/llm_provider_bench.py score
"""

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

BENCH = Path(os.path.expanduser("~/.cairn/eval/llm-bench"))
DEFAULT_PROBES = os.path.expanduser("~/.cairn/eval/probes-v2-20260721-final.json")


def _candidates_path() -> Path:
    return BENCH / "candidates.jsonl"


def _gold_path() -> Path:
    return BENCH / "gold.jsonl"


# --------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------


def cmd_sample(args) -> None:
    from cairn.sqlite_store import SQLiteStore

    BENCH.mkdir(parents=True, exist_ok=True)
    probes = json.load(open(args.probes))
    store = SQLiteStore()
    try:
        # Collect unique (query, memory_id, silver) triples; resolve content.
        seen = set()
        by_grade = defaultdict(list)
        for pr in probes["probes"]:
            q = pr["query_text"]
            for mid, silver in pr["qrels"].items():
                key = (q, mid)
                if key in seen:
                    continue
                seen.add(key)
                node = store.get_node(mid)
                if node is None:
                    continue
                by_grade[int(silver)].append({
                    "query": q, "memory_id": mid,
                    "content": node.content[:600], "silver": int(silver),
                })
    finally:
        store.close()

    # Stratify: n_per_grade from each silver bin so the sample spans 0-3.
    import random

    rng = random.Random(args.seed)
    out = []
    for g in (0, 1, 2, 3):
        pool = by_grade.get(g, [])
        rng.shuffle(pool)
        out.extend(pool[: args.n_per_grade])
    rng.shuffle(out)
    for i, row in enumerate(out):
        row["id"] = i

    with open(_candidates_path(), "w") as f:
        for row in out:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(out)} candidate pairs -> {_candidates_path()}")
    print("Hand-label into gold.jsonl: {id, query, content, gold} (0-3).")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def cmd_run(args) -> None:
    gold = [json.loads(line) for line in open(_gold_path())]
    from cairn import llm
    from cairn.evaluation.probe_set import judge_grade

    provider = os.environ.get("CAIRN_LLM_PROVIDER", "?")
    if args.model:
        os.environ["CAIRN_LLM_MODEL_FAST"] = args.model
    resolved_model = os.environ.get("CAIRN_LLM_MODEL_FAST", "(default)")

    # Fail fast: one probe call must return a usable grade.
    probe = judge_grade(gold[0]["query"], gold[0]["content"])
    if probe is None:
        print(f"FATAL: {args.label} produced no gradable output on the first pair "
              f"(provider={provider}, model={resolved_model}). Check the key/model id.")
        raise SystemExit(2)

    records = []
    for run_idx in (1, 2):  # twice, for self-consistency
        for row in gold:
            llm.reset_usage()
            t0 = time.monotonic()
            grade = judge_grade(row["query"], row["content"])
            dt_ms = (time.monotonic() - t0) * 1000
            usage = llm.get_last_usage()
            records.append({
                "id": row["id"], "run": run_idx, "grade": grade,
                "latency_ms": round(dt_ms, 1),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
            })

    BENCH.mkdir(parents=True, exist_ok=True)
    out = BENCH / f"run_{args.label}.json"
    with open(out, "w") as f:
        json.dump({
            "label": args.label, "provider": provider, "model": resolved_model,
            "n_pairs": len(gold), "records": records,
        }, f, indent=1)
    ok = sum(1 for r in records if r["grade"] is not None)
    print(f"[{args.label}] {ok}/{len(records)} graded -> {out}")


# --------------------------------------------------------------------------
# score
# --------------------------------------------------------------------------


def _openrouter_prices() -> dict:
    """model_id -> (prompt_$per_tok, completion_$per_tok) from OpenRouter."""
    try:
        import urllib.request

        req = urllib.request.Request("https://openrouter.ai/api/v1/models",
                                     headers={"User-Agent": "cairn-bench"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
        out = {}
        for m in data.get("data", []):
            p = m.get("pricing", {})
            try:
                out[m["id"]] = (float(p.get("prompt", 0)), float(p.get("completion", 0)))
            except (TypeError, ValueError):
                pass
        return out
    except Exception as e:
        print(f"  (price fetch failed: {e}; $/1k will be blank)")
        return {}


def cmd_score(args) -> None:
    gold = {r["id"]: int(r["gold"]) for r in (json.loads(x) for x in open(_gold_path()))}
    prices = _openrouter_prices()
    rows = []
    for path in sorted(BENCH.glob("run_*.json")):
        d = json.load(open(path))
        recs = d["records"]
        # run 1 = accuracy vs gold; run 1 vs run 2 = self-consistency.
        r1 = {r["id"]: r["grade"] for r in recs if r["run"] == 1}
        r2 = {r["id"]: r["grade"] for r in recs if r["run"] == 2}
        graded = [i for i in gold if r1.get(i) is not None]
        n = len(graded) or 1
        exact = sum(1 for i in graded if r1[i] == gold[i]) / n
        mae = sum(abs(r1[i] - gold[i]) for i in graded) / n
        bad = sum(1 for i in graded if abs(r1[i] - gold[i]) >= 2) / n
        both = [i for i in gold if r1.get(i) is not None and r2.get(i) is not None]
        consist = (sum(1 for i in both if r1[i] == r2[i]) / len(both)) if both else 0.0
        fails = sum(1 for r in recs if r["grade"] is None) / (len(recs) or 1)
        lat = sorted(r["latency_ms"] for r in recs if r["grade"] is not None)
        p50 = lat[len(lat) // 2] if lat else 0.0
        # $/1k judgments from measured tokens x live price.
        tin = [r["input_tokens"] for r in recs if r.get("input_tokens")]
        tout = [r["output_tokens"] for r in recs if r.get("output_tokens")]
        cost_1k = None
        pk = prices.get(d.get("model"))
        if pk and tin:
            avg_in = sum(tin) / len(tin)
            avg_out = (sum(tout) / len(tout)) if tout else 0
            cost_1k = (avg_in * pk[0] + avg_out * pk[1]) * 1000
        rows.append({
            "label": d["label"], "model": d.get("model"), "accuracy": round(exact, 3),
            "mae": round(mae, 3), "bad_miss": round(bad, 3), "self_consistency": round(consist, 3),
            "parse_fail": round(fails, 3), "p50_ms": round(p50, 0),
            "cost_per_1k": round(cost_1k, 3) if cost_1k is not None else None,
        })

    rows.sort(key=lambda r: (-r["accuracy"], r["mae"]))
    print(f"\n{'model':32} {'acc':>6} {'MAE':>6} {'bad':>6} {'consist':>8} {'p50ms':>7} {'$/1k':>8}")
    for r in rows:
        c = f"{r['cost_per_1k']:.3f}" if r["cost_per_1k"] is not None else "—"
        print(f"{r['label']:32} {r['accuracy']:>6} {r['mae']:>6} {r['bad_miss']:>6} "
              f"{r['self_consistency']:>8} {int(r['p50_ms']):>7} {c:>8}")
    with open(BENCH / "ranking.json", "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\nRanking -> {BENCH / 'ranking.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--probes", default=DEFAULT_PROBES)
    s.add_argument("--n-per-grade", type=int, default=15)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_sample)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--model", default="")
    r.set_defaults(func=cmd_run)
    sc = sub.add_parser("score")
    sc.set_defaults(func=cmd_score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
