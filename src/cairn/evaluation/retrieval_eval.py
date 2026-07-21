# Copyright 2025-2026 OMEGA Memory Maintainers
# SPDX-License-Identifier: Apache-2.0
"""Retrieval quality evaluation pipeline for Cairn.

Measures how well the retrieval pipeline surfaces relevant memories
by generating probe queries from known memories and evaluating results.

Two modes:
- Basic (no LLM): Sample memories, extract keyword queries, measure hit rate & MRR.
- Judge (with LLM): Generate natural queries and score relevance with LLM-as-judge.

Usage via CLI:
    cairn eval-retrieval                    # Basic mode (no API costs)
    cairn eval-retrieval --judge            # LLM judge mode (uses Anthropic API)
    cairn eval-retrieval --sample-size 50   # Larger sample
    cairn eval-retrieval --output eval.json # Save report
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from cairn.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    """A test query derived from a known memory."""

    source_memory_id: str
    source_content: str
    source_event_type: str
    query_text: str
    query_method: str  # "keyword" or "llm"


@dataclass
class JudgedResult:
    """A retrieved memory with optional LLM relevance score."""

    memory_id: str
    content_preview: str
    event_type: str
    retrieval_score: float
    rank: int
    relevance: Optional[int] = None  # 0-3 LLM judge score
    is_source: bool = False


@dataclass
class ProbeResult:
    """Results of running a single probe query."""

    probe: Probe
    results: List[JudgedResult]
    hit: bool  # Source memory found in top-K
    source_rank: Optional[int] = None  # 1-indexed rank, None if miss
    reciprocal_rank: float = 0.0


@dataclass
class EvalReport:
    """Complete evaluation report with metrics."""

    timestamp: str
    sample_size: int
    top_k: int
    mode: str  # "basic" or "judge"
    total_memories: int
    seed: int = 42

    # Core metrics (always computed)
    hit_rate: float = 0.0
    mrr: float = 0.0

    # LLM judge metrics (judge mode only)
    precision_at_k: Optional[float] = None
    ndcg_at_k: Optional[float] = None
    avg_relevance: Optional[float] = None

    # Breakdown by event type
    by_event_type: Dict[str, Dict[str, float]] = field(default_factory=dict)
    probe_results: List[Dict[str, Any]] = field(default_factory=list)

    # Cost tracking
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    duration_seconds: float = 0.0

    # Multi-rubric judge metrics (arxiv 2602.19320)
    rubric_scores: Optional[Dict[str, float]] = None
    rubric_agreement: Optional[float] = None


# ---------------------------------------------------------------------------
# Keyword-based query generation
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "must",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "between",
        "under",
        "again",
        "then",
        "once",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "because",
        "but",
        "and",
        "or",
        "if",
        "while",
        "that",
        "this",
        "it",
        "its",
        "which",
        "what",
        "who",
        "whom",
        "these",
        "those",
        "am",
        "about",
        "also",
        "up",
        "out",
        "over",
        "any",
        # Cairn-specific noise words that appear in many memories
        "memory",
        "memories",
        "committed",
        "files",
        "changes",
        "session",
        "updated",
        "added",
        "removed",
        "fixed",
        "implemented",
        "created",
    }
)

_TERM_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_.-]{2,}")


def _extract_key_terms(content: str, max_terms: int = 5) -> List[str]:
    """Extract key terms from memory content for keyword-based probing."""
    text = content[:300].lower()
    terms = _TERM_RE.findall(text)
    terms = [t for t in terms if t not in _STOP_WORDS and len(t) > 2]

    seen: set = set()
    unique: list = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique[:max_terms]


def generate_keyword_probe(memory_id: str, content: str, event_type: str) -> Optional[Probe]:
    """Generate a keyword-based probe query from a memory."""
    terms = _extract_key_terms(content)
    if len(terms) < 2:
        return None
    return Probe(
        source_memory_id=memory_id,
        source_content=content[:500],
        source_event_type=event_type,
        query_text=" ".join(terms),
        query_method="keyword",
    )


# ---------------------------------------------------------------------------
# LLM-based query generation and judging
# ---------------------------------------------------------------------------

_DEFAULT_JUDGE_MODEL = "claude-haiku-4-5-20251001"


def _call_llm(
    prompt: str,
    system: str = "",
    model: str = _DEFAULT_JUDGE_MODEL,
    max_tokens: int = 200,
) -> Tuple[str, int, int]:
    """Call Anthropic API. Returns (text, input_tokens, output_tokens)."""
    import anthropic

    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    text = response.content[0].text if response.content else ""
    return text, response.usage.input_tokens, response.usage.output_tokens


def generate_llm_probe(
    memory_id: str,
    content: str,
    event_type: str,
    model: str = _DEFAULT_JUDGE_MODEL,
) -> Tuple[Optional[Probe], int, int]:
    """Generate a natural-language probe query using an LLM.

    Returns (probe_or_None, input_tokens, output_tokens).
    """
    prompt = (
        "Given this memory stored by an AI coding agent, generate a short "
        "search query (5-15 words) that a user or agent might type to find "
        "this information later. Use different wording than the original.\n\n"
        f"Memory type: {event_type}\n"
        f"Memory content:\n{content[:500]}\n\n"
        "Respond with ONLY the search query, nothing else."
    )
    try:
        text, in_tok, out_tok = _call_llm(prompt, model=model)
        query_text = text.strip().strip("\"'")
        if not query_text or len(query_text) < 5:
            return None, in_tok, out_tok
        return (
            Probe(
                source_memory_id=memory_id,
                source_content=content[:500],
                source_event_type=event_type,
                query_text=query_text,
                query_method="llm",
            ),
            in_tok,
            out_tok,
        )
    except Exception as e:
        logger.warning("LLM probe generation failed: %s", e)
        return None, 0, 0


def judge_relevance(
    query: str,
    memory_content: str,
    model: str = _DEFAULT_JUDGE_MODEL,
) -> Tuple[int, int, int]:
    """Score relevance of a retrieved memory to a query (0-3).

    Returns (score, input_tokens, output_tokens).
    """
    prompt = (
        "Score how relevant this retrieved memory is to the search query.\n\n"
        f"Query: {query}\n\n"
        f"Retrieved memory:\n{memory_content[:500]}\n\n"
        "Score 0-3:\n"
        "0 = Not relevant at all\n"
        "1 = Tangentially related\n"
        "2 = Relevant (partially answers or provides useful context)\n"
        "3 = Highly relevant (directly answers the query)\n\n"
        "Respond with ONLY a single digit (0, 1, 2, or 3)."
    )
    try:
        text, in_tok, out_tok = _call_llm(prompt, model=model, max_tokens=5)
        score_str = text.strip()
        score = int(score_str[0]) if score_str and score_str[0].isdigit() else 1
        return min(3, max(0, score)), in_tok, out_tok
    except Exception as e:
        logger.warning("LLM judging failed: %s", e)
        return 1, 0, 0


# ---------------------------------------------------------------------------
# Multi-rubric LLM judge (arxiv 2602.19320 §4.2)
# ---------------------------------------------------------------------------


@dataclass
class SemanticJudgeResult:
    """Multi-rubric judge output with inter-rubric agreement."""

    rubric_scores: Dict[str, float]
    aggregate_score: float
    rubric_agreement: float  # std-dev of rubric scores (lower = more agreement)


DEFAULT_RUBRICS: Dict[str, str] = {
    "factual_accuracy": (
        "Does the retrieved memory contain facts that correctly answer or "
        "support the query? Score 0-3."
    ),
    "semantic_coherence": (
        "Is the retrieved memory semantically related to the query's intent, "
        "even if worded differently? Score 0-3."
    ),
    "reasoning_quality": (
        "Does the retrieved memory provide useful reasoning context — "
        "causal links, decision rationale, or actionable insight? Score 0-3."
    ),
}


def judge_relevance_multi_rubric(
    query: str,
    memory_content: str,
    model: str = _DEFAULT_JUDGE_MODEL,
    rubrics: Optional[Dict[str, str]] = None,
) -> Tuple[SemanticJudgeResult, int, int]:
    """Score relevance across multiple rubrics (arxiv 2602.19320).

    Returns (SemanticJudgeResult, total_input_tokens, total_output_tokens).
    """
    import statistics

    rubrics = rubrics or DEFAULT_RUBRICS
    scores: Dict[str, float] = {}
    total_in = 0
    total_out = 0

    for rubric_name, rubric_desc in rubrics.items():
        prompt = (
            f"Evaluate this retrieved memory against the following rubric.\n\n"
            f"Query: {query}\n\n"
            f"Retrieved memory:\n{memory_content[:500]}\n\n"
            f"Rubric — {rubric_name}:\n{rubric_desc}\n\n"
            "Respond with ONLY a single digit (0, 1, 2, or 3)."
        )
        try:
            text, in_tok, out_tok = _call_llm(prompt, model=model, max_tokens=5)
            total_in += in_tok
            total_out += out_tok
            score_str = text.strip()
            score = int(score_str[0]) if score_str and score_str[0].isdigit() else 1
            scores[rubric_name] = float(min(3, max(0, score)))
        except Exception as e:
            logger.warning("Multi-rubric judge failed for %s: %s", rubric_name, e)
            scores[rubric_name] = 1.0

    values = list(scores.values())
    aggregate = sum(values) / len(values) if values else 0.0
    agreement = statistics.stdev(values) if len(values) >= 2 else 0.0

    return (
        SemanticJudgeResult(
            rubric_scores=scores,
            aggregate_score=round(aggregate, 2),
            rubric_agreement=round(agreement, 3),
        ),
        total_in,
        total_out,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

_EVALUABLE_TYPES = [
    "decision",
    "lesson_learned",
    "error_pattern",
    "user_preference",
    "constraint",
    "memory",
    "user_fact",
]


def sample_memories(store: "SQLiteStore", sample_size: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample a diverse set of memories for evaluation.

    Stratifies by event_type to ensure coverage. Filters out superseded
    and very short memories.
    """
    rng = random.Random(seed)

    type_counts = store.get_type_stats()
    if not type_counts:
        return []

    available = {t: c for t, c in type_counts.items() if t in _EVALUABLE_TYPES and c > 0}
    if not available:
        available = {t: c for t, c in type_counts.items() if c > 0}

    total = sum(available.values())
    if total == 0:
        return []

    # Proportional quotas, minimum 1 per present type
    quotas: Dict[str, int] = {}
    remaining = sample_size
    for etype in sorted(available, key=lambda t: available[t]):
        quota = max(1, round(sample_size * available[etype] / total))
        quota = min(quota, available[etype], remaining)
        quotas[etype] = quota
        remaining -= quota
        if remaining <= 0:
            break

    # Distribute leftover to largest types
    if remaining > 0:
        for etype in sorted(available, key=lambda t: available[t], reverse=True):
            add = min(remaining, available[etype] - quotas.get(etype, 0))
            if add > 0:
                quotas[etype] = quotas.get(etype, 0) + add
                remaining -= add
            if remaining <= 0:
                break

    samples: list = []
    for etype, quota in quotas.items():
        type_memories = store.get_by_type(etype, limit=200)
        candidates = [m for m in type_memories if not (m.metadata or {}).get("superseded") and len(m.content) >= 30]
        if not candidates:
            continue
        selected = rng.sample(candidates, min(quota, len(candidates)))
        for m in selected:
            samples.append(
                {
                    "id": m.id,
                    "content": m.content,
                    "event_type": (m.metadata or {}).get("event_type", "memory"),
                    "created_at": m.created_at.isoformat() if m.created_at else None,
                    "access_count": m.access_count,
                }
            )

    rng.shuffle(samples)
    return samples[:sample_size]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _dcg(scores: List[float], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def _ndcg(scores: List[float], k: int) -> float:
    """Normalized DCG at k."""
    ideal = _dcg(sorted(scores, reverse=True), k)
    if ideal == 0:
        return 0.0
    return _dcg(scores, k) / ideal


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_evaluation(
    db_path: Optional[str] = None,
    sample_size: int = 20,
    top_k: int = 5,
    judge: bool = False,
    model: str = _DEFAULT_JUDGE_MODEL,
    seed: int = 42,
    output_path: Optional[str] = None,
) -> EvalReport:
    """Run the retrieval evaluation pipeline.

    Args:
        db_path: Path to cairn.db. Uses default location if None.
        sample_size: Number of memories to probe.
        top_k: Number of results to retrieve per probe.
        judge: Use LLM to generate queries and score relevance.
        model: Anthropic model for LLM calls (judge mode only).
        seed: Random seed for reproducible sampling.
        output_path: Save JSON report to this path.

    Returns:
        EvalReport with metrics and per-probe details.
    """
    from cairn.sqlite_store import SQLiteStore

    start_time = time.monotonic()

    if db_path:
        store = SQLiteStore(db_path)
    else:
        from cairn.bridge import _get_store

        store = _get_store()

    total_memories = store.node_count()

    memories = sample_memories(store, sample_size=sample_size, seed=seed)
    if not memories:
        logger.warning("No memories to evaluate")
        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_size=0,
            top_k=top_k,
            mode="judge" if judge else "basic",
            total_memories=total_memories,
            seed=seed,
        )

    report = EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sample_size=len(memories),
        top_k=top_k,
        mode="judge" if judge else "basic",
        total_memories=total_memories,
        seed=seed,
    )

    # --- Generate probes ---
    probes: list = []
    for mem in memories:
        if judge:
            probe, in_tok, out_tok = generate_llm_probe(mem["id"], mem["content"], mem["event_type"], model=model)
            report.llm_calls += 1
            report.llm_input_tokens += in_tok
            report.llm_output_tokens += out_tok
        else:
            probe = generate_keyword_probe(mem["id"], mem["content"], mem["event_type"])
        if probe:
            probes.append(probe)

    if not probes:
        logger.warning("No valid probes generated")
        report.duration_seconds = round(time.monotonic() - start_time, 2)
        return report

    # --- Run probes through retrieval pipeline ---
    probe_results: List[ProbeResult] = []
    for probe in probes:
        results = store.query(probe.query_text, limit=top_k)

        judged: list = []
        source_rank: Optional[int] = None
        for rank_idx, result in enumerate(results):
            is_source = result.id == probe.source_memory_id
            if is_source:
                source_rank = rank_idx + 1

            relevance = None
            if judge:
                relevance, in_tok, out_tok = judge_relevance(probe.query_text, result.content, model=model)
                report.llm_calls += 1
                report.llm_input_tokens += in_tok
                report.llm_output_tokens += out_tok

            judged.append(
                JudgedResult(
                    memory_id=result.id,
                    content_preview=result.content[:200],
                    event_type=(result.metadata or {}).get("event_type", "memory"),
                    retrieval_score=round(result.relevance or 0.0, 4),
                    rank=rank_idx + 1,
                    relevance=relevance,
                    is_source=is_source,
                )
            )

        rr = 1.0 / source_rank if source_rank else 0.0
        probe_results.append(
            ProbeResult(
                probe=probe,
                results=judged,
                hit=source_rank is not None,
                source_rank=source_rank,
                reciprocal_rank=rr,
            )
        )

    # --- Compute metrics ---
    report.hit_rate = sum(1 for p in probe_results if p.hit) / len(probe_results)
    report.mrr = sum(p.reciprocal_rank for p in probe_results) / len(probe_results)

    if judge:
        precisions: list = []
        ndcgs: list = []
        all_scores: list = []

        for pr in probe_results:
            scores = [j.relevance or 0 for j in pr.results]
            all_scores.extend(scores)
            if scores:
                precisions.append(sum(1 for s in scores if s >= 2) / len(scores))
                ndcgs.append(_ndcg(scores, top_k))

        if precisions:
            report.precision_at_k = round(sum(precisions) / len(precisions), 4)
        if ndcgs:
            report.ndcg_at_k = round(sum(ndcgs) / len(ndcgs), 4)
        if all_scores:
            report.avg_relevance = round(sum(all_scores) / len(all_scores), 2)

    # --- Breakdown by event type ---
    type_groups: Dict[str, List[ProbeResult]] = {}
    for pr in probe_results:
        type_groups.setdefault(pr.probe.source_event_type, []).append(pr)

    for etype, group in type_groups.items():
        report.by_event_type[etype] = {
            "count": len(group),
            "hit_rate": round(sum(1 for p in group if p.hit) / len(group), 4),
            "mrr": round(sum(p.reciprocal_rank for p in group) / len(group), 4),
        }

    # --- Serialize probe details ---
    for pr in probe_results:
        report.probe_results.append(
            {
                "query": pr.probe.query_text,
                "query_method": pr.probe.query_method,
                "source_id": pr.probe.source_memory_id,
                "source_type": pr.probe.source_event_type,
                "hit": pr.hit,
                "source_rank": pr.source_rank,
                "results": [
                    {
                        "id": j.memory_id,
                        "rank": j.rank,
                        "score": j.retrieval_score,
                        "relevance": j.relevance,
                        "is_source": j.is_source,
                        "preview": j.content_preview[:100],
                    }
                    for j in pr.results
                ],
            }
        )

    report.duration_seconds = round(time.monotonic() - start_time, 2)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("Report saved to %s", output_path)

    return report


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_report(report: EvalReport) -> str:
    """Format an EvalReport as human-readable markdown."""
    lines = [
        "# Cairn Retrieval Evaluation Report",
        "",
        f"**Date:** {report.timestamp[:19]}",
        f"**Mode:** {report.mode}",
        f"**Sample:** {report.sample_size} probes from {report.total_memories} memories",
        f"**Top-K:** {report.top_k}",
        f"**Seed:** {report.seed}",
        f"**Duration:** {report.duration_seconds}s",
        "",
        "## Core Metrics",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Hit Rate@{report.top_k} | {report.hit_rate:.1%} |",
        f"| MRR | {report.mrr:.3f} |",
    ]

    if report.mode == "judge":
        if report.precision_at_k is not None:
            lines.append(f"| Precision@{report.top_k} | {report.precision_at_k:.1%} |")
        if report.ndcg_at_k is not None:
            lines.append(f"| NDCG@{report.top_k} | {report.ndcg_at_k:.3f} |")
        if report.avg_relevance is not None:
            lines.append(f"| Avg Relevance | {report.avg_relevance:.2f}/3.0 |")

    if report.by_event_type:
        lines.extend(
            [
                "",
                "## By Event Type",
                "",
                "| Type | Count | Hit Rate | MRR |",
                "|------|-------|----------|-----|",
            ]
        )
        for etype, metrics in sorted(report.by_event_type.items()):
            lines.append(f"| {etype} | {metrics['count']} | {metrics['hit_rate']:.1%} | {metrics['mrr']:.3f} |")

    if report.mode == "judge" and report.llm_calls > 0:
        lines.extend(
            [
                "",
                "## Cost",
                f"- LLM calls: {report.llm_calls}",
                f"- Input tokens: {report.llm_input_tokens:,}",
                f"- Output tokens: {report.llm_output_tokens:,}",
            ]
        )

    misses = [p for p in report.probe_results if not p["hit"]]
    if misses:
        lines.extend(
            [
                "",
                f"## Misses ({len(misses)}/{report.sample_size} probes)",
                "",
            ]
        )
        for miss in misses[:10]:
            lines.append(f"- **Query:** `{miss['query'][:80]}`")
            lines.append(f"  Source: `{miss['source_id']}` ({miss['source_type']})")
            if miss.get("results"):
                top = miss["results"][0]
                lines.append(f"  Top result: `{top['id']}` (score {top['score']})")

    return "\n".join(lines)


# ===========================================================================
# Eval v2 — non-self-referential probes with judged qrels
# ===========================================================================
#
# v1 above measures "can we re-find a memory from its own wording" — useful as
# a smoke signal, structurally over-optimistic. v2 runs frozen probe sets
# (see probe_set.py) whose relevance labels are independent LLM-judged qrels,
# so it can arbitrate scoring changes (e.g. the RRF fusion-mode A/B).


@dataclass
class ProbeResultV2:
    """Per-probe outcome against judged qrels."""

    query_text: str
    reciprocal_rank: float          # 1/rank of first grade>=2 result, else 0
    precision_at_k: float           # fraction of top-K with grade>=2
    ndcg_at_k: float                # graded nDCG (unjudged results = grade 0)
    returned: int
    judged_returned: int            # how many returned results had a qrel entry
    result_ids: List[str] = field(default_factory=list)


@dataclass
class EvalReportV2:
    """Aggregate metrics for one run of a frozen probe set."""

    timestamp: str
    probe_set_path: str
    probe_set_sha256: str
    probe_count: int
    top_k: int
    variant: str                    # label for this run's configuration
    env_overrides: Dict[str, str] = field(default_factory=dict)
    expansion_enabled: bool = False
    cross_encoder_enabled: bool = True
    db_path: str = ""
    mrr: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    hit_rate: float = 0.0           # probes with >=1 grade>=2 result in top-K
    abstentions: int = 0            # probes returning zero results
    pool_coverage: float = 0.0      # judged_returned / returned, averaged
    per_probe: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


def snapshot_db(src_db: str, dest_dir: Optional[str] = None) -> str:
    """Copy a cairn.db via the SQLite online-backup API into a temp dir.

    Eval queries mutate access/retrieval counts — they must never run against
    the live store. Returns the path of the snapshot copy.
    """
    import sqlite3
    import tempfile

    if dest_dir is None:
        dest_dir = tempfile.mkdtemp(prefix="cairn-eval-")
    dest = str(Path(dest_dir) / "cairn.db")
    src = sqlite3.connect(src_db, timeout=30)
    dst = sqlite3.connect(dest)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return dest


def run_evaluation_v2(
    probe_set_path: str,
    db_path: Optional[str] = None,
    top_k: int = 5,
    env: Optional[Dict[str, str]] = None,
    variant: str = "default",
) -> EvalReportV2:
    """Run a frozen probe set. Snapshot-by-default; provider-free.

    When ``db_path`` is None the live DB is snapshotted first. When a path is
    given it is used directly (compare_ab passes per-arm copies so arm 1's
    access-count bumps can't bias arm 2). ``env`` holds the variant's
    environment overrides, restored afterwards.

    Determinism controls for comparisons: query expansion and query logging
    are disabled for the duration; the cross-encoder is left as configured.
    """
    import os

    from cairn.evaluation.probe_set import load_probe_set

    probes, meta = load_probe_set(probe_set_path)

    if db_path is None:
        from cairn.paths import db_path as live_db
        db_path = snapshot_db(str(live_db()))

    overrides = dict(env or {})
    forced = {"CAIRN_QUERY_EXPANSION": "0", "CAIRN_QUERY_LOG": "0"}
    all_keys = set(forced) | set(overrides)
    saved = {k: os.environ.get(k) for k in all_keys}
    os.environ.update(forced)
    os.environ.update(overrides)

    start = time.time()
    try:
        from cairn.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path=db_path)
        try:
            per: List[ProbeResultV2] = []
            for probe in probes:
                try:
                    results = store.query(probe.query_text, limit=top_k)
                except Exception:
                    logger.warning("probe query failed: %r", probe.query_text, exc_info=True)
                    results = []
                ids = [r.id for r in results]
                grades = [probe.qrels.get(i, 0) for i in ids]
                judged = sum(1 for i in ids if i in probe.qrels)

                rr = 0.0
                for rank, g in enumerate(grades, start=1):
                    if g >= 2:
                        rr = 1.0 / rank
                        break
                p_at_k = (sum(1 for g in grades if g >= 2) / top_k) if top_k else 0.0
                # nDCG ideal comes from the full qrel set, not just returned
                ideal = sorted(probe.qrels.values(), reverse=True)
                ideal_dcg = _dcg([float(g) for g in ideal], top_k)
                ndcg = (_dcg([float(g) for g in grades], top_k) / ideal_dcg) if ideal_dcg > 0 else 0.0

                per.append(ProbeResultV2(
                    query_text=probe.query_text,
                    reciprocal_rank=rr,
                    precision_at_k=p_at_k,
                    ndcg_at_k=ndcg,
                    returned=len(ids),
                    judged_returned=judged,
                    result_ids=ids,
                ))
        finally:
            store.close()

        n = len(per) or 1
        with_results = [p for p in per if p.returned]
        report = EvalReportV2(
            timestamp=datetime.now(timezone.utc).isoformat(),
            probe_set_path=probe_set_path,
            probe_set_sha256=str(meta.get("content_sha256", "")),
            probe_count=len(per),
            top_k=top_k,
            variant=variant,
            env_overrides=overrides,
            expansion_enabled=False,
            cross_encoder_enabled=os.environ.get("CAIRN_CROSS_ENCODER") != "0",
            db_path=db_path,
            mrr=sum(p.reciprocal_rank for p in per) / n,
            precision_at_k=sum(p.precision_at_k for p in per) / n,
            ndcg_at_k=sum(p.ndcg_at_k for p in per) / n,
            hit_rate=sum(1 for p in per if p.reciprocal_rank > 0) / n,
            abstentions=sum(1 for p in per if not p.returned),
            pool_coverage=(
                sum(p.judged_returned / p.returned for p in with_results) / len(with_results)
                if with_results else 0.0
            ),
            per_probe=[asdict(p) for p in per],
            duration_seconds=round(time.time() - start, 2),
        )
        return report
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def write_history_row(report: EvalReportV2, csv_path: Optional[str] = None) -> str:
    """Append a run to eval-history.csv (column 2 = MRR; the doctor's trend
    parser reads float(row[1]))."""
    if csv_path is None:
        from cairn.paths import cairn_home
        logs = cairn_home() / "logs"
        logs.mkdir(parents=True, exist_ok=True, mode=0o700)
        csv_path = str(logs / "eval-history.csv")
    p = Path(csv_path)
    header = "date,mrr,ndcg,p_at_k,probe_version,variant\n"
    line = (
        f"{report.timestamp},{report.mrr:.4f},{report.ndcg_at_k:.4f},"
        f"{report.precision_at_k:.4f},{report.probe_set_sha256[:12]},{report.variant}\n"
    )
    if not p.exists() or not p.read_text().strip():
        p.write_text(header + line)
    else:
        with p.open("a") as fh:
            fh.write(line)
    return csv_path


def sign_test(deltas: List[float]) -> Tuple[int, int, float]:
    """Two-sided exact binomial sign test over paired deltas.

    Returns (n_positive, n_negative, p_value); zero deltas are discarded.
    """
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n = pos + neg
    if n == 0:
        return 0, 0, 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return pos, neg, min(1.0, 2 * tail)


def compare_ab(
    probe_set_path: str,
    variant_a: Tuple[str, Dict[str, str]],
    variant_b: Tuple[str, Dict[str, str]],
    src_db: Optional[str] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Paired A/B of two env-variant configurations on identical snapshots.

    Each variant is (label, env_overrides). Each arm gets its OWN copy of the
    base snapshot so arm A's access-count bumps cannot bias arm B
    (access-aware decay reads those counts). Returns per-metric deltas (B - A)
    and a two-sided sign test over per-probe reciprocal-rank deltas.
    """
    import shutil
    import tempfile

    if src_db is None:
        from cairn.paths import db_path as live_db
        src_db = str(live_db())

    base_dir = tempfile.mkdtemp(prefix="cairn-ab-")
    base = snapshot_db(src_db, base_dir)
    arms: Dict[str, EvalReportV2] = {}
    for idx, (label, env) in enumerate((variant_a, variant_b)):
        arm_db = str(Path(base_dir) / f"cairn-arm{idx}.db")
        shutil.copy2(base, arm_db)
        arms[label] = run_evaluation_v2(
            probe_set_path, db_path=arm_db, top_k=top_k, env=env, variant=label,
        )

    label_a, label_b = variant_a[0], variant_b[0]
    ra, rb = arms[label_a], arms[label_b]
    rr_deltas = [
        b["reciprocal_rank"] - a["reciprocal_rank"]
        for a, b in zip(ra.per_probe, rb.per_probe)
    ]
    pos, neg, p = sign_test(rr_deltas)
    return {
        "a": asdict(ra),
        "b": asdict(rb),
        "labels": [label_a, label_b],
        "delta": {
            "mrr": rb.mrr - ra.mrr,
            "ndcg_at_k": rb.ndcg_at_k - ra.ndcg_at_k,
            "precision_at_k": rb.precision_at_k - ra.precision_at_k,
            "hit_rate": rb.hit_rate - ra.hit_rate,
        },
        "sign_test": {"b_wins": pos, "a_wins": neg, "p_value": p},
    }


def format_report_v2(report: EvalReportV2) -> str:
    """Human-readable summary of a v2 run."""
    lines = [
        "# Retrieval Eval v2",
        f"- Probe set: {report.probe_set_path} ({report.probe_count} probes, sha {report.probe_set_sha256[:12]})",
        f"- Variant: {report.variant} {report.env_overrides or ''} | top-K: {report.top_k} | expansion: off | CE: {'on' if report.cross_encoder_enabled else 'off'}",
        "",
        f"| MRR | nDCG@{report.top_k} | P@{report.top_k} | hit-rate | abstentions | pool coverage |",
        "|-----|------|------|----------|-------------|---------------|",
        f"| {report.mrr:.3f} | {report.ndcg_at_k:.3f} | {report.precision_at_k:.3f} "
        f"| {report.hit_rate:.3f} | {report.abstentions} | {report.pool_coverage:.2f} |",
        f"\nDuration: {report.duration_seconds}s | DB: {report.db_path}",
    ]
    return "\n".join(lines)
