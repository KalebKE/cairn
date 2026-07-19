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
