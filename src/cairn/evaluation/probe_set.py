"""Non-self-referential probe sets for retrieval evaluation (v2).

The v1 harness derives each probe query FROM the target memory and scores
"did the source come back" — which over-estimates retrieval quality (queries
share the target's wording) and cannot arbitrate scoring changes.

v2 probes carry an independent relevance label instead:

- The sampled memory only *seeds a topic*; an LLM writes the question a
  developer would actually ask, deliberately avoiding the memory's wording.
- Candidates are pooled from BOTH RRF fusion modes (plus the seed and random
  negatives) so an A/B between modes is not biased toward either pool.
- Every pooled (query, memory) pair is judged 0-3 by an LLM; the frozen
  ``qrels`` map is the ground truth. Runs against frozen probes need no LLM.

Probe sets are frozen, versioned JSON under ``CAIRN_HOME/eval/`` — memory
content is personal and never belongs in the repo.
"""
import hashlib
import json
import logging
import os
import random
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("cairn.evaluation.probe_set")

PROBE_SET_VERSION = 2

_WORD_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_\-\.]{3,}")

_GEN_SYSTEM = (
    "You write realistic search queries for a developer's personal memory "
    "system. Output ONLY the query text, nothing else."
)

_GEN_PROMPT = (
    "Here is a stored memory from a developer's knowledge base:\n\n"
    "---\n{content}\n---\n\n"
    "Write the short question or search phrase this developer would actually "
    "type when they NEED this information later, before knowing it exists.\n"
    "Rules:\n"
    "- Do NOT reuse the memory's distinctive words, identifiers, or phrasing; "
    "describe the need in different words.\n"
    "- 4-15 words, no quotes, no punctuation-only output.\n"
    "- It must still be answerable by the memory above."
)

_JUDGE_SYSTEM = (
    "You judge search relevance for a developer's memory system. "
    "Respond with ONLY a single digit."
)

_JUDGE_PROMPT = (
    "Score how relevant this retrieved memory is to the search query.\n\n"
    "Query: {query}\n\n"
    "Retrieved memory:\n{content}\n\n"
    "Score 0-3:\n"
    "0 = Not relevant at all\n"
    "1 = Tangentially related\n"
    "2 = Relevant (partially answers or provides useful context)\n"
    "3 = Highly relevant (directly answers the query)\n\n"
    "Respond with ONLY a single digit (0, 1, 2, or 3)."
)


@dataclass
class ProbeV2:
    """A query with an independent judged relevance set (qrels)."""

    query_text: str
    method: str  # "llm-divergent"
    qrels: Dict[str, int]  # memory_id -> grade 0-3
    meta: Dict[str, Any] = field(default_factory=dict)


def _content_words(text: str) -> set:
    return {w.lower() for w in _WORD_RE.findall(text)}


def lexical_overlap(query: str, content: str) -> float:
    """Fraction of the query's content words that also appear in `content`."""
    qw = _content_words(query)
    if not qw:
        return 1.0
    cw = _content_words(content)
    return len(qw & cw) / len(qw)


def _llm(prompt: str, system: str, max_tokens: int = 100) -> str:
    from cairn.llm import llm_complete

    return llm_complete(
        prompt, system, max_tokens=max_tokens, temperature=0.3,
        timeout=15.0, model_tier="fast",
    )


def generate_divergent_query(content: str, max_overlap: float = 0.5) -> Optional[Tuple[str, float]]:
    """LLM-generate a query that does not reuse the memory's wording.

    Returns (query, overlap) or None when generation fails or stays too
    lexically close after one retry.
    """
    prompt = _GEN_PROMPT.format(content=content[:1200])
    for attempt in range(2):
        q = _llm(prompt, _GEN_SYSTEM, max_tokens=60).strip().strip('"')
        if not q or len(q.split()) < 3:
            continue
        ov = lexical_overlap(q, content)
        if ov <= max_overlap:
            return q, ov
        # retry once, harder
        prompt = (
            _GEN_PROMPT.format(content=content[:1200])
            + f"\n\nYour previous attempt reused too many of the memory's words: \"{q}\". "
            "Rephrase using entirely different vocabulary."
        )
    return None


def judge_grade(query: str, content: str) -> Optional[int]:
    """Judge (query, memory) relevance 0-3 via the configured LLM provider.

    Returns None on provider failure (caller decides how to handle).
    """
    text = _llm(
        _JUDGE_PROMPT.format(query=query, content=content[:600]),
        _JUDGE_SYSTEM,
        max_tokens=5,
    ).strip()
    if text and text[0].isdigit():
        return min(3, max(0, int(text[0])))
    return None


def _pool_candidates(store, query: str, seed_id: str, top_k: int,
                     negatives: int, rng: random.Random) -> List[str]:
    """Union of a deep top-K pass + the seed + random negatives.

    Pooling deeper than eval top-K keeps the frozen qrels useful when a
    config variant shuffles ranks (judged pool covers more than one ordering).
    """
    ids: List[str] = []
    try:
        results = store.query(query, limit=top_k)
    except Exception:
        logger.warning("pooling query failed", exc_info=True)
        results = []
    for r in results:
        if r.id not in ids:
            ids.append(r.id)

    if seed_id not in ids:
        ids.append(seed_id)

    # Random negatives from the wider store
    try:
        rows = store._conn.execute(
            "SELECT node_id FROM memories WHERE node_id NOT IN ({}) "
            "ORDER BY node_id LIMIT 500".format(",".join("?" * len(ids))),
            ids,
        ).fetchall()
        pool = [r[0] for r in rows]
        rng.shuffle(pool)
        ids.extend(pool[:negatives])
    except Exception:
        logger.debug("negative sampling failed", exc_info=True)
    return ids


def build_probe_set(
    store,
    size: int = 30,
    seed: int = 42,
    top_k: int = 10,
    negatives: int = 3,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build and freeze a judged probe set against `store`.

    The caller is responsible for passing a SNAPSHOT store — pooling queries
    mutate access counts on whatever DB they run against. Query logging is
    disabled for the duration so probe traffic never pollutes the replay
    corpus.
    """
    from cairn.evaluation.retrieval_eval import sample_memories

    prev_qlog = os.environ.get("CAIRN_QUERY_LOG")
    os.environ["CAIRN_QUERY_LOG"] = "0"
    try:
        rng = random.Random(seed)
        samples = sample_memories(store, sample_size=size, seed=seed)
        probes: List[ProbeV2] = []
        skipped: List[Dict[str, str]] = []

        for mem in samples:
            gen = generate_divergent_query(mem["content"])
            if gen is None:
                skipped.append({"id": mem["id"], "reason": "generation_failed_or_overlap"})
                continue
            query, overlap = gen

            candidate_ids = _pool_candidates(store, query, mem["id"], top_k, negatives, rng)
            qrels: Dict[str, int] = {}
            for cid in candidate_ids:
                node = store.get_node(cid)
                if node is None:
                    continue
                grade = judge_grade(query, node.content)
                if grade is None:
                    skipped.append({"id": mem["id"], "reason": "judge_failed"})
                    qrels = {}
                    break
                qrels[cid] = grade

            if not qrels:
                continue
            if max(qrels.values()) < 2:
                skipped.append({"id": mem["id"], "reason": "no_relevant_in_pool"})
                continue

            probes.append(ProbeV2(
                query_text=query,
                method="llm-divergent",
                qrels=qrels,
                meta={
                    "seed_memory_id": mem["id"],
                    "event_type_hint": mem["event_type"],
                    "lexical_overlap": round(overlap, 3),
                },
            ))

        store_rows = store.node_count()
        payload = {
            "version": PROBE_SET_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "requested_size": size,
            "probe_count": len(probes),
            "store_rows": store_rows,
            "skipped": skipped,
            "probes": [asdict(p) for p in probes],
        }
        blob = json.dumps(payload["probes"], sort_keys=True).encode()
        payload["content_sha256"] = hashlib.sha256(blob).hexdigest()

        if out_path is None:
            from cairn.paths import cairn_home
            eval_dir = cairn_home() / "eval"
            eval_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            out_path = str(eval_dir / f"probes-v2-{stamp}-s{seed}.json")

        Path(out_path).write_text(json.dumps(payload, indent=2))
        payload["path"] = out_path
        return payload
    finally:
        if prev_qlog is None:
            os.environ.pop("CAIRN_QUERY_LOG", None)
        else:
            os.environ["CAIRN_QUERY_LOG"] = prev_qlog


def load_probe_set(path: str) -> Tuple[List[ProbeV2], Dict[str, Any]]:
    """Load and validate a frozen probe set. Returns (probes, meta)."""
    data = json.loads(Path(path).read_text())
    if data.get("version") != PROBE_SET_VERSION:
        raise ValueError(
            f"probe set version {data.get('version')} != expected {PROBE_SET_VERSION}"
        )
    probes = []
    for p in data.get("probes", []):
        if not p.get("query_text") or not isinstance(p.get("qrels"), dict):
            raise ValueError(f"malformed probe entry: {p!r:.120}")
        probes.append(ProbeV2(
            query_text=p["query_text"],
            method=p.get("method", "unknown"),
            qrels={k: int(v) for k, v in p["qrels"].items()},
            meta=p.get("meta", {}),
        ))
    if not probes:
        raise ValueError(f"probe set at {path} contains no probes")
    meta = {k: v for k, v in data.items() if k != "probes"}
    return probes, meta
