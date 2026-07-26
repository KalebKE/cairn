"""Query pipeline mixin for SQLiteStore."""

import logging
import math
import os
import re
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


from ._types import (
    MemoryResult,
    SurfacingContext,
    QueryIntent,
    _SURFACING_THRESHOLDS,
    _INTENT_WEIGHTS,
    _RRF_K,
    _canonicalize,
    _CONJUNCTION_PATTERN,
    _CLAUSE_STARTS,
    _deserialize_f32,
    _cosine_similarity,
)

logger = logging.getLogger("cairn.sqlite_store")

# Failed-import caches: Python does NOT cache failed imports, and these
# optional modules are probed on hot query paths (per candidate, per query).
_THOMPSON_UNAVAILABLE = False
_PATTERN_LEARNER_UNAVAILABLE = False

# Strong signal short-circuit thresholds (QMD-inspired)
STRONG_SIGNAL_THRESHOLD = float(os.environ.get("CAIRN_STRONG_SIGNAL_THRESHOLD", "0.85"))
STRONG_SIGNAL_GAP = float(os.environ.get("CAIRN_STRONG_SIGNAL_GAP", "0.15"))

# Adaptive retry: when confidence < threshold, retry with relaxed params
ADAPTIVE_RETRY_THRESHOLD = float(os.environ.get("CAIRN_ADAPTIVE_RETRY_THRESHOLD", "0.3"))
ADAPTIVE_RETRY_RELAXATION = float(os.environ.get("CAIRN_ADAPTIVE_RETRY_RELAXATION", "0.6"))


@dataclass
class RetrievalContext:
    """Per-query state for the retrieval pipeline.

    Replaces threading 8 shared mutable containers + a dozen config values
    through every ``_query_phase_*`` signature. Config fields are read-only
    inputs; the container fields (all_results … query_emb) are the mutable
    accumulators the phases populate in place. Each phase unpacks the fields
    it needs at the top and otherwise operates on the same objects, so the
    in-place mutation semantics (e.g. node_scores.clear()/update() in boost)
    are preserved exactly.
    """
    # --- config (read-only inputs) ---
    query_text: str
    limit: int
    session_id: Optional[str]
    exclude_types: Optional[List[str]]
    include_infrastructure: bool
    project_path: str
    scope: str
    context_file: str
    context_tags: Optional[List[str]]
    temporal_range: Optional[tuple]
    entity_id: Optional[str]
    agent_type: Optional[str]
    query_hint: Optional[str]
    surfacing_context: Optional[SurfacingContext]
    temporal_boost_only: bool
    perspective: Optional[str]
    valid_at: Optional[str]
    query_intent: Any
    skip_vec: bool
    query_embedding: Optional[List[float]]
    # weights (ALMA profile × intent × ctx-boost)
    pw_vec: float
    pw_text: float
    pw_word: float
    pw_ctx: float
    pw_graph: float
    # abstention thresholds (surfacing profile, relaxed on retry)
    ctx_min_vec: float
    ctx_min_text: float
    ctx_min_composite: float
    # cache / timing
    cache_key: Any
    now_mono: float
    # --- mutable accumulators (populated in place by phases) ---
    all_results: Dict[str, "MemoryResult"] = field(default_factory=dict)
    node_scores: Dict[str, float] = field(default_factory=dict)
    raw_vec_sims: Dict[str, float] = field(default_factory=dict)
    vec_ranked: List[Tuple[str, float]] = field(default_factory=list)
    text_ranked: List[Tuple[str, float]] = field(default_factory=list)
    temporal_ranked: List[Tuple[str, float]] = field(default_factory=list)
    entity_ranked: List[Tuple[str, float]] = field(default_factory=list)
    query_emb: Optional[List[float]] = None


class QueryMixin:

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        limit: int = 10,
        session_id: Optional[str] = None,
        use_cache: bool = True,
        expand_query: bool = True,
        exclude_types: Optional[List[str]] = None,
        include_infrastructure: bool = False,
        project_path: str = "",
        scope: str = "project",
        context_file: str = "",
        context_tags: Optional[List[str]] = None,
        temporal_range: Optional[tuple] = None,
        entity_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        query_hint: Optional[str] = None,
        surfacing_context: Optional[SurfacingContext] = None,
        temporal_boost_only: bool = False,
        perspective: Optional[str] = None,
        valid_at: Optional[str] = None,
        _is_retry: bool = False,
        query_embedding: Optional[List[float]] = None,
    ) -> List[MemoryResult]:
        """Search memories using vector similarity + text matching.

        When context_file or context_tags are provided, results whose tags,
        project, or file paths overlap with the current context receive a
        relevance boost, improving results for the user's active work.

        When use_cache is True (default), identical queries within the TTL
        window return cached results, avoiding the full vector+FTS5 pipeline.

        surfacing_context controls dynamic threshold profiles (#4 Engram).
        """
        from ._types import (
            _QUERY_CACHE_MAX,
            _QUERY_CACHE_TTL_S,
            _QUERY_CACHE_WARM_TTL_S,
            _HOT_CACHE_REFRESH_S,
            _SESSION_CACHE_MAX,
            _CLEANUP_INTERVAL,
        )
        import cairn.sqlite_store._types as _types_mod

        _t0_agency = _time.monotonic()
        now_mono = _time.monotonic()

        # Resolve surfacing context thresholds (#4)
        _ctx = surfacing_context or SurfacingContext.GENERAL
        _ctx_thresholds = _SURFACING_THRESHOLDS.get(_ctx, _SURFACING_THRESHOLDS[SurfacingContext.GENERAL])
        ctx_min_vec, ctx_min_text, ctx_min_composite, ctx_weight_boost = _ctx_thresholds

        # Adaptive retry: relax abstention thresholds on retry pass
        if _is_retry:
            _relax = ADAPTIVE_RETRY_RELAXATION
            ctx_min_vec *= _relax
            ctx_min_text *= _relax
            ctx_min_composite *= max(_relax - 0.1, 0.3)

        # --- Query result cache: check (tiered TTL #2) ---
        _cache_key = None
        if use_cache:
            _cache_key = (
                query_text, limit, session_id,
                tuple(sorted(exclude_types)) if exclude_types else (),
                include_infrastructure, project_path, scope,
                context_file, tuple(context_tags) if context_tags else (),
                temporal_range, entity_id, agent_type, query_hint,
                surfacing_context, temporal_boost_only, perspective,
                valid_at,
            )
            with self._cache_lock:
                cached = self._query_cache.get(_cache_key)
                if cached is not None:
                    ts, results, confidence = cached
                    ttl = _QUERY_CACHE_WARM_TTL_S if confidence > 0.7 else _QUERY_CACHE_TTL_S
                    if (now_mono - ts) < ttl:
                        self._query_cache.move_to_end(_cache_key)
                        self.stats["queries"] += 1
                        self.stats["hits"] += 1
                        self._record_timing("read", (_time.monotonic() - _t0_agency) * 1000)
                        return results
                    else:
                        del self._query_cache[_cache_key]
        if _types_mod._last_cleanup is None or (now_mono - _types_mod._last_cleanup) > _CLEANUP_INTERVAL:
            _types_mod._last_cleanup = now_mono
            try:
                self.cleanup_expired()
            except Exception as e:
                logger.warning(f"Periodic cleanup failed: {e}")

        self.stats["queries"] += 1

        # Query decomposition: split compound queries into sub-queries,
        # run each independently, merge results with max-score dedup.
        sub_queries = self._decompose_query(query_text) if self._decompose_queries else None
        if sub_queries and not getattr(self, "_in_decomposition", False):
            try:
                self._in_decomposition = True  # Prevent recursive decomposition
                merged: Dict[str, MemoryResult] = {}
                merged_scores: Dict[str, float] = {}
                for sq in sub_queries:
                    sq_results = self.query(
                        sq, limit=limit, session_id=session_id,
                        use_cache=use_cache, expand_query=expand_query,
                        exclude_types=exclude_types,
                        include_infrastructure=include_infrastructure,
                        project_path=project_path, scope=scope,
                        context_file=context_file, context_tags=context_tags,
                        temporal_range=temporal_range, entity_id=entity_id,
                        agent_type=agent_type, query_hint=query_hint,
                        surfacing_context=surfacing_context,
                        temporal_boost_only=temporal_boost_only,
                    )
                    for r in sq_results:
                        if r.id not in merged or r.relevance > merged[r.id].relevance:
                            merged[r.id] = r
                            merged_scores[r.id] = max(merged_scores.get(r.id, 0), r.relevance)
                        elif r.id in merged:
                            # Boost memories found by multiple sub-queries
                            merged_scores[r.id] *= 1.15
                # Sort by merged score, re-normalize, return top-limit
                sorted_ids = sorted(merged_scores, key=merged_scores.get, reverse=True)[:limit]
                max_score = merged_scores[sorted_ids[0]] if sorted_ids else 1.0
                final = []
                for nid in sorted_ids:
                    r = merged[nid]
                    r.relevance = merged_scores[nid] / max_score
                    final.append(r)
                if _cache_key is not None:
                    confidence = sum(r.relevance for r in final[:3]) / min(len(final), 3) if final else 0.0
                    with self._cache_lock:
                        self._query_cache[_cache_key] = (now_mono, final, confidence)
                        while len(self._query_cache) > _QUERY_CACHE_MAX:
                            self._query_cache.popitem(last=False)
                self._record_timing("read", (_time.monotonic() - _t0_agency) * 1000)
                return final
            finally:
                self._in_decomposition = False

        if (now_mono - self._hot_cache_ts) > _HOT_CACHE_REFRESH_S:
            self._refresh_hot_cache()
        fast_path_results = self._fast_path_lookup(query_text, limit=limit)
        if fast_path_results:
            self.stats["fast_path_hits"] = self.stats.get("fast_path_hits", 0) + 1
            if _cache_key is not None:
                with self._cache_lock:
                    self._query_cache[_cache_key] = (now_mono, fast_path_results, 0.9)
                    while len(self._query_cache) > _QUERY_CACHE_MAX:
                        self._query_cache.popitem(last=False)
            if session_id:
                self._session_cache[session_id] = fast_path_results
                if len(self._session_cache) > _SESSION_CACHE_MAX:
                    # Evict oldest entry (first inserted)
                    try:
                        oldest = next(iter(self._session_cache))
                        del self._session_cache[oldest]
                    except StopIteration:
                        pass
            return fast_path_results
        hot_results = self._check_hot_tier(query_text, limit=limit)
        if hot_results:
            self.stats["hot_cache_hits"] = self.stats.get("hot_cache_hits", 0) + 1
        query_intent = self._classify_query_intent(query_text)
        _intent_key = f"intent_{query_intent.value}" if query_intent else "intent_none"
        self.stats[_intent_key] = self.stats.get(_intent_key, 0) + 1

        # Resolve retrieval profile for phase weighting (ALMA-inspired)
        _profile = self._retrieval_profiles_merged.get(
            query_hint, self._retrieval_profiles_merged.get("_default", (1.0, 1.0, 1.0, 1.0, 1.0))
        ) if query_hint else self._retrieval_profiles_merged.get("_default", (1.0, 1.0, 1.0, 1.0, 1.0))
        pw_vec, pw_text, pw_word, pw_ctx, pw_graph = _profile

        # Apply adaptive intent-based weights (#3)
        if query_intent:
            iw = _INTENT_WEIGHTS.get(query_intent, (1.0, 1.0, 1.0, 1.0, 1.0))
            pw_vec *= iw[0]
            pw_text *= iw[1]
            pw_word *= iw[2]
            pw_ctx *= iw[3]
            pw_graph *= iw[4]

        # Apply context weight boost (#4)
        pw_ctx *= ctx_weight_boost

        # Keyword pre-filter: skip embedding for keyword-driven queries
        skip_vec = self._is_keyword_sufficient(query_text) or query_intent == QueryIntent.NAVIGATIONAL
        if skip_vec:
            self.stats["vec_skips"] = self.stats.get("vec_skips", 0) + 1

        # --- Build the per-query retrieval context (replaces threading 8
        # shared containers + config through every phase signature) ---
        ctx = RetrievalContext(
            query_text=query_text, limit=limit, session_id=session_id,
            exclude_types=exclude_types, include_infrastructure=include_infrastructure,
            project_path=project_path, scope=scope, context_file=context_file,
            context_tags=context_tags, temporal_range=temporal_range,
            entity_id=entity_id, agent_type=agent_type, query_hint=query_hint,
            surfacing_context=surfacing_context, temporal_boost_only=temporal_boost_only,
            perspective=perspective, valid_at=valid_at, query_intent=query_intent,
            skip_vec=skip_vec, query_embedding=query_embedding,
            pw_vec=pw_vec, pw_text=pw_text, pw_word=pw_word, pw_ctx=pw_ctx, pw_graph=pw_graph,
            ctx_min_vec=ctx_min_vec, ctx_min_text=ctx_min_text, ctx_min_composite=ctx_min_composite,
            cache_key=_cache_key, now_mono=now_mono,
        )

        # Seed with hot cache results (#2) — bypass RRF, merge later
        for hr in hot_results:
            ctx.all_results[hr.id] = hr
            ctx.node_scores[hr.id] = hr.relevance * 0.8

        # Phase 1: Vector similarity search
        ctx.query_emb = self._query_phase_vec(ctx)

        # Phase 2: FTS5 text search + temporal retrieval
        self._query_phase_fts(ctx)

        # Phase 2.5: Strong signal short-circuit (QMD-inspired)
        # When FTS5 finds a slam-dunk match, skip vector/reranker phases.
        if (
            not ctx.skip_vec
            and len(ctx.text_ranked) >= 2
            and ctx.text_ranked[0][1] >= STRONG_SIGNAL_THRESHOLD
            and (ctx.text_ranked[0][1] - ctx.text_ranked[1][1]) >= STRONG_SIGNAL_GAP
        ):
            self.stats["strong_signal_shortcuts"] = self.stats.get("strong_signal_shortcuts", 0) + 1
            # Knowledge-fit (fork): apply the same metadata scoring as the
            # fusion phase. Raw text scores let a lexically-sharp
            # task_completion (weight 1.4) outrank a decision/lesson (2.0)
            # that would have won under fusion — the shortcut must not
            # bypass type weighting, feedback, priority, or decay.
            for nid, score in ctx.text_ranked:
                _node = ctx.all_results.get(nid)
                if _node is not None:
                    score *= self._metadata_score_factor(_node)
                ctx.node_scores[nid] = max(ctx.node_scores.get(nid, 0), score)
            self._query_phase_filter(ctx)
            # Still run Phase 5 for agent_type filtering and contextual boosts
            self._query_phase_boost(ctx)
            _result, _conf = self._query_phase_assemble(ctx)
            # Strong-signal path: skip adaptive retry (already high confidence)
            self._record_timing("read", (_time.monotonic() - _t0_agency) * 1000)
            return _result

        # Phase 2.7: LLM-based query expansion (QMD-inspired). OPT-IN via
        # CAIRN_QUERY_EXPANSION=1; no-ops keyless. It is a synchronous
        # cloud-LLM call at temperature 0.3 inside the read path, which
        # makes retrieval nondeterministic and latency-coupled to the
        # provider — and the 2026-07 paired LongMemEval A/B measured its
        # recall benefit as inside noise (McNemar p=0.29 over 500 qs).
        # Generates semantic variants for vague queries to improve recall.
        # Skipped for ambient surfacing (surfacing_context set): the surfacing
        # hook runs on every file edit and must stay in the tens-of-ms range —
        # an LLM round-trip (600ms-3s) there is the exact latency failure that
        # got hooks disabled historically. Explicit queries keep expansion.
        if expand_query and surfacing_context is None:
            self._query_phase_expand(ctx)

        # Phase 2.6: Entity-match channel (tags/fact terms/project/entity_id).
        # After the strong-signal shortcut so slam-dunk exits don't pay for
        # it; the shortcut intentionally bypasses fusion entirely.
        try:
            ctx.entity_ranked.extend(
                self._entity_term_search(ctx.query_text, ctx.limit, ctx.all_results)
            )
        except Exception as e:
            logger.debug("Entity term channel failed: %s", e)

        # Phase 3: RRF score fusion + metadata scoring + word/tag overlap
        self._query_phase_fusion(ctx)

        # Phase 4: Filter expired, superseded, flagged, infrastructure, scoped
        self._query_phase_filter(ctx)

        # Phase 5: Contextual boosting + entity/agent filtering
        self._query_phase_boost(ctx)

        # Phase 6: Graph expansion + cross-encoder reranking
        self._query_phase_rerank(ctx)

        # Phase 7: Assembly (sort, dedup, abstention, normalize, cache, track)
        _result, _conf = self._query_phase_assemble(ctx)

        # Phase 7.5: Adaptive retry — if confidence is low, retry with relaxed params.
        # Only retry when we got *some* results but they're low-quality.
        # If abstention produced zero results, that's intentional — don't override.
        if (
            _result
            and _conf < ADAPTIVE_RETRY_THRESHOLD
            and ADAPTIVE_RETRY_THRESHOLD > 0
            and not _is_retry
        ):
            retry_result = self._adaptive_retry_query(
                query_text=query_text,
                limit=limit,
                session_id=session_id,
                use_cache=False,
                expand_query=expand_query,
                exclude_types=exclude_types,
                include_infrastructure=include_infrastructure,
                project_path=project_path,
                scope=scope,
                context_file=context_file,
                context_tags=context_tags,
                entity_id=entity_id,
                agent_type=agent_type,
                surfacing_context=surfacing_context,
                temporal_boost_only=temporal_boost_only,
                perspective=perspective,
                valid_at=valid_at,
                original_confidence=_conf,
                ctx_min_vec=ctx_min_vec,
                ctx_min_text=ctx_min_text,
                ctx_min_composite=ctx_min_composite,
            )
            if retry_result is not None:
                _result = retry_result

        self._record_timing("read", (_time.monotonic() - _t0_agency) * 1000)
        return _result



    # ------------------------------------------------------------------
    # query() phase methods — extracted for readability
    # ------------------------------------------------------------------

    def _query_phase_vec(self, ctx: "RetrievalContext") -> Optional[List[float]]:
        """Phase 1: Vector similarity search with batch hydration."""
        query_text = ctx.query_text
        skip_vec = ctx.skip_vec
        entity_id = ctx.entity_id
        limit = ctx.limit
        all_results = ctx.all_results
        vec_ranked = ctx.vec_ranked
        raw_vec_sims = ctx.raw_vec_sims
        query_embedding = ctx.query_embedding
        query_emb: Optional[List[float]] = None
        if self._vec_available and not skip_vec:
            try:
                from cairn.embedding import generate_embedding, is_embedding_degraded

                query_emb = query_embedding or generate_embedding(query_text, mode="query")
                if is_embedding_degraded() and not getattr(self, "_hash_fallback_warned", False):
                    logger.warning(
                        "Query using hash-fallback embeddings — vector results will be low quality. "
                        "Check ONNX model installation."
                    )
                    self._hash_fallback_warned = True
                if query_emb:
                    vec_mult = 5
                    vec_limit = max(limit * vec_mult, self._MIN_VEC_CANDIDATES)
                    vec_results = self._vec_query(query_emb, limit=vec_limit)
                    if vec_results:
                        # Batch hydration: single WHERE IN query instead of N individual SELECTs
                        rowids = [rowid for rowid, _ in vec_results]
                        distances = {rowid: dist for rowid, dist in vec_results}
                        placeholders = ",".join("?" * len(rowids))
                        if entity_id:
                            rows = self._conn.execute(
                                f"""SELECT id, node_id, content, metadata, created_at,
                                           access_count, last_accessed, ttl_seconds
                                    FROM memories WHERE id IN ({placeholders})
                                    AND (entity_id = ? OR entity_id IS NULL)""",
                                (*rowids, entity_id),
                            ).fetchall()
                        else:
                            rows = self._conn.execute(
                                f"""SELECT id, node_id, content, metadata, created_at,
                                           access_count, last_accessed, ttl_seconds
                                    FROM memories WHERE id IN ({placeholders})""",
                                rowids,
                            ).fetchall()
                        for row in rows:
                            db_rowid = row[0]
                            result = self._row_to_result(row[1:])
                            similarity = 1.0 - distances[db_rowid]
                            if similarity < 0.1:
                                continue
                            result.relevance = similarity
                            raw_vec_sims[result.id] = similarity
                            all_results[result.id] = result
                            vec_ranked.append((result.id, similarity))
            except Exception as e:
                logger.debug(f"Vector search failed: {e}")
        return query_emb

    def _query_phase_fts(self, ctx: "RetrievalContext") -> None:
        """Phase 2: FTS5 text search + temporal retrieval channel."""
        query_text = ctx.query_text
        temporal_range = ctx.temporal_range
        entity_id = ctx.entity_id
        limit = ctx.limit
        all_results = ctx.all_results
        text_ranked = ctx.text_ranked
        temporal_ranked = ctx.temporal_ranked
        text_mult = 4 if temporal_range else 3
        text_results = self._text_search(query_text, limit=limit * text_mult, entity_id=entity_id)
        for result in text_results:
            if result.id not in all_results:
                all_results[result.id] = result
            text_ranked.append((result.id, result.relevance))

        # Temporal retrieval channel (P4) — date-proximity scoring
        if temporal_range:
            try:
                t_start, t_end = temporal_range
                temporal_ranked.extend(
                    self._temporal_search(
                        t_start, t_end, limit=limit * 3, entity_id=entity_id,
                    )
                )
                # Batch-hydrate any new results (was one SELECT per node)
                _missing = [nid for nid, _s in temporal_ranked if nid not in all_results]
                if _missing:
                    _ph = ",".join("?" * len(_missing))
                    for row in self._conn.execute(
                        f"""SELECT node_id, content, metadata, created_at,
                                   access_count, last_accessed, ttl_seconds
                            FROM memories WHERE node_id IN ({_ph})""",
                        _missing,
                    ).fetchall():
                        all_results[row[0]] = self._row_to_result(row)
            except Exception as e:
                logger.debug("Temporal retrieval channel failed: %s", e)

    # Entity-match channel weight in RRF fusion — modest vs vec/text so a
    # tag/project hit reinforces but never dominates dual-channel matches.
    _ENTITY_CHANNEL_WEIGHT = 0.8

    _ENTITY_STOPWORDS = frozenset({
        "the", "and", "for", "with", "that", "this", "from", "what", "how",
        "why", "when", "where", "who", "which", "was", "were", "are", "has",
        "have", "had", "not", "but", "all", "any", "can", "could", "should",
        "would", "will", "into", "onto", "about", "over", "under", "then",
        "than", "them", "they", "their", "there", "here", "our", "your",
        "its", "it's", "did", "does", "doing", "done", "use", "used", "using",
    })

    def _entity_term_search(
        self, query_text: str, limit: int,
        all_results: Dict[str, "MemoryResult"],
    ) -> List[Tuple[str, float]]:
        """Structured-field match channel: query tokens against metadata tags
        (which include extracted fact terms), the project column, and
        entity_id. Pure SQL + string ops — no LLM, no embedding — so it is
        safe under ambient surfacing latency budgets. Hydrates unseen rows
        into *all_results* like the other channel phases.

        Returns [(node_id, score)] ranked by hit count; empty list when the
        query yields no candidate entity tokens (channel skipped).
        Replaces the removed Pro entity-graph expansion.
        """
        import re as _re

        tokens = []
        for tok in _re.split(r"[^\w./-]+", query_text.lower()):
            tok = tok.strip("./-")
            if len(tok) >= 3 and tok not in self._ENTITY_STOPWORDS:
                tokens.append(tok)
            if len(tokens) >= 8:
                break
        if not tokens:
            return []

        # One scan, per-token hit count. Quoted-token LIKE anchors matches to
        # JSON string boundaries in metadata (tags are lowercased at
        # extraction); project/entity_id get exact-ish column matches.
        # COALESCE every nullable column: `1 + NULL` is NULL in SQLite, so a
        # single NULL term would zero out the whole hits sum.
        hit_exprs = []
        params: List[Any] = []
        for tok in tokens:
            esc = tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            hit_exprs.append("(COALESCE(metadata, '') LIKE ? ESCAPE '\\')")
            params.append(f'%"{esc}"%')
        hit_exprs.append("(COALESCE(project, '') LIKE '%' || ? || '%')")
        params.append(tokens[0])
        hit_exprs.append("(COALESCE(entity_id, '') = ?)")
        params.append(tokens[0])

        sql = f"""
            SELECT node_id, content, metadata, created_at,
                   access_count, last_accessed, ttl_seconds, hits
            FROM (SELECT node_id, content, metadata, created_at,
                         access_count, last_accessed, ttl_seconds,
                         ({" + ".join(hit_exprs)}) AS hits
                  FROM memories
                  WHERE (status IS NULL OR status NOT IN ('superseded', 'archived')))
            WHERE hits > 0
            ORDER BY hits DESC, created_at DESC
            LIMIT ?
        """
        params.append(limit * 3)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except Exception as e:
            logger.debug("Entity term channel failed: %s", e)
            return []

        ranked: List[Tuple[str, float]] = []
        for row in rows:
            score = min(1.0, row[7] / len(tokens))
            ranked.append((row[0], score))
            if row[0] not in all_results:
                all_results[row[0]] = self._row_to_result(row[:7])
        return ranked

    _EXPANSION_WEIGHT_DISCOUNT = 0.8  # Expanded variants are weighted down vs original

    def _query_phase_expand(self, ctx: "RetrievalContext") -> None:
        """Phase 2.7: LLM-based query expansion (opt-in).

        Generates lexical and vector variants of the query via a fast LLM,
        then runs additional FTS5/vector searches for each variant. Results
        are merged into the existing ranked lists with a weight discount.
        """
        query_text = ctx.query_text
        query_intent = ctx.query_intent
        skip_vec = ctx.skip_vec
        entity_id = ctx.entity_id
        limit = ctx.limit
        all_results = ctx.all_results
        vec_ranked = ctx.vec_ranked
        text_ranked = ctx.text_ranked
        raw_vec_sims = ctx.raw_vec_sims
        from cairn.query_expansion import is_expansion_enabled

        if not is_expansion_enabled():
            return
        if self._is_keyword_sufficient(query_text):
            return
        if getattr(self, "_in_decomposition", False):
            return
        if query_intent == QueryIntent.NAVIGATIONAL:
            return
        if query_intent == QueryIntent.FACTUAL:
            # Only skip expansion for factual queries with specific technical terms
            # (e.g., "What embedding model does Cairn use?" has "Cairn").
            # Vague factual queries like "What is the user's location?" need expansion
            # because the stored content uses different vocabulary ("Timezone Asia/Singapore").
            if self._TECH_TERM_RE.search(query_text) or self._ENTITY_RE.search(query_text):
                return

        try:
            from cairn.query_expansion import expand_query

            include_hyde = query_intent == QueryIntent.CONCEPTUAL
            expansion = expand_query(query_text, include_hyde=include_hyde)

            lex_variants = expansion.get("lex", [])[:2]  # Cap at 2 FTS searches
            vec_variants = expansion.get("vec", [])[:1]  # Cap at 1 vec variant (embedding is expensive)
            hyde_passage = expansion.get("hyde", "")
            # Skip HyDE for short queries (<5 words) — passage quality is low
            if hyde_passage and len(query_text.split()) < 5:
                hyde_passage = ""

            discount = self._EXPANSION_WEIGHT_DISCOUNT

            # Run FTS5 for each lexical variant
            for variant in lex_variants:
                if not variant or variant.strip() == query_text.strip():
                    continue
                try:
                    var_results = self._text_search(
                        variant, limit=limit * 2, entity_id=entity_id,
                    )
                    for result in var_results:
                        if result.id not in all_results:
                            all_results[result.id] = result
                        text_ranked.append((result.id, result.relevance * discount))
                except Exception:
                    pass

            # Run vector search for each vec variant + HyDE passage
            # Uses batch embedding to generate all variant embeddings in one call
            if self._vec_available and not skip_vec:
                from cairn.embedding import generate_embeddings_batch

                search_texts = [v for v in vec_variants if v and v.strip() != query_text.strip()]

                if search_texts or (hyde_passage and include_hyde):
                    try:
                        # Query variants embed as queries; the HyDE passage
                        # mimics a stored document, so it gets the doc prefix.
                        embeddings = generate_embeddings_batch(search_texts, mode="query")
                        if hyde_passage and include_hyde:
                            embeddings = embeddings + generate_embeddings_batch(
                                [hyde_passage], mode="document"
                            )
                    except Exception:
                        embeddings = []

                    for emb in embeddings:
                        if not emb:
                            continue
                        try:
                            vec_limit = max(limit * 3, self._MIN_VEC_CANDIDATES)
                            vec_results = self._vec_query(emb, limit=vec_limit)
                            if vec_results:
                                rowids = [rowid for rowid, _ in vec_results]
                                distances = {rowid: dist for rowid, dist in vec_results}
                                placeholders = ",".join("?" * len(rowids))
                                if entity_id:
                                    rows = self._conn.execute(
                                        f"""SELECT id, node_id, content, metadata, created_at,
                                                   access_count, last_accessed, ttl_seconds
                                            FROM memories WHERE id IN ({placeholders})
                                            AND (entity_id = ? OR entity_id IS NULL)""",
                                        (*rowids, entity_id),
                                    ).fetchall()
                                else:
                                    rows = self._conn.execute(
                                        f"""SELECT id, node_id, content, metadata, created_at,
                                                   access_count, last_accessed, ttl_seconds
                                            FROM memories WHERE id IN ({placeholders})""",
                                        rowids,
                                    ).fetchall()
                                for row in rows:
                                    db_rowid = row[0]
                                    result = self._row_to_result(row[1:])
                                    similarity = 1.0 - distances[db_rowid]
                                    if similarity < 0.1:
                                        continue
                                    discounted = similarity * discount
                                    if result.id not in all_results:
                                        result.relevance = discounted
                                        all_results[result.id] = result
                                    if result.id not in raw_vec_sims:
                                        raw_vec_sims[result.id] = discounted
                                    vec_ranked.append((result.id, discounted))
                        except Exception:
                            pass

            expanded_count = len(lex_variants) + len(vec_variants) + (1 if hyde_passage else 0)
            if expanded_count:
                self.stats["query_expansions"] = self.stats.get("query_expansions", 0) + 1
        except Exception as e:
            logger.debug("Query expansion phase failed: %s", e)

    def _query_phase_fusion(
        self, ctx: "RetrievalContext"
    ) -> None:
        """Phase 3: Reciprocal Rank Fusion + metadata scoring + word/preference boosts."""
        query_text = ctx.query_text
        all_results = ctx.all_results
        node_scores = ctx.node_scores
        vec_ranked = ctx.vec_ranked
        text_ranked = ctx.text_ranked
        temporal_ranked = ctx.temporal_ranked
        entity_ranked = ctx.entity_ranked
        pw_vec = ctx.pw_vec
        pw_text = ctx.pw_text
        pw_word = ctx.pw_word
        pw_ctx = ctx.pw_ctx
        perspective = ctx.perspective
        # RRF fusion
        rrf_channels = [vec_ranked, text_ranked]
        rrf_weights = [pw_vec, pw_text]
        if temporal_ranked:
            rrf_channels.append(temporal_ranked)
            rrf_weights.append(1.2)  # Temporal channel weight
        if entity_ranked:
            rrf_channels.append(entity_ranked)
            rrf_weights.append(self._ENTITY_CHANNEL_WEIGHT)

        rrf_scores = self._rrf_fuse(rrf_channels, weights=rrf_weights)

        # Apply metadata factors on RRF base scores
        for nid, rrf_score in rrf_scores.items():
            if nid not in all_results:
                continue
            node = all_results[nid]
            event_type = node.metadata.get("event_type", "")
            score = rrf_score * self._metadata_score_factor(node)
            # Apply perspective-based type boost (behavioral diversity)
            if perspective and perspective in self._PERSPECTIVE_BOOSTS:
                score *= self._PERSPECTIVE_BOOSTS[perspective].get(event_type, 1.0)
            # Thompson sampling boost (outcome-correlated learning)
            score *= self._get_thompson_boost(event_type)
            # Consolidation quality boost (compacted knowledge nodes)
            cq = node.metadata.get("consolidation_quality", 0)
            if cq > 0:
                score *= 1.0 + min(cq, 3.0) * 0.1  # up to 1.3x
            # Merge with hot cache scores (take max)
            node_scores[nid] = max(node_scores.get(nid, 0.0), score)

        # Word/tag overlap boost
        _query_words = [w for w in query_text.lower().split() if len(w) > 2]
        if _query_words:
            for nid in list(node_scores.keys()):
                node = all_results[nid]
                content_lower = node.content.lower()
                tag_text = " ".join(str(t).lower() for t in (node.metadata.get("tags") or []))
                searchable = content_lower + " " + tag_text
                word_ratio = self._word_overlap(_query_words, searchable)
                if word_ratio > 0:
                    # Dampen boost for negatively-rated memories so outdated
                    # facts can't use word overlap to outrank updated versions
                    fb = node.metadata.get("feedback_score", 0)
                    fb_mod = 0.5 if fb < 0 else 1.0
                    node_scores[nid] *= 1.0 + word_ratio * 0.5 * fb_mod * pw_word

        # Preference signal boost
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
                    node_scores[nid] *= 1.5  # Extra boost for preference matches

    def _query_phase_filter(self, ctx: "RetrievalContext") -> None:
        """Phase 4: Filter expired, superseded, flagged, infrastructure, and scoped results."""
        all_results = ctx.all_results
        node_scores = ctx.node_scores
        exclude_types = ctx.exclude_types
        include_infrastructure = ctx.include_infrastructure
        session_id = ctx.session_id
        project_path = ctx.project_path
        scope = ctx.scope
        valid_at = ctx.valid_at
        # Filter expired
        for nid in list(all_results.keys()):
            if all_results[nid].is_expired():
                del all_results[nid]
                node_scores.pop(nid, None)

        # Filter superseded and archived (archived: rollup raw sources kept on
        # disk for fidelity — reachable via derived_from edges, never surfaced)
        for nid in list(all_results.keys()):
            _m = all_results[nid].metadata
            if _m.get("superseded") or _m.get("archived"):
                del all_results[nid]
                node_scores.pop(nid, None)

        # Bi-temporal point-in-time filter
        if valid_at and all_results:
            nids = list(all_results.keys())
            # Batch query: find which node_ids are NOT valid at the given point in time
            placeholders = ",".join("?" * len(nids))
            invalid_rows = self._conn.execute(
                f"""SELECT node_id FROM memories
                    WHERE node_id IN ({placeholders})
                    AND (
                        (valid_from IS NOT NULL AND valid_from > ?)
                        OR (valid_until IS NOT NULL AND valid_until <= ?)
                    )""",
                (*nids, valid_at, valid_at),
            ).fetchall()
            invalid_nids = {r[0] for r in invalid_rows}
            for nid in invalid_nids:
                if nid in all_results:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Filter flagged-for-review (negative feedback threshold reached)
        for nid in list(all_results.keys()):
            if all_results[nid].metadata.get("flagged_for_review"):
                del all_results[nid]
                node_scores.pop(nid, None)

        # Filter infrastructure types
        excluded = set(exclude_types) if exclude_types else set()
        if not include_infrastructure:
            excluded |= self._INFRASTRUCTURE_TYPES
        if excluded:
            for nid in list(all_results.keys()):
                etype = all_results[nid].metadata.get("event_type", "")
                if etype in excluded:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Session filter
        if session_id:
            for nid in list(all_results.keys()):
                node_session = all_results[nid].metadata.get("session_id", "")
                if node_session and node_session != session_id:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Project filter
        if project_path and scope == "project":
            for nid in list(all_results.keys()):
                node_project = all_results[nid].metadata.get("project", "")
                if node_project and node_project != project_path:
                    del all_results[nid]
                    node_scores.pop(nid, None)

        # Session scope filter: restrict to caller's session only
        if session_id and scope == "session":
            for nid in list(all_results.keys()):
                node_session = all_results[nid].metadata.get("session_id", "")
                if node_session and node_session != session_id:
                    del all_results[nid]
                    node_scores.pop(nid, None)

    def _query_phase_boost(self, ctx: "RetrievalContext") -> None:
        """Phase 5: Contextual re-ranking, cluster boost, temporal constraint, entity/agent filter."""
        query_text = ctx.query_text
        query_emb = ctx.query_emb
        all_results = ctx.all_results
        node_scores = ctx.node_scores
        context_file = ctx.context_file
        context_tags = ctx.context_tags
        temporal_range = ctx.temporal_range
        temporal_boost_only = ctx.temporal_boost_only
        pw_ctx = ctx.pw_ctx
        entity_id = ctx.entity_id
        agent_type = ctx.agent_type
        # Contextual re-ranking
        if context_file or context_tags:
            context_set: Set[str] = set()
            if context_file:
                # Extract filename stem and path components as context signals
                from pathlib import PurePosixPath

                p = PurePosixPath(context_file)
                context_set.add(p.stem.lower())
                context_set.add(p.name.lower())
                for part in p.parts:
                    if len(part) > 2 and part not in ("/", "."):
                        context_set.add(part.lower())
            if context_tags:
                context_set.update(t.lower() for t in context_tags)

            if context_set:
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    node_tags = set(str(t).lower() for t in (node.metadata.get("tags") or []))
                    node_project = (node.metadata.get("project") or "").lower()
                    node_content_lower = node.content.lower()

                    # Count context signal matches
                    tag_overlap = len(context_set & node_tags)
                    project_match = 1 if node_project and any(c in node_project for c in context_set) else 0
                    content_match = sum(1 for c in context_set if c in node_content_lower)

                    # Apply graduated boost: 10% per tag match, 15% for project, 5% per content hit (capped)
                    boost = 1.0 + ((tag_overlap * 0.10) + (project_match * 0.15) + (min(content_match, 3) * 0.05)) * pw_ctx
                    node_scores[nid] *= boost

        # Cluster co-boost
        try:
            if query_emb is not None:
                cluster_boosts, _ret_clusters = self._compute_cluster_boosts(query_emb)
                if cluster_boosts:
                    # Build node_id -> cluster_id mapping from cluster member lists
                    _nid_to_cid: Dict[str, int] = {}
                    for _rc in _ret_clusters:
                        _cid = _rc["cluster_id"]
                        if _cid in cluster_boosts:
                            for _mid in _rc.get("member_node_ids", []):
                                _nid_to_cid[_mid] = _cid
                    for nid in list(node_scores.keys()):
                        _mem_cid = _nid_to_cid.get(nid)
                        if _mem_cid is not None and _mem_cid in cluster_boosts:
                            node_scores[nid] *= cluster_boosts[_mem_cid]
        except Exception as e:
            logger.debug("Clustering boost failed: %s", e)

        # Temporal constraint — in-range boost, out-of-range penalty
        if temporal_range:
            try:
                t_start, t_end = temporal_range
                for nid in list(node_scores.keys()):
                    node = all_results[nid]
                    # Prefer referenced_date (explicit event time), fall back to created_at
                    ref_date = node.metadata.get("referenced_date") or ""
                    if not ref_date and node.created_at:
                        ref_date = node.created_at.isoformat() if hasattr(node.created_at, "isoformat") else str(node.created_at)
                    if ref_date:
                        if t_start <= ref_date <= t_end:
                            node_scores[nid] *= 1.3  # In-range boost
                        elif temporal_boost_only:
                            node_scores[nid] *= 0.85  # Soft penalty for auto-inferred ranges
                        else:
                            # Hard penalty only when the memory has an explicit event date;
                            # soft penalty when falling back to created_at (uncertain proxy)
                            has_explicit_date = bool(node.metadata.get("referenced_date"))
                            if has_explicit_date:
                                node_scores[nid] *= 0.15  # Explicit date, wrong range
                            else:
                                node_scores[nid] *= 0.70  # Inferred date, uncertain
                    # No date at all: leave score unchanged (neutral)
            except Exception as e:
                logger.debug("Temporal constraint failed: %s", e)

        # Entity filtering (post-scoring, same pattern as project/event_type filters)
        if entity_id:
            filtered_ids = set()
            # Batch lookup: collect IDs that need DB fallback
            needs_lookup = []
            meta_resolved = {}  # nid -> entity from metadata
            for nid, node in all_results.items():
                node_entity = None
                if hasattr(node, "metadata") and node.metadata:
                    node_entity = node.metadata.get("entity_id")
                if node_entity is not None:
                    meta_resolved[nid] = node_entity
                else:
                    needs_lookup.append(nid)
            # Single batch query for unresolved nodes
            if needs_lookup:
                try:
                    placeholders = ",".join("?" * len(needs_lookup))
                    rows = self._conn.execute(
                        f"SELECT node_id, entity_id FROM memories WHERE node_id IN ({placeholders})",
                        needs_lookup,
                    ).fetchall()
                    for row_nid, row_eid in rows:
                        meta_resolved[row_nid] = row_eid
                except Exception as e:
                    logger.debug("Batch entity lookup failed: %s", e)
            for nid in all_results:
                resolved = meta_resolved.get(nid)
                if resolved == entity_id or resolved is None:
                    filtered_ids.add(nid)
            _filtered = {k: v for k, v in node_scores.items() if k in filtered_ids}
            node_scores.clear()
            node_scores.update(_filtered)

        # Agent type filtering (post-scoring, same pattern as entity_id)
        if agent_type:
            filtered_ids = set()
            needs_lookup = []
            meta_resolved = {}
            for nid, node in all_results.items():
                node_agent_type = None
                if hasattr(node, "metadata") and node.metadata:
                    node_agent_type = node.metadata.get("agent_type")
                if node_agent_type is not None:
                    meta_resolved[nid] = node_agent_type
                else:
                    needs_lookup.append(nid)
            if needs_lookup:
                try:
                    placeholders = ",".join("?" * len(needs_lookup))
                    rows = self._conn.execute(
                        f"SELECT node_id, agent_type FROM memories WHERE node_id IN ({placeholders})",
                        needs_lookup,
                    ).fetchall()
                    for row_nid, row_at in rows:
                        meta_resolved[row_nid] = row_at
                except Exception as e:
                    logger.debug("Batch agent_type lookup failed: %s", e)
            for nid in all_results:
                if meta_resolved.get(nid) == agent_type:
                    filtered_ids.add(nid)
            _filtered = {k: v for k, v in node_scores.items() if k in filtered_ids}
            node_scores.clear()
            node_scores.update(_filtered)

    def _query_phase_rerank(self, ctx: "RetrievalContext") -> None:
        """Phase 6: Graph expansion + cross-encoder reranking + plugin modifiers."""
        query_text = ctx.query_text
        all_results = ctx.all_results
        node_scores = ctx.node_scores
        limit = ctx.limit
        pw_graph = ctx.pw_graph
        # Multi-hop graph traversal with spreading activation (P6).
        _MAX_GRAPH_HOPS = 2
        _HOP_DECAY = 0.4  # Score multiplier per hop (0.4 for hop 1, 0.16 for hop 2)
        if node_scores and limit >= 3:
            try:
                top_ids = sorted(node_scores, key=node_scores.get, reverse=True)[:5]
                existing_ids = set(node_scores.keys())
                for seed_id in top_ids:
                    chain = self.get_related_chain(
                        seed_id, max_hops=_MAX_GRAPH_HOPS, min_weight=0.2,
                        exclude_ids=existing_ids, _include_results=True,
                    )
                    seed_score = node_scores[seed_id]
                    for entry in chain:
                        nbr_id = entry["node_id"]
                        if nbr_id in existing_ids:
                            continue
                        result = entry["_result"]
                        if (result.is_expired() or result.metadata.get("superseded")
                                or result.metadata.get("archived")):
                            continue
                        hop = entry["hop"]
                        weight = entry["weight"]
                        nbr_score = seed_score * (_HOP_DECAY ** hop) * min(weight, 1.0) * pw_graph
                        if nbr_score >= self._MIN_COMPOSITE_SCORE:
                            all_results[nbr_id] = result
                            node_scores[nbr_id] = nbr_score
                            existing_ids.add(nbr_id)
            except Exception as e:
                logger.debug("Graph multi-hop traversal failed: %s", e)

        # Cross-encoder reranking (P2) — rescore top candidates.
        # 20 (not 10): the position-aware weights below give rank 11+ the
        # aggressive 0.50 tier — with only 10 candidates that tier was
        # unreachable dead code and deep results never got reranked.
        _RERANK_CANDIDATES = 20
        if node_scores and len(node_scores) > 1:
            try:
                from cairn.reranker import cross_encoder_score

                top_ids_for_rerank = sorted(
                    node_scores, key=node_scores.get, reverse=True
                )[:_RERANK_CANDIDATES]
                passages = [all_results[nid].content for nid in top_ids_for_rerank]
                # P2: Include temporal metadata for date-aware reranking
                temporal_meta = []
                for nid in top_ids_for_rerank:
                    node = all_results[nid]
                    date_str = node.metadata.get("referenced_date", "")
                    if not date_str and node.created_at:
                        date_str = node.created_at.isoformat() if hasattr(node.created_at, "isoformat") else str(node.created_at)
                    temporal_meta.append(date_str or "")
                ce_scores = cross_encoder_score(
                    query_text, passages, temporal_metadata=temporal_meta,
                )
                if ce_scores is not None and len(ce_scores) == len(top_ids_for_rerank):
                    # Normalize CE scores to [0, 1] range
                    ce_min = min(ce_scores)
                    ce_max = max(ce_scores)
                    ce_range = ce_max - ce_min
                    if ce_range > 0:
                        ce_norm = [(s - ce_min) / ce_range for s in ce_scores]
                    else:
                        ce_norm = [0.5] * len(ce_scores)

                    # CE application mode — call-time env read so the eval
                    # A/B can flip it per-arm (CAIRN_QUERY_EXPANSION pattern).
                    _ce_mode = os.environ.get("CAIRN_CE_MODE", "hybrid")
                    if _ce_mode == "hybrid":
                        # Resort rank-1 only: the CE's single favorite takes
                        # the top fused score (capturing resort's MRR gain);
                        # everyone else keeps the gentle boost (protecting
                        # P@K, which full resort degraded in the A/B).
                        for i, nid in enumerate(top_ids_for_rerank):
                            ce_w = 0.15 if i < 3 else (0.30 if i < 10 else 0.50)
                            node_scores[nid] *= 1.0 + ce_w * ce_norm[i]
                        best_i = max(range(len(top_ids_for_rerank)),
                                     key=lambda i: ce_scores[i])
                        # Confidence margin on SIGMOID(raw logits) — min-max
                        # normalized scores stretch any tiny difference to a
                        # full-range margin, so they can't express indifference.
                        _sig = [1.0 / (1.0 + math.exp(-min(30.0, max(-30.0, sc))))
                                for sc in ce_scores]
                        _rest = [_sig[i] for i in range(len(_sig)) if i != best_i]
                        _margin = _sig[best_i] - max(_rest) if _rest else 0.0
                        # Only override rank 1 when the CE is CONFIDENT — an
                        # indifferent CE (tiny margin) must not trample the
                        # metadata knowledge weighting (priority/type order).
                        # Threshold is env-tunable because it is calibrated to
                        # a reranker's logit distribution: 0.10 was measured
                        # for ms-marco and will not transfer to another CE
                        # without a re-sweep.
                        try:
                            _ce_margin = float(os.environ.get("CAIRN_CE_MARGIN", "0.10"))
                        except ValueError:
                            _ce_margin = 0.10
                        if _margin >= _ce_margin:
                            top_score = max(node_scores[n] for n in top_ids_for_rerank)
                            node_scores[top_ids_for_rerank[best_i]] = top_score * 1.01
                    elif _ce_mode == "resort":
                        # Full re-sort: permute the EXISTING fused scores among
                        # the reranked candidates in CE order. The score
                        # multiset is unchanged (downstream abstention floors
                        # keep their meaning); only the ordering of the top-K
                        # changes, and the CE fully decides it.
                        existing = sorted(
                            (node_scores[nid] for nid in top_ids_for_rerank),
                            reverse=True,
                        )
                        ce_order = sorted(
                            range(len(top_ids_for_rerank)),
                            key=lambda i: ce_norm[i],
                            reverse=True,
                        )
                        for slot, i in enumerate(ce_order):
                            node_scores[top_ids_for_rerank[i]] = existing[slot]
                    else:
                        # Position-aware CE boost (QMD-inspired): top RRF
                        # results are already high-confidence from
                        # multi-channel fusion, so reranker has less override
                        # power. Lower-ranked results benefit more.
                        for i, nid in enumerate(top_ids_for_rerank):
                            if i < 3:
                                ce_w = 0.15   # Rank 1-3: preserve exact matches
                            elif i < 10:
                                ce_w = 0.30   # Rank 4-10: balanced
                            else:
                                ce_w = 0.50   # Rank 11+: trust reranker more
                            node_scores[nid] *= 1.0 + ce_w * ce_norm[i]
            except ImportError:
                pass  # reranker module not available
            except Exception as e:
                logger.debug("Cross-encoder reranking failed: %s", e)

        # Plugin score modifiers
        if self._score_modifiers and node_scores:
            for nid in list(node_scores.keys()):
                meta = all_results[nid].metadata if nid in all_results else {}
                for modifier in self._score_modifiers:
                    try:
                        node_scores[nid] = modifier(nid, node_scores[nid], meta)
                    except Exception as e:
                        logger.debug("Plugin score modifier failed: %s", e)

    def _adaptive_retry_query(
        self,
        query_text: str,
        limit: int,
        session_id: Optional[str],
        use_cache: bool,
        expand_query: bool,
        exclude_types: Optional[List[str]],
        include_infrastructure: bool,
        project_path: str,
        scope: str,
        context_file: str,
        context_tags: Optional[List[str]],
        entity_id: Optional[str],
        agent_type: Optional[str],
        surfacing_context: Optional[SurfacingContext],
        temporal_boost_only: bool,
        perspective: Optional[str],
        valid_at: Optional[str],
        original_confidence: float,
        ctx_min_vec: float,
        ctx_min_text: float,
        ctx_min_composite: float,
    ) -> Optional[List["MemoryResult"]]:
        """Retry query with relaxed parameters when confidence is low.

        Drops temporal_range and query_hint, relaxes abstention thresholds,
        and enlarges the candidate pool. Returns results only if retry
        confidence exceeds original confidence.
        """
        self.stats["adaptive_retries"] = self.stats.get("adaptive_retries", 0) + 1
        relaxation = ADAPTIVE_RETRY_RELAXATION

        retry_results = self.query(
            query_text=query_text,
            limit=limit,
            session_id=session_id,
            use_cache=False,
            expand_query=False,        # Skip expansion on retry — already done in first pass
            exclude_types=exclude_types,
            include_infrastructure=include_infrastructure,
            project_path=project_path,
            scope=scope,
            context_file=context_file,
            context_tags=context_tags,
            temporal_range=None,       # Drop temporal filter
            entity_id=entity_id,
            agent_type=agent_type,
            query_hint=None,           # Drop event_type hint
            surfacing_context=surfacing_context,
            temporal_boost_only=temporal_boost_only,
            perspective=perspective,
            valid_at=valid_at,
            _is_retry=True,            # Prevent infinite loops
        )

        if not retry_results:
            return None

        # Check if retry produced meaningfully better confidence.
        # Require both improvement AND a minimum quality floor to avoid
        # promoting off-topic results that only survived relaxed thresholds.
        retry_conf = sum(r.relevance for r in retry_results[:3]) / min(len(retry_results), 3)
        if retry_conf > original_confidence and retry_conf >= 0.15:
            return retry_results
        return None

    def _query_phase_assemble(self, ctx: "RetrievalContext") -> Tuple[List["MemoryResult"], float]:
        """Phase 7: Sort, dedup, abstention, normalize, cache, and track results.

        Returns (results, confidence) where confidence is avg top-3 relevance.
        """
        query_text = ctx.query_text
        all_results = ctx.all_results
        node_scores = ctx.node_scores
        raw_vec_sims = ctx.raw_vec_sims
        limit = ctx.limit
        _cache_key = ctx.cache_key
        now_mono = ctx.now_mono
        session_id = ctx.session_id
        query_hint = ctx.query_hint
        ctx_min_vec = ctx.ctx_min_vec
        ctx_min_text = ctx.ctx_min_text
        ctx_min_composite = ctx.ctx_min_composite
        from ._types import (
            _QUERY_CACHE_MAX,
            _SESSION_CACHE_MAX,
            _TRAILING_HASH_RE,
        )

        # Sort and dedup
        sorted_ids = sorted(node_scores.keys(), key=lambda x: node_scores[x], reverse=True)

        # Vec-top guard: the fused ranking can bury the vector channel's
        # clear winner under text-channel pile-on — the LongMemEval
        # preference oracle showed max(cosine, fused) R@5 0.933 vs fused
        # 0.800, i.e. the channels fail on DIFFERENT questions. Guarantee
        # the vec channel's top candidate survives into the final top-3
        # (never above the fused top-2, so exact-match rank-1 wins are
        # preserved). Abstention still applies downstream. Default ON:
        # +5/-0 on the 113-probe set (MRR 0.870 -> 0.888, hit rate 0.965
        # -> 0.991, 2026-07-25); CAIRN_VEC_TOP_GUARD=0 to disable.
        if os.environ.get("CAIRN_VEC_TOP_GUARD", "1") == "1":
            try:
                _depth = max(1, int(os.environ.get("CAIRN_VEC_GUARD_DEPTH", "1")))
            except ValueError:
                _depth = 1
            _vec_best = [n for n, _ in sorted(
                raw_vec_sims.items(), key=lambda kv: -kv[1])[:_depth]]
            self._apply_channel_top_guard(sorted_ids, node_scores, _vec_best, insert_at=2)
        # Text-top guard (experiment knob, default OFF — measured): symmetric
        # case where the text channel's #1 is right but fusion buries it.
        # Together with CAIRN_VEC_GUARD_DEPTH=2 this hits the LongMemEval
        # preference oracle exactly (0.867 -> 0.933), but the 113-probe
        # live-store eval vetoed both as defaults: +0/-2, hit rate 0.991 ->
        # 0.982 (2026-07-25). Long-document corpora can opt in via env;
        # short keyword-dense stores lose. Benchmark wins that cost real
        # retrieval quality don't ship.
        if os.environ.get("CAIRN_TEXT_TOP_GUARD", "0") == "1" and ctx.text_ranked:
            _seen_t: Dict[str, float] = {}
            for n, s in ctx.text_ranked:
                if n not in _seen_t or s > _seen_t[n]:
                    _seen_t[n] = s
            _text_best = [max(_seen_t, key=lambda n: _seen_t[n])]
            self._apply_channel_top_guard(sorted_ids, node_scores, _text_best, insert_at=3)

        seen_content: Set[str] = set()
        deduped: List[MemoryResult] = []
        for nid in sorted_ids:
            node = all_results[nid]
            normalized = " ".join(node.content.lower().split())[:150]
            normalized = _TRAILING_HASH_RE.sub("", normalized)
            if normalized in seen_content:
                continue
            seen_content.add(normalized)
            deduped.append(node)
            if len(deduped) >= limit:
                break

        # Semantic dedup — catch near-duplicates missed by exact string match
        if self._vec_available and len(deduped) > 1:
            _sem_threshold = float(os.environ.get("CAIRN_SEMANTIC_DEDUP_THRESHOLD", "0.92"))
            if _sem_threshold < 1.0:
                deduped = self._semantic_dedup(deduped, node_scores, _sem_threshold)

        # Abstention — filter low-quality results before normalization
        if deduped:
            # Precompute query words for text-result word-overlap check
            query_words = [w for w in query_text.lower().split() if len(w) > 2]
            filtered = []
            for n in deduped:
                score = node_scores.get(n.id, 0.0)
                # Universal composite floor (catches temporal penalty, etc.)
                if score < ctx_min_composite:
                    continue
                if n.id in raw_vec_sims:
                    # Vec result: require minimum cosine similarity (dynamic #4)
                    if raw_vec_sims[n.id] >= ctx_min_vec:
                        filtered.append(n)
                    elif query_words:
                        # Fallback: vec result below threshold can survive
                        # if content + tags have strong word overlap with query
                        content_lower = n.content.lower()
                        tag_text = " ".join(str(t).lower() for t in (n.metadata.get("tags") or []))
                        searchable = content_lower + " " + tag_text
                        if self._word_overlap(query_words, searchable) >= ctx_min_text:
                            filtered.append(n)
                else:
                    # Text-only result: require minimum raw word overlap (dynamic #4)
                    if query_words:
                        content_lower = n.content.lower()
                        if self._word_overlap(query_words, content_lower) >= ctx_min_text:
                            filtered.append(n)
                    else:
                        filtered.append(n)
            deduped = filtered

        # Normalize relevance scores
        if deduped:
            max_score = max(node_scores.get(n.id, 0.0) for n in deduped)
            for node in deduped:
                raw = node_scores.get(node.id, 0.0)
                node.relevance = round(raw / max_score, 3) if max_score > 0 else 0.0

        # Compute strength: composite of all scoring signals, normalized to [0, 1]
        if deduped:
            raw_strengths = {}
            for node in deduped:
                event_type = (node.metadata or {}).get("event_type", "")
                type_weight = self._TYPE_WEIGHTS.get(event_type, 1.0)
                fb_score = (node.metadata or {}).get("feedback_score", 0)
                fb_factor = self._compute_fb_factor(fb_score)
                _la = node.last_accessed.isoformat() if node.last_accessed else None
                _ca = node.created_at.isoformat() if node.created_at else None
                decay = self._compute_decay_factor(event_type, _la, _ca, node.access_count or 0)
                raw_strengths[node.id] = node.relevance * type_weight * fb_factor * decay
            max_strength = max(raw_strengths.values()) if raw_strengths else 1.0
            for node in deduped:
                node.strength = round(raw_strengths[node.id] / max_strength, 3) if max_strength > 0 else 0.0

        if deduped:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1

        # Compute confidence (always, not just for cache)
        _confidence = 0.0
        if deduped:
            _confidence = sum(n.relevance for n in deduped[:3]) / min(len(deduped), 3)

        # --- Query result cache: store (tiered TTL #2) ---
        if _cache_key is not None:
            with self._cache_lock:
                self._query_cache[_cache_key] = (now_mono, deduped, _confidence)
                while len(self._query_cache) > _QUERY_CACHE_MAX:
                    self._query_cache.popitem(last=False)
        if session_id and deduped:
            self._session_cache[session_id] = deduped
            if len(self._session_cache) > _SESSION_CACHE_MAX:
                try:
                    oldest = next(iter(self._session_cache))
                    del self._session_cache[oldest]
                except StopIteration:
                    pass

        # --- A/B feedback tracking: record retrieval context for returned results ---
        with self._cache_lock:
            for n in deduped:
                self._recent_query_context[n.id] = {
                    "query_text": query_text[:200],
                    "query_hint": query_hint,
                    "score": round(node_scores.get(n.id, 0.0), 4),
                    "vec_sim": round(raw_vec_sims.get(n.id, 0.0), 4),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._recent_query_context.move_to_end(n.id)
            while len(self._recent_query_context) > self._QUERY_CONTEXT_MAX:
                self._recent_query_context.popitem(last=False)

        # --- Batch access_count + retrieval_count increment for returned results ---
        if deduped:
            _returned_ids = [n.id for n in deduped]
            _now = datetime.now(timezone.utc).isoformat()
            try:
                ph = ",".join("?" * len(_returned_ids))
                self._conn.execute(
                    f"UPDATE memories SET access_count = access_count + 1, "
                    f"retrieval_count = COALESCE(retrieval_count, 0) + 1, "
                    f"last_accessed = ? WHERE node_id IN ({ph})",
                    [_now] + _returned_ids,
                )
                self._commit()
            except Exception:
                logger.debug("access_count batch update failed", exc_info=True)

        # --- Durable query log: real-query corpus for retrieval eval ---
        # Deliberately outside `if deduped:` — zero-result queries are signal.
        # Call-time env read so tests/eval runs can disable per-case.
        if os.environ.get("CAIRN_QUERY_LOG", "1") != "0":
            try:
                _sc = ctx.surfacing_context
                _sc_str = (
                    str(getattr(_sc, "value", _sc)) if _sc is not None else None
                )
                self._conn.execute(
                    "INSERT INTO query_log (ts, query_text, surfacing_context, "
                    "session_id, result_count, top_score) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        query_text[:500],
                        _sc_str,
                        session_id,
                        len(deduped),
                        round(max(node_scores.values()), 4) if node_scores else 0.0,
                    ),
                )
                self._commit()
            except Exception:
                logger.debug("query_log insert failed", exc_info=True)

        # Annotate results with embedding backend for transparency
        try:
            from cairn.embedding import get_active_backend
            _backend = get_active_backend() or "hash-fallback"
            for r in deduped:
                if r.metadata is not None:
                    r.metadata["_embedding_backend"] = _backend
        except ImportError:
            pass

        # Annotate results with query confidence
        for r in deduped:
            if r.metadata is not None:
                r.metadata["_query_confidence"] = round(_confidence, 3)

        return deduped, _confidence



    def _compute_cluster_boosts(
        self, query_embedding: List[float],
    ) -> Tuple[Dict[int, float], List[dict]]:
        """Compute boost factors for clusters whose centroids are close to the query.

        Returns (boosts, clusters) where boosts is {cluster_id: boost_factor}
        (1.05-1.15x for cosine similarity > 0.5) and clusters is the raw
        cluster list from get_clusters_for_retrieval (avoids a second DB fetch).
        """
        global _PATTERN_LEARNER_UNAVAILABLE
        if _PATTERN_LEARNER_UNAVAILABLE:
            return {}, []
        try:
            from cairn.pattern_learner import PatternLearner
        except ImportError:
            _PATTERN_LEARNER_UNAVAILABLE = True
            return {}, []
        try:
            learner = PatternLearner(store=self)
            clusters = learner.get_clusters_for_retrieval()
        except Exception as e:
            logger.debug("Cluster retrieval failed: %s", e)
            return {}, []

        boosts: Dict[int, float] = {}
        for cluster in clusters:
            centroid = cluster.get("centroid")
            if centroid is None:
                continue
            similarity = _cosine_similarity(query_embedding, centroid)
            if similarity > 0.5:
                # Scale: 0.5 sim -> 1.05x, 0.8 sim -> 1.15x (capped)
                boost = 1.0 + min((similarity - 0.5) * 0.33, 0.15)
                boosts[cluster["cluster_id"]] = boost
        return boosts, clusters

    # Pre-compiled regex for keyword detection
    _CAMELCASE_RE = re.compile(r'\b[A-Z][a-z]+[A-Z]\w*\b')
    _FILEPATH_RE = re.compile(r'/[a-zA-Z][\w/.]*')
    _LITERAL_PATTERNS_RE = re.compile(
        r'(?:'
        r'\b[a-z]+_[a-z_]+\b'          # snake_case identifiers (cairn_store, query_hint)
        r'|\bmem-[0-9a-f]{8,}\b'       # memory IDs (mem-4c9e8659)
        r'|\bv?\d+\.\d+(?:\.\d+)?\b'   # version strings (v0.10.2, 1.0.0)
        r'|https?://'                   # URLs
        r'|\b[0-9a-f]{8,}\b'           # hex strings 8+ chars (commit SHAs, UUIDs)
        r')'
    )

    @staticmethod
    def _is_keyword_sufficient(query_text: str) -> bool:
        """Detect if a query is keyword-driven enough to skip vector embedding.

        Conservative heuristics — only skips when the query clearly contains
        code identifiers, file paths, or quoted phrases where semantic search
        adds latency without improving results.
        """
        # Contains backticks (code spans)
        if '`' in query_text:
            return True
        # Contains file paths (/foo/bar or ./baz)
        if QueryMixin._FILEPATH_RE.search(query_text):
            return True
        # Quoted phrase search
        stripped = query_text.strip()
        if stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 2:
            return True
        # CamelCase identifiers (e.g., SQLiteStore, MemoryResult)
        if QueryMixin._CAMELCASE_RE.search(query_text):
            return True
        # Literal patterns: snake_case, memory IDs, versions, URLs, hex strings
        if QueryMixin._LITERAL_PATTERNS_RE.search(query_text):
            return True
        return False

    # Regex patterns for keyword extraction (P5)
    _ENTITY_RE = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')  # Multi-word proper nouns
    _DATE_RE = re.compile(
        r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|'
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:,?\s+\d{4})?)\b',
        re.IGNORECASE,
    )
    _NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\s*(?:%|percent|dollars?|USD|EUR|GB|MB|TB|KB|pp)(?:\b|(?=\s|$))')
    _TECH_TERM_RE = re.compile(r'\b(?:[A-Z][a-z]*[A-Z]\w*|[a-z]+_[a-z_]+|[A-Z]{2,}(?:\d+)?)\b')

    @staticmethod
    def _extract_keywords(content: str) -> str:
        """Extract key entities, dates, numbers, and technical terms from content (P5).

        Lightweight regex-based extraction (no LLM). Returns space-separated
        keywords suitable for appending to FTS5 index content.
        """
        keywords: List[str] = []

        # Multi-word proper nouns (e.g., "John Smith", "New York")
        keywords.extend(m.group() for m in QueryMixin._ENTITY_RE.finditer(content))

        # Dates in various formats
        keywords.extend(m.group() for m in QueryMixin._DATE_RE.finditer(content))

        # Numbers with units
        keywords.extend(m.group() for m in QueryMixin._NUMBER_RE.finditer(content))

        # Technical terms (CamelCase, snake_case, ACRONYMS)
        keywords.extend(m.group() for m in QueryMixin._TECH_TERM_RE.finditer(content))

        # Deduplicate while preserving order
        seen: Set[str] = set()
        unique: List[str] = []
        for kw in keywords:
            lower = kw.lower()
            if lower not in seen:
                seen.add(lower)
                unique.append(kw)

        return " ".join(unique[:50])  # Cap at 50 keywords

    @staticmethod
    def _rrf_fuse(
        ranked_lists: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None,
        k: int = _RRF_K,
    ) -> Dict[str, float]:
        """Reciprocal Rank Fusion across multiple ranked retrieval channels.

        Combines rank-based signals from heterogeneous scoring sources (vector
        similarity, BM25, temporal proximity) into a single unified score.
        Documents found by multiple channels naturally receive higher scores.

        Canonical weighted RRF (Cormack et al. 2009):
            score(d) = sum_c [ w_c / (k + rank_c(d) + 1) ]
        followed by a max-normalization of the combined scores to [0, 1] —
        downstream abstention floors (ctx_min_composite 0.08-0.10,
        _MIN_COMPOSITE_SCORE) are calibrated to that scale; raw RRF peaks at
        ~1/(k+1) and would abstain on everything.

        Historical note: this function used to min-max normalize each channel
        to [0,1] before weighting ("so one channel's outlier doesn't compress
        the others"). That step was provably inert: every non-empty channel's
        top rank scores exactly 1/(k+1), so the per-channel division was one
        uniform scalar across all channels, cancelled by the final max-norm.
        Verified bit-identical over randomized inputs (see
        test_rrf_canonical_equals_historical_renorm) and removed.

        Args:
            ranked_lists: List of ranked results per channel. Each is
                [(doc_id, raw_score), ...] sorted by score descending.
            weights: Per-channel weights. Defaults to uniform 1.0.
            k: Smoothing constant (default 60, from Cormack et al. 2009).

        Returns:
            Dict of {doc_id: rrf_score} normalized to [0, 1] range.
        """
        if not ranked_lists:
            return {}
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        scores: Dict[str, float] = {}
        for channel_idx, ranked in enumerate(ranked_lists):
            w = weights[channel_idx]
            for rank_pos, (doc_id, _raw_score) in enumerate(ranked):
                scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank_pos + 1)

        # Normalize combined scores to [0, 1] (downstream floors assume this).
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {did: s / max_score for did, s in scores.items()}

        return scores

    def _semantic_dedup(
        self,
        results: List["MemoryResult"],
        node_scores: Dict[str, float],
        threshold: float = 0.92,
    ) -> List["MemoryResult"]:
        """Remove semantic near-duplicates from query results.

        Pairwise cosine similarity on stored embeddings. When a pair exceeds
        the threshold, the lower-scored item is dropped (results are already
        sorted by score desc). O(n^2) where n <= ~20; ~1ms total.
        """
        if len(results) <= 1:
            return results

        # Batch-fetch embeddings (single query instead of N+1)
        embeddings: Dict[str, List[float]] = {}
        if self._vec_available:
            node_ids = [r.id for r in results]
            ph = ",".join("?" * len(node_ids))
            try:
                rows = self._conn.execute(
                    f"""SELECT m.node_id, v.embedding
                        FROM memories m
                        JOIN memories_vec v ON v.rowid = m.id
                        WHERE m.node_id IN ({ph})""",
                    node_ids,
                ).fetchall()
                for nid, emb_bytes in rows:
                    if emb_bytes:
                        embeddings[nid] = _deserialize_f32(emb_bytes)
            except Exception:
                logger.debug("Batch embedding fetch failed, falling back to individual", exc_info=True)
                for r in results:
                    emb = self.get_embedding(r.id)
                    if emb:
                        embeddings[r.id] = emb

        if len(embeddings) < 2:
            return results

        # Mark items to drop (lower-scored of each near-duplicate pair)
        drop_ids: set = set()
        indexed = list(results)
        for i in range(len(indexed)):
            if indexed[i].id in drop_ids:
                continue
            emb_a = embeddings.get(indexed[i].id)
            if not emb_a:
                continue
            for j in range(i + 1, len(indexed)):
                if indexed[j].id in drop_ids:
                    continue
                emb_b = embeddings.get(indexed[j].id)
                if not emb_b:
                    continue
                dot = sum(x * y for x, y in zip(emb_a, emb_b))
                norm_a = sum(x * x for x in emb_a) ** 0.5
                norm_b = sum(x * x for x in emb_b) ** 0.5
                if norm_a == 0 or norm_b == 0:
                    continue
                cosine_sim = dot / (norm_a * norm_b)
                if cosine_sim >= threshold:
                    # Drop the lower-scored item (j is always lower since sorted desc)
                    drop_ids.add(indexed[j].id)

        if drop_ids:
            self.stats.setdefault("semantic_dedup_query", 0)
            self.stats["semantic_dedup_query"] += len(drop_ids)

        return [r for r in results if r.id not in drop_ids]

    @staticmethod
    def _word_overlap(query_words: list, searchable: str) -> float:
        """Compute word overlap ratio with lightweight stemming and canonicalization.

        Checks exact substring match first, then falls back to
        suffix-stripped stems to handle morphological variants
        (e.g., deploy/deployed/deployment all share stem 'deploy').
        Applies NFKC canonicalization (#6) for better matching.
        """
        if not query_words:
            return 0.0
        searchable = _canonicalize(searchable)
        _SUFFIXES = (
            "ation",
            "tion",
            "ment",
            "ing",
            "ness",
            "ity",
            "ous",
            "ive",
            "able",
            "ed",
            "er",
            "es",
            "ly",
            "al",
            "s",
        )
        matched = 0
        for w in query_words:
            cw = _canonicalize(w)
            if cw in searchable:
                matched += 1
            else:
                # Lightweight stemming: strip one common suffix
                stem = cw
                for suffix in _SUFFIXES:
                    if cw.endswith(suffix) and len(cw) - len(suffix) >= 3:
                        stem = cw[: -len(suffix)]
                        break
                if stem != cw and stem in searchable:
                    matched += 1
        return matched / len(query_words)

    # Durable-knowledge types whose decay slows with use: a decision or lesson
    # that keeps getting recalled is load-bearing knowledge, not aging exhaust.
    _ACCESS_AWARE_DECAY_TYPES = frozenset({
        "decision", "lesson_learned", "advisor_insight", "skill_template",
        "project_history",
    })

    def _metadata_score_factor(self, node) -> float:
        """Shared metadata scoring used by BOTH the fusion phase and the
        strong-signal short-circuit: type weight x feedback x priority x decay.

        Kept in one place so no retrieval path can bypass knowledge weighting.
        """
        event_type = node.metadata.get("event_type", "")
        type_weight = self._TYPE_WEIGHTS.get(event_type, 1.0)
        fb_factor = self._compute_fb_factor(node.metadata.get("feedback_score", 0))
        # Defensive coercion: priority is user-supplied metadata and has shown
        # up as text ("high") in the wild — a raise here would abort the whole
        # query for every candidate set containing the malformed row.
        try:
            priority = float(node.metadata.get("priority", 3))
        except (TypeError, ValueError):
            priority = 3.0
        priority = min(5.0, max(1.0, priority))
        priority_factor = 0.7 + (priority * 0.08)
        # capture_confidence as a ranking signal (audit follow-up) — env-gated
        # OFF until the A/B shows it earns a default.
        confidence_factor = 1.0
        if os.environ.get("CAIRN_CONFIDENCE_SIGNAL", "0") == "1":
            confidence_factor = {"high": 1.1, "low": 0.8}.get(
                node.metadata.get("capture_confidence"), 1.0
            )
        _la = node.last_accessed.isoformat() if node.last_accessed else None
        _ca = node.created_at.isoformat() if node.created_at else None
        decay_factor = self._compute_decay_factor(
            event_type, _la, _ca, node.access_count or 0
        )
        return type_weight * fb_factor * priority_factor * decay_factor * confidence_factor

    def _compute_decay_factor(self, event_type: str, last_accessed: Optional[str],
                               created_at: Optional[str],
                               access_count: int = 0) -> float:
        """Compute time-decay factor for query scoring.

        Uses exponential decay: factor = max(floor, exp(-lambda * days))
        Protected types (lambda=0) return 1.0 immediately.
        For durable-knowledge types (_ACCESS_AWARE_DECAY_TYPES), each access
        reduces lambda by 15% (floor 0.002, ~346-day half-life).
        """
        lam = self._DECAY_LAMBDAS.get(event_type, 0.02)
        if lam == 0.0:
            return 1.0

        # Access-aware decay: well-used knowledge persists longer
        if event_type in self._ACCESS_AWARE_DECAY_TYPES and access_count > 0:
            lam = max(0.002, lam * (0.85 ** min(access_count, 10)))

        # Use last_accessed if available, else created_at
        ref_str = last_accessed or created_at
        if not ref_str:
            return 1.0

        try:
            ref_dt = self._parse_dt(ref_str)
            if ref_dt is None:
                return 1.0
            days = (datetime.now(timezone.utc) - ref_dt).total_seconds() / 86400.0
            if days <= 0:
                return 1.0
            floor = self._DECAY_FLOOR if access_count > 0 else self._DECAY_FLOOR_NEVER_ACCESSED
            return max(floor, math.exp(-lam * days))
        except Exception as e:
            logger.debug("Decay computation failed: %s", e)
            return 1.0

    @staticmethod
    def _compute_fb_factor(fb_score: int) -> float:
        """Compute feedback factor for query scoring.

        Amplified formula: positive feedback gives meaningful boost,
        negative feedback aggressively demotes.
        """
        if fb_score >= 0:
            return 1.0 + min(fb_score, 10) * 0.15  # +5 -> 1.75x, +10 -> 2.5x (capped)
        else:
            return max(0.2, 1.0 + fb_score * 0.2)  # -2 -> 0.6x, -4 -> 0.2x (floor)

    def _get_thompson_boost(self, event_type: str) -> float:
        """Get Thompson sampling boost factor for an event type.

        Returns 1.0 (neutral) if Thompson module unavailable or insufficient data.
        The unavailability flag is cached: Python does not cache FAILED imports,
        and this runs once per candidate per query — without the flag a missing
        module costs a sys.path walk for every scored result.
        """
        global _THOMPSON_UNAVAILABLE
        if _THOMPSON_UNAVAILABLE:
            return 1.0
        try:
            from cairn.thompson import ThompsonBandit
        except ImportError:
            _THOMPSON_UNAVAILABLE = True
            return 1.0
        try:
            bandit = ThompsonBandit(store=self)
            arm_id = f"event_type:{event_type}" if event_type else "event_type:unknown"
            return bandit.get_boost_factor(arm_id)
        except Exception as e:
            logger.debug("Thompson sampling boost skipped: %s", e)
            return 1.0

    # Factual signals match WHOLE WORDS only. Bare substring matching
    # classified "the whole pipeline broke" as FACTUAL ("who" in "whole")
    # and "somewhere to store X" as FACTUAL ("where" in "somewhere") —
    # and FACTUAL weights (0.3 vec / 1.5 text) all but disable semantic
    # retrieval for the misclassified query.
    _FACTUAL_SIGNAL_RE = re.compile(
        r"\b(?:what (?:was|is|are)|which|when (?:did|was)|who|where"
        r"|did (?:we|i)|was there|is there|decision about|preference for"
        r"|error with|bug in|remind me|remember)\b"
    )
    # Bare single-word signals (which/who/where) were A/B'd against a
    # multiword-only variant on the 113-probe set (2026-07-25): zero probes
    # changed outcome (+0/-0). Kept as-is — no evidence either way.

    @staticmethod
    def _apply_channel_top_guard(
        sorted_ids: List[str],
        node_scores: Dict[str, float],
        channel_best: List[str],
        insert_at: int,
    ) -> None:
        """Splice a channel's best candidate(s) into the fused ranking at
        ``insert_at`` (best-first, consecutive slots) if fusion buried them
        deeper. In-place; skips candidates that didn't survive scoring or
        already sit at/above their target slot."""
        for offset, cand in enumerate(channel_best):
            if cand not in node_scores or cand not in sorted_ids:
                continue
            target = insert_at + offset
            pos = sorted_ids.index(cand)
            if pos > target:
                sorted_ids.remove(cand)
                sorted_ids.insert(target, cand)

    def _apply_vec_top_guard(
        self,
        sorted_ids: List[str],
        raw_vec_sims: Dict[str, float],
        node_scores: Dict[str, float],
    ) -> None:
        """Back-compat wrapper: vec channel's single top into index 2."""
        if not raw_vec_sims:
            return
        vec_top = max(raw_vec_sims, key=lambda n: raw_vec_sims[n])
        self._apply_channel_top_guard(sorted_ids, node_scores, [vec_top], insert_at=2)

    def _classify_query_intent(self, query_text: str) -> Optional[QueryIntent]:
        """Classify query intent for adaptive retrieval budget (#3)."""
        if self._is_keyword_sufficient(query_text):
            return QueryIntent.NAVIGATIONAL
        query_lower = query_text.lower()
        if self._FACTUAL_SIGNAL_RE.search(query_lower):
            return QueryIntent.FACTUAL
        _CONCEPTUAL_SIGNALS = (
            "how does", "how do", "how to", "why does", "why do", "why is",
            "explain", "understand", "overview", "architecture", "design",
            "pattern", "approach", "strategy", "concept",
        )
        if any(query_lower.startswith(sig) or sig in query_lower for sig in _CONCEPTUAL_SIGNALS):
            return QueryIntent.CONCEPTUAL
        return None

    @staticmethod
    def _decompose_query(query_text: str) -> Optional[List[str]]:
        """Split compound queries into independent sub-queries.

        Detects conjunctions ("and", "also", "as well as") and comma-separated
        clauses to decompose multi-part questions into atomic sub-queries.
        Returns None if the query is not compound (single topic).

        Only decomposes when each sub-query is independently meaningful
        (>= 12 chars and >= 2 words after splitting).
        """
        text = query_text.strip()
        if len(text) < 25:
            return None  # Too short to be compound

        # Avoid splitting inside quoted strings
        if text.count('"') >= 2 or text.count("'") >= 2:
            return None

        # Try splitting on explicit conjunctions
        # Pattern: "X and Y", "X, and Y", "X as well as Y", "X and also Y"
        parts = _CONJUNCTION_PATTERN.split(text)

        # Also try comma-separated clauses (e.g., "what auth did we use, which API framework")
        if len(parts) <= 1:
            # Only split on commas if each part looks like an independent clause
            # (starts with a question word or verb)
            comma_parts = [p.strip() for p in text.split(",")]
            if len(comma_parts) >= 2:
                if all(_CLAUSE_STARTS.match(p.strip()) for p in comma_parts if p.strip()):
                    parts = comma_parts

        if len(parts) <= 1:
            return None

        # Validate each sub-query is meaningful
        valid_parts = []
        for p in parts:
            p = p.strip().rstrip("?.,;!")
            if len(p) >= 12 and len(p.split()) >= 2:
                valid_parts.append(p)

        if len(valid_parts) < 2:
            return None  # Not enough meaningful sub-queries

        return valid_parts[:4]  # Cap at 4 sub-queries to avoid explosion
