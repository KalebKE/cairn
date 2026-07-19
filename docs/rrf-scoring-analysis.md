# RRF Scoring Pipeline Analysis

> Generated 2026-03-06. Source files analyzed at HEAD.

## Overview

Cairn's query pipeline uses a multi-phase scoring architecture:

1. **Fast-path / hot-cache** short-circuits (trigram, in-memory)
2. **Vector similarity** (sqlite-vec, cosine distance)
3. **FTS5 text search** (BM25 + word-match blend)
4. **Temporal retrieval** (date-proximity scoring)
5. **Strong signal short-circuit** (skip fusion when FTS5 is decisive)
6. **LLM query expansion** (opt-in, conceptual queries only)
7. **RRF score fusion** (vector + FTS5 + temporal channels)
8. **Metadata multipliers** (type weight, feedback, priority, decay, Thompson)
9. **Word/tag overlap boost** (post-fusion additive)
10. **Filtering** (expired, superseded, infrastructure, project/session scope)
11. **Contextual boosting** (file/tag context, cluster co-boost, temporal constraint)
12. **Graph expansion + cross-encoder reranking**
13. **Assembly** (dedup, abstention, normalization, caching)

---

## Parameter Inventory

### 1. RRF K Constant

| Parameter | Value | File | Line |
|-----------|-------|------|------|
| `_RRF_K` | `60` | `_types.py` | 36 |

Used in `_rrf_fuse()` at `_query.py:1468`:
```
channel_scores[doc_id] = 1.0 / (k + rank_pos + 1)
```
Higher k = flatter rank curve (less difference between rank 1 and rank 20). Lower k = steeper (top ranks dominate).

### 2. RRF Channel Weights (Phase 3)

Set at `_query.py:613-617`:

| Channel | Weight Source | Default |
|---------|-------------|---------|
| Vector (vec_ranked) | `pw_vec` (from retrieval profile x intent weight) | 1.0 |
| FTS5 (text_ranked) | `pw_text` (from retrieval profile x intent weight) | 1.0 |
| Temporal (temporal_ranked) | Hardcoded | **1.2** |

The `pw_vec` and `pw_text` are computed as:
```
profile_weight * intent_weight
```

### 3. Retrieval Profiles (ALMA-inspired)

Defined at `_base.py:205-220`. Tuple: `(vec, text, word_overlap, context, graph)`.

| Profile | vec | text | word | ctx | graph |
|---------|-----|------|------|-----|-------|
| `_default` | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| `error_pattern` | 0.3 | 1.5 | 2.0 | 0.5 | 0.3 |
| `decision` | 0.8 | 0.6 | 0.5 | 1.0 | 2.0 |
| `lesson_learned` | 1.5 | 0.8 | 0.5 | 0.8 | 1.0 |
| `user_preference` | 0.6 | 1.0 | 1.5 | 0.3 | 0.3 |
| `knowledge-update` | 0.8 | 1.3 | 1.5 | 1.0 | 1.0 |
| `multi-session` | 1.3 | 1.0 | 1.3 | 1.0 | 1.0 |
| `temporal-reasoning` | 1.0 | 1.3 | 1.3 | 1.0 | 1.0 |

### 4. Intent Weights (Adaptive)

Defined at `_types.py:94-98`. Multiplied onto retrieval profile weights at `_query.py:218-223`.

| Intent | vec | text | word | ctx | graph |
|--------|-----|------|------|-----|-------|
| FACTUAL | 0.3 | 1.5 | 1.8 | 1.0 | 1.0 |
| CONCEPTUAL | 1.8 | 0.5 | 0.3 | 1.0 | 1.0 |
| NAVIGATIONAL | 0.1 | 2.0 | 2.0 | 0.5 | 0.3 |

### 5. FTS5 Internal Scoring (`_search.py:149-150`)

BM25 rank normalized to [0.1, 1.0], then blended:
```
relevance = 0.7 * bm25_norm + 0.3 * word_ratio
```

### 6. RRF Per-Channel Normalization (`_query.py:1469-1473`)

Each channel's RRF scores are independently normalized to [0,1] before weighted accumulation. Final combined scores are also normalized to [0,1].

### 7. Type Weights (Metadata Multiplier)

Defined at `_base.py:35-74`. Applied at `_query.py:627`. Range: 0.05 (`file_summary`) to 3.0 (`constraint`, `reminder`).

### 8. Feedback Factor (`_query.py:1639-1648`)

```
positive: 1.0 + min(fb_score, 10) * 0.15   (max 2.5x)
negative: max(0.2, 1.0 + fb_score * 0.2)    (min 0.2x)
```

### 9. Priority Factor (`_query.py:633-634`)

```
priority_factor = 0.7 + (priority * 0.08)
```
Range: 0.78 (priority=1) to 1.10 (priority=5).

### 10. Decay Factor (`_query.py:1603-1636`)

Exponential decay: `max(floor, exp(-lambda * days))`.
- `_DECAY_FLOOR` = 0.35 (accessed memories), `_base.py:193`
- `_DECAY_FLOOR_NEVER_ACCESSED` = 0.15, `_base.py:194`
- Per-type lambdas at `_base.py:178-192`

### 11. Consolidation Quality Boost (`_query.py:642-644`)

```
score *= 1.0 + min(cq, 3.0) * 0.1   (up to 1.3x)
```

### 12. Word Overlap Boost (Post-Fusion) (`_query.py:656-662`)

```
node_scores[nid] *= 1.0 + word_ratio * 0.5 * fb_mod * pw_word
```
Where `fb_mod` = 0.5 for negatively-rated, 1.0 otherwise.

### 13. Preference Signal Boost (`_query.py:676`)

`user_preference` type memories get 1.5x when query contains preference keywords.

### 14. Cross-Encoder Reranker Integration (`_query.py:1019-1061`)

Position-aware CE boost (QMD-inspired):

| Rank Position | CE Weight (`ce_w`) |
|--------------|-------------------|
| 1-3 | 0.15 |
| 4-10 | 0.30 |
| 11+ | 0.50 |

Applied as: `node_scores[nid] *= 1.0 + ce_w * ce_norm[i]`

CE scores normalized to [0,1] before application. Reranks top 20 candidates (`_RERANK_CANDIDATES = 20`).

### 15. Graph Expansion (`_query.py:990-1017`)

- `_HOP_DECAY` = 0.4 per hop
- `_MAX_GRAPH_HOPS` = 2
- Score: `seed_score * (0.4^hop) * min(weight, 1.0) * pw_graph`

### 16. Contextual Boost (`_query.py:806-807`)

```
boost = 1.0 + ((tag_overlap * 0.10) + (project_match * 0.15) + (min(content_match, 3) * 0.05)) * pw_ctx
```

### 17. Surfacing Context Thresholds (`_types.py:73-80`)

| Context | min_vec | min_text | min_composite | ctx_weight_boost |
|---------|---------|----------|---------------|-----------------|
| GENERAL | 0.50 | 0.35 | 0.10 | 1.0 |
| ERROR_DEBUG | 0.40 | 0.45 | 0.08 | 1.0 |
| FILE_EDIT | 0.50 | 0.35 | 0.10 | 2.0 |
| SESSION_START | 0.45 | 0.40 | 0.10 | 1.0 |
| PLANNING | 0.45 | 0.40 | 0.10 | 1.5 |
| REVIEW | 0.45 | 0.40 | 0.10 | 1.5 |

### 18. Strong Signal Short-Circuit (`_query.py:29-30`)

| Parameter | Value | Env Var |
|-----------|-------|---------|
| `STRONG_SIGNAL_THRESHOLD` | 0.85 | `CAIRN_STRONG_SIGNAL_THRESHOLD` |
| `STRONG_SIGNAL_GAP` | 0.15 | `CAIRN_STRONG_SIGNAL_GAP` |

### 19. Adaptive Retry (`_query.py:33-34`)

| Parameter | Value | Env Var |
|-----------|-------|---------|
| `ADAPTIVE_RETRY_THRESHOLD` | 0.3 | `CAIRN_ADAPTIVE_RETRY_THRESHOLD` |
| `ADAPTIVE_RETRY_RELAXATION` | 0.6 | `CAIRN_ADAPTIVE_RETRY_RELAXATION` |

### 20. Hot Cache Seeding (`_query.py:243-244`)

Hot cache results are seeded into `node_scores` at **0.8x their relevance**, bypassing RRF:
```
node_scores[hr.id] = hr.relevance * 0.8
```

### 21. Expansion Weight Discount (`_query.py:481`)

Query expansion variants are weighted at **0.8x** (`_EXPANSION_WEIGHT_DISCOUNT`).

### 22. Temporal In-Range / Out-of-Range Boost (`_query.py:841-851`)

| Condition | Multiplier |
|-----------|-----------|
| In-range | 1.3x |
| Out-of-range, soft (temporal_boost_only) | 0.85x |
| Out-of-range, explicit date | 0.15x |
| Out-of-range, inferred date | 0.70x |

### 23. Semantic Dedup Threshold (`_query.py:1187`)

Default: 0.92 cosine similarity. Env var: `CAIRN_SEMANTIC_DEDUP_THRESHOLD`.

---

## Full Scoring Formula

For a document `d` found by vector channel at rank `v_rank` and FTS5 channel at rank `t_rank`:

```
# Step 1: Per-channel RRF (each normalized to [0,1])
vec_rrf(d)  = (1/(k + v_rank + 1)) / max_vec_rrf
text_rrf(d) = (1/(k + t_rank + 1)) / max_text_rrf

# Step 2: Weighted fusion (normalized to [0,1])
rrf(d) = (pw_vec * vec_rrf(d) + pw_text * text_rrf(d)) / max_combined

# Step 3: Metadata multipliers
score(d) = rrf(d) * type_weight * fb_factor * priority_factor * decay_factor * thompson_boost

# Step 4: Consolidation quality
if cq > 0: score(d) *= 1.0 + min(cq, 3.0) * 0.1

# Step 5: Word/tag overlap (post-fusion)
score(d) *= 1.0 + word_ratio * 0.5 * fb_mod * pw_word

# Step 6: Contextual boost
score(d) *= ctx_boost

# Step 7: Cluster co-boost
score(d) *= cluster_boost  (1.05-1.15x)

# Step 8: Temporal constraint
score(d) *= temporal_factor  (0.15x to 1.3x)

# Step 9: Cross-encoder rerank
score(d) *= 1.0 + ce_w * ce_norm(d)

# Step 10: Final normalization to [0,1]
relevance(d) = score(d) / max_score
```

---

## Parameters Most Likely to Impact Retrieval Quality

**High Impact** (directly controls rank ordering):
1. **`_RRF_K`** (60) -- controls rank curve steepness across all channels
2. **FTS5 BM25/word blend** (0.7/0.3) -- determines text channel quality
3. **Intent weights** -- large multipliers (0.1x to 2.0x) that swing channel importance
4. **Cross-encoder position weights** (0.15/0.30/0.50) -- controls reranker influence
5. **Retrieval profile weights** -- per-type channel emphasis

**Medium Impact** (affects subset of queries):
6. **Type weights** -- 0.05x to 3.0x range creates large score differentials
7. **Temporal channel weight** (1.2) -- hardcoded, not tunable via profiles
8. **Word overlap boost coefficient** (0.5) -- post-fusion additive
9. **Surfacing context thresholds** -- abstention sensitivity

**Lower Impact** (fine-tuning):
10. **Decay lambdas and floors** -- affects stale content ranking
11. **Priority factor** (0.78-1.10x range is narrow)
12. **Consolidation quality boost** (up to 1.3x, rare)
