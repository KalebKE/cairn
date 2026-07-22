"""Maintenance -- consolidation, compaction, connection discovery, insight
synthesis. Peeled from bridge/__init__.py (Wave 6)."""
import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import cairn.bridge as _bridge
from cairn.types import TTLCategory

logger = logging.getLogger("cairn.bridge")



# ---------------------------------------------------------------------------
# Public API -- Consolidation
# ---------------------------------------------------------------------------


def _auto_backup_before_consolidate():
    """Create a backup before consolidation (rotate to keep last 3)."""
    db_path = _bridge.CAIRN_HOME / "cairn.db"
    if not db_path.exists():
        return
    try:
        import sqlite3
        from cairn.crypto import secure_connect

        backups_dir = _bridge.CAIRN_HOME / "backups"
        backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        backup_path = backups_dir / f"pre-consolidate-{timestamp}.db"
        src = sqlite3.connect(str(db_path))
        dst = secure_connect(backup_path)
        src.backup(dst)
        dst.close()
        src.close()
        logger.info(f"Pre-consolidation backup: {backup_path}")
        # Rotate — keep only last 3
        backups = sorted(backups_dir.glob("pre-consolidate-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in backups[3:]:
            old.unlink()
    except Exception as e:
        logger.warning(f"Auto-backup before consolidation failed: {e}")


def consolidate(prune_days: int = 14, max_summaries: int = 50) -> str:
    """Run memory consolidation: prune stale entries, cap summaries, clean edges.

    Returns formatted markdown report.
    """
    _auto_backup_before_consolidate()
    db = _bridge._get_store()
    before = db.node_count()
    stats = db.consolidate(prune_days=prune_days, max_summaries=max_summaries)
    after = stats.get("node_count_after", before)
    removed = before - after

    output = "# Memory Consolidation Report\n\n"
    output += f"**Before:** {before} memories\n"
    output += f"**After:** {after} memories\n"
    output += f"**Removed:** {removed} total\n\n"
    output += "## Breakdown\n\n"
    output += f"- **Stale (0 access, >{prune_days}d old):** {stats.get('pruned_stale', 0)}\n"
    output += f"- **Session summaries (beyond cap of {max_summaries}):** {stats.get('pruned_summaries', 0)}\n"
    output += f"- **Orphaned edges:** {stats.get('pruned_edges', 0)}\n"
    output += f"- **Strength-decayed:** {stats.get('decayed_memories', 0)}\n"
    output += f"- **Merged entities:** {stats.get('merged_entities', 0)}\n"

    if removed == 0:
        output += "\n*Nothing to consolidate — memory store is clean.*\n"
    else:
        logger.info(f"Consolidation: removed {removed} memories ({stats})")

    return output


# ---------------------------------------------------------------------------
# Public API -- Memory Compaction
# ---------------------------------------------------------------------------


def _smart_extract(cluster) -> str:
    """Extract diverse, information-dense sentences from a cluster of memories.

    Scores sentences by: unique-word count (IDF-like), sentence length
    (diminishing returns), presence of proper nouns / code tokens, and
    cross-memory term frequency (words appearing in 2+ cluster members
    are more generalizable — ALMA-inspired strategy extraction).

    For large clusters (5+), extracts a strategy header from the most
    common bigram theme across cluster members.

    Skips near-duplicate sentences (Jaccard > 0.7).
    Orders selected sentences chronologically by source memory.
    Returns consolidated text capped at 1000 chars.
    """
    # Build cross-memory word frequency map (words appearing in 2+ members)
    word_to_members: dict = {}  # word -> set of node indices
    for idx, node in enumerate(cluster):
        for w in set(node.content.lower().split()):
            if len(w) > 3:
                word_to_members.setdefault(w, set()).add(idx)
    cross_freq_words = {w for w, members in word_to_members.items() if len(members) >= 2}

    # Collect all sentences with source metadata
    all_sentences = []  # [(sentence, density_score, created_at)]
    seen_keys: set = set()

    for node in cluster:
        created = node.created_at.isoformat() if node.created_at else ""
        for sentence in re.split(r"(?<=[.!?])\s+", node.content):
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            key = " ".join(sentence.lower().split())[:100]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            words = sentence.split()
            unique_words = len(set(w.lower() for w in words if len(w) > 3))

            # Proper nouns / capitalized words (not sentence-start)
            proper_nouns = len([w for w in words[1:] if w[0].isupper()]) if len(words) > 1 else 0

            # Code tokens: backtick spans, paths, CamelCase
            code_tokens = len(re.findall(r"`[^`]+`|/[\w/.]+|\b[A-Z][a-z]+[A-Z]\w*\b", sentence))

            # Diminishing returns on length
            length_score = min(len(sentence), 200) / 200.0

            # Cross-memory term frequency boost (ALMA-inspired)
            cross_freq = sum(1 for w in words if w.lower() in cross_freq_words)

            density = (unique_words * 1.0 + proper_nouns * 1.5 + code_tokens * 2.0
                       + length_score * 3.0 + cross_freq * 0.8)
            all_sentences.append((sentence, density, created))

    if not all_sentences:
        return ""

    # Sort by density (highest first)
    all_sentences.sort(key=lambda x: x[1], reverse=True)

    # Select top-K diverse sentences
    selected = []
    for sentence, _score, created in all_sentences:
        # Check diversity against already selected
        is_diverse = all(_bridge._jaccard(sentence.lower(), sel[0].lower(), min_word_len=3) < 0.7 for sel in selected)
        if is_diverse:
            selected.append((sentence, created))
            if len(selected) >= 8:  # Max sentences to consider
                break

    # Order chronologically by source memory created_at
    selected.sort(key=lambda x: x[1])

    # Build consolidated text (cap at 1000 chars)
    consolidated = " ".join(s for s, _ in selected)

    # Strategy header for large clusters (5+ members): extract common bigram theme
    if len(cluster) >= 5:
        bigram_counter: Counter = Counter()
        for node in cluster:
            words = [w.lower() for w in node.content.split() if len(w) > 3]
            for w1, w2 in zip(words, words[1:]):
                bigram_counter[(w1, w2)] += 1
        if bigram_counter:
            top_bigram, top_count = bigram_counter.most_common(1)[0]
            if top_count >= 3:  # Only if bigram appears in 3+ members
                theme = f"{top_bigram[0]} {top_bigram[1]}"
                consolidated = f"Strategy: {theme}. {consolidated}"

    if len(consolidated) > 1000:
        consolidated = consolidated[:997] + "..."

    return consolidated


def compact(
    event_type: str = "lesson_learned",
    similarity_threshold: float = 0.60,
    min_cluster_size: int = 3,
    dry_run: bool = False,
) -> str:
    """Compact clusters of related memories into consolidated knowledge nodes.

    Unlike deduplicate() which removes exact/near duplicates, compact() finds
    clusters of semantically related memories and creates new summary nodes
    that consolidate the knowledge, marking originals as superseded.

    Returns formatted markdown report.
    """
    db = _bridge._get_store()
    all_candidates = db.get_by_type(event_type, limit=500)
    # Filter out superseded memories — these were already compacted into a
    # consolidated node.  Re-including them causes nested "[Consolidated from]"
    # prefixes and duplicate consolidated nodes.
    candidates = [
        n for n in all_candidates
        if not (n.metadata or {}).get("superseded")
    ]

    if len(candidates) < min_cluster_size:
        return (
            f"# Memory Compaction\n\n"
            f"Only {len(candidates)} `{event_type}` memories found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to compact.\n"
        )

    # Build word sets for Jaccard clustering
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]

    # Union-find style clustering
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break  # All items assigned, no more clusters possible
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# Memory Compaction\n\n"
            f"No clusters found with >= {min_cluster_size} similar `{event_type}` memories "
            f"at {similarity_threshold:.0%} similarity. Store is already compact.\n"
        )

    # Build report and optionally perform compaction
    output = f"# Memory Compaction {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**Event type:** {event_type}\n"
    output += f"**Similarity threshold:** {similarity_threshold:.0%}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        # Sort by content length (longest first — most information)
        cluster.sort(key=lambda n: len(n.content), reverse=True)

        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        output += f"## Cluster {ci} ({len(cluster)} memories)\n\n"
        output += f"**Summary:** {consolidated[:200]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            # Strip any existing "[Consolidated from ...]" prefix from the
            # extracted content to prevent nested consolidation headers.
            consolidated = re.sub(
                r"^(\[Consolidated from \d+ memories\]\s*)+",
                "",
                consolidated,
            ).lstrip()
            # Prefix consolidated content to distinguish from originals (avoids dedup)
            compact_header = f"[Consolidated from {len(cluster)} memories] "
            compact_content = compact_header + consolidated

            # Create the consolidated node with quality metadata
            # Quality scale: 1.0 (min cluster) to 3.0 (10+ members)
            consolidation_quality = min(3.0, 1.0 + (len(cluster) - min_cluster_size) * 0.3)
            meta = {
                "event_type": event_type,
                "source": "compaction",
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
                "consolidation_quality": round(consolidation_quality, 2),
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=TTLCategory.for_event_type(event_type),
                skip_inference=True,  # Bypass embedding dedup
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded + log to forgetting audit trail
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["compacted_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, event_type,
                    "compaction_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} memories\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would compact:** {sum(len(c) for c in clusters)} memories into {len(clusters)} nodes\n"
    else:
        output += f"**Compacted:** {total_compacted} memories into {total_created} consolidated nodes\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- Active Connection Discovery (Consolidation Daemon)
# ---------------------------------------------------------------------------


def discover_connections(
    lookback_hours: int = 24,
    similarity_threshold: float = 0.70,
    max_memories: int = 100,
    max_connections_per_memory: int = 3,
    dry_run: bool = False,
) -> str:
    """Actively discover and link related memories that aren't yet connected.

    Scans recent memories, finds semantically similar ones that lack edges,
    and creates 'related' edges between them. When cross-cutting patterns
    are found (clusters spanning multiple event types), generates
    advisor_insight entries.

    This is the core of the active consolidation daemon — it generates
    new knowledge from existing memories rather than just pruning.

    Args:
        lookback_hours: How far back to scan for unlinked memories.
        similarity_threshold: Minimum cosine similarity to create an edge (0.0-1.0).
        max_memories: Maximum memories to process per run.
        max_connections_per_memory: Maximum new edges per memory.
        dry_run: If True, report what would be linked without modifying.

    Returns:
        Formatted markdown report.
    """
    db = _bridge._get_store()

    if not db._vec_available:
        return "# Connection Discovery\n\nVector search unavailable — cannot discover connections.\n"

    # Phase 1: Find recent memories without many edges
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    candidates = db._conn.execute(
        """SELECT m.node_id, m.content, m.event_type, m.id, m.created_at,
                  m.entity_id, m.status
           FROM memories m
           WHERE m.created_at > ?
             AND (m.status IS NULL OR m.status = 'active')
             AND m.event_type NOT IN ('session_summary', 'coordination_snapshot',
                                       'session_respawn', 'code_chunk', 'file_summary')
           ORDER BY m.created_at DESC
           LIMIT ?""",
        (cutoff, max_memories),
    ).fetchall()

    if not candidates:
        return (
            f"# Connection Discovery\n\n"
            f"No recent active memories in the last {lookback_hours}h to analyze.\n"
        )

    # Phase 2: For each candidate, find similar memories and create edges
    edges_created = 0
    edges_skipped = 0
    cross_type_clusters = []  # Track cross-type connections for insight generation

    # Get existing edges for candidates to avoid redundant checks
    candidate_ids = {c[0] for c in candidates}
    existing_edges = set()
    if candidate_ids:
        placeholders = ",".join("?" * len(candidate_ids))
        rows = db._conn.execute(
            f"SELECT source_id, target_id FROM edges WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})",
            list(candidate_ids) + list(candidate_ids),
        ).fetchall()
        for r in rows:
            existing_edges.add((r[0], r[1]))
            existing_edges.add((r[1], r[0]))  # Bidirectional check

    report_lines = []

    for node_id, content, event_type, rowid, created_at, entity_id, status in candidates:
        # Get embedding for this memory
        try:
            emb_row = db._conn.execute(
                "SELECT embedding FROM memories_vec WHERE rowid = ?", (rowid,)
            ).fetchone()
            if not emb_row:
                continue
        except Exception:
            continue

        # Find similar memories
        import struct
        _EMBED_DIM = 384
        expected_size = _EMBED_DIM * 4  # 4 bytes per float
        if len(emb_row[0]) != expected_size:
            continue
        embedding = list(struct.unpack(f"{_EMBED_DIM}f", emb_row[0]))
        similar = db._vec_query(embedding, limit=max_connections_per_memory + 5)

        connections_made = 0
        for sim_rowid, distance in similar:
            if connections_made >= max_connections_per_memory:
                break

            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue

            # Look up the similar memory
            sim_row = db._conn.execute(
                "SELECT node_id, event_type, content FROM memories WHERE id = ?",
                (sim_rowid,),
            ).fetchone()
            if not sim_row or sim_row[0] == node_id:
                continue

            sim_node_id, sim_event_type, sim_content = sim_row

            # Skip if edge already exists
            if (node_id, sim_node_id) in existing_edges:
                edges_skipped += 1
                continue

            # Create the edge
            if not dry_run:
                db.add_edge(
                    source_id=node_id,
                    target_id=sim_node_id,
                    edge_type="related",
                    weight=round(similarity, 3),
                    metadata={"source": "discover_connections", "auto": True},
                )

            existing_edges.add((node_id, sim_node_id))
            existing_edges.add((sim_node_id, node_id))
            edges_created += 1
            connections_made += 1

            # Track cross-type connections for insight generation
            if event_type != sim_event_type:
                cross_type_clusters.append({
                    "source_id": node_id,
                    "source_type": event_type,
                    "source_preview": content[:80],
                    "target_id": sim_node_id,
                    "target_type": sim_event_type,
                    "target_preview": sim_content[:80],
                    "similarity": round(similarity, 3),
                })

            report_lines.append(
                f"  - `{node_id[:16]}` ({event_type}) ↔ `{sim_node_id[:16]}` "
                f"({sim_event_type}) [{similarity:.0%}]"
            )

    # Phase 3: Generate insights from cross-type patterns
    insights_generated = 0
    insight_lines = []

    if cross_type_clusters and not dry_run:
        # Group cross-type connections by type pairs
        type_pairs: Dict[tuple, list] = {}
        for conn in cross_type_clusters:
            pair = tuple(sorted([conn["source_type"], conn["target_type"]]))
            type_pairs.setdefault(pair, []).append(conn)

        # Generate insight for type pairs with 3+ connections
        for pair, connections in type_pairs.items():
            if len(connections) >= 3:
                previews = [
                    f"- {c['source_preview']}... ↔ {c['target_preview']}..."
                    for c in connections[:5]
                ]
                insight_content = (
                    f"Cross-cutting pattern: {len(connections)} connections discovered "
                    f"between {pair[0]} and {pair[1]} memories.\n"
                    f"Examples:\n" + "\n".join(previews)
                )
                try:
                    _bridge.auto_capture(
                        content=insight_content,
                        event_type="advisor_insight",
                        metadata={
                            "category": "system_insight",
                            "source": "discover_connections",
                            "type_pair": list(pair),
                            "connection_count": len(connections),
                        },
                        entity_id="cairn",
                    )
                    insights_generated += 1
                    insight_lines.append(
                        f"  - {pair[0]} ↔ {pair[1]}: {len(connections)} connections"
                    )
                except Exception as e:
                    logger.debug("Failed to store cross-type insight: %s", e)

    # Format report
    mode = "(DRY RUN) " if dry_run else ""
    output = f"# Connection Discovery {mode}Report\n\n"
    output += f"**Scanned:** {len(candidates)} memories (last {lookback_hours}h)\n"
    output += f"**New edges:** {edges_created}\n"
    output += f"**Skipped (existing):** {edges_skipped}\n"
    output += f"**Cross-type insights:** {insights_generated}\n\n"

    if report_lines:
        output += "## Connections\n"
        output += "\n".join(report_lines[:30])
        if len(report_lines) > 30:
            output += f"\n  ... and {len(report_lines) - 30} more\n"
        output += "\n\n"

    if insight_lines:
        output += "## Cross-Type Patterns\n"
        output += "\n".join(insight_lines)
        output += "\n\n"

    if not report_lines and not insight_lines:
        output += "*No new connections found. Memories are already well-linked or too diverse.*\n"

    return output


# ---------------------------------------------------------------------------
# Public API -- System Insight Synthesis
# ---------------------------------------------------------------------------


def synthesize_system_insights(
    similarity_threshold: float = 0.50,
    min_cluster_size: int = 3,
    dry_run: bool = True,
) -> str:
    """Synthesize clusters of system insights into consolidated subsystem briefs.

    Like compact() but scoped to advisor_insight memories with category=system_insight.
    Consolidated nodes inherit the system_insight category and permanent TTL.

    Args:
        similarity_threshold: Jaccard similarity threshold for clustering (lower = broader clusters).
        min_cluster_size: Minimum insights in a cluster to trigger synthesis.
        dry_run: If True, report what would be synthesized without modifying anything.

    Returns:
        Formatted markdown report.
    """
    db = _bridge._get_store()
    all_insights = db.get_by_type("advisor_insight", limit=500)

    # Filter to system_insight category
    candidates = []
    for node in all_insights:
        meta = node.metadata or {}
        if meta.get("category") == "system_insight":
            candidates.append(node)

    if len(candidates) < min_cluster_size:
        return (
            f"# System Insight Synthesis\n\n"
            f"Only {len(candidates)} system insights found "
            f"(minimum cluster size: {min_cluster_size}). Nothing to synthesize.\n"
        )

    # Jaccard clustering (same algorithm as compact())
    def _norm(text: str) -> set:
        return {re.sub(r"[^\w]", "", w) for w in text.lower().split() if len(w) > 3}

    node_words = [(node, _norm(node.content)) for node in candidates]
    assigned: set = set()
    clusters: List[List] = []

    for i in range(len(node_words)):
        if len(assigned) >= len(node_words):
            break
        node_i, words_i = node_words[i]
        if node_i.id in assigned or not words_i:
            continue

        cluster = [node_i]
        assigned.add(node_i.id)

        for j in range(i + 1, len(node_words)):
            node_j, words_j = node_words[j]
            if node_j.id in assigned or not words_j:
                continue
            intersection = len(words_i & words_j)
            union = len(words_i | words_j)
            if union and (intersection / union) >= similarity_threshold:
                cluster.append(node_j)
                assigned.add(node_j.id)

        if len(cluster) >= min_cluster_size:
            clusters.append(cluster)

    if not clusters:
        return (
            f"# System Insight Synthesis\n\n"
            f"No clusters found with >= {min_cluster_size} similar system insights "
            f"at {similarity_threshold:.0%} similarity. Insights are already diverse.\n"
        )

    output = f"# System Insight Synthesis {'(DRY RUN)' if dry_run else 'Report'}\n\n"
    output += f"**System insights:** {len(candidates)}\n"
    output += f"**Clusters found:** {len(clusters)}\n\n"

    total_compacted = 0
    total_created = 0

    for ci, cluster in enumerate(clusters, 1):
        cluster.sort(key=lambda n: len(n.content), reverse=True)
        consolidated = _smart_extract(cluster)

        # Merge tags from all cluster members
        merged_tags: set = set()
        total_access = 0
        for node in cluster:
            merged_tags.update(str(t) for t in (node.metadata or {}).get("tags", []))
            total_access += getattr(node, "access_count", 0) or 0

        # Identify primary subsystem from most common tag
        primary_subsystem = max(merged_tags, key=lambda t: sum(
            1 for n in cluster if t in (n.metadata or {}).get("tags", [])
        )) if merged_tags else "general"

        output += f"## Cluster {ci}: {primary_subsystem} ({len(cluster)} insights)\n\n"
        output += f"**Summary:** {consolidated[:300]}...\n"
        for node in cluster[:5]:
            preview = node.content[:80]
            output += f"- `{node.id[:12]}`: {preview}\n"
        if len(cluster) > 5:
            output += f"- ... and {len(cluster) - 5} more\n"
        output += "\n"

        if not dry_run:
            compact_header = f"[Subsystem brief: {primary_subsystem}] "
            compact_content = compact_header + consolidated

            meta = {
                "event_type": "advisor_insight",
                "category": "system_insight",
                "source": "insight_synthesis",
                "subsystem": primary_subsystem,
                "compacted_from": [n.id for n in cluster],
                "compacted_count": len(cluster),
                "tags": sorted(merged_tags)[:15],
            }
            new_id = db.store(
                content=compact_content,
                metadata=meta,
                ttl_seconds=None,  # Permanent
                skip_inference=True,
            )
            db.update_node(new_id, access_count=total_access)

            # Mark originals as superseded
            for node in cluster:
                nmeta = dict(node.metadata or {})
                nmeta["superseded"] = True
                nmeta["superseded_by"] = new_id
                nmeta["synthesized_at"] = datetime.now(timezone.utc).isoformat()
                db.update_node(node.id, metadata=nmeta)
                db._log_forgetting_external(
                    node.id, node.content, "advisor_insight",
                    "insight_synthesis_superseded", {"superseded_by": new_id},
                )
                db.queue_cloud_delete_by_node_id(node.id)

            total_compacted += len(cluster)
            total_created += 1
            output += f"**Created:** `{new_id[:12]}` | **Superseded:** {len(cluster)} insights\n\n"

    output += "---\n"
    if dry_run:
        output += f"**Would synthesize:** {sum(len(c) for c in clusters)} insights into {len(clusters)} subsystem briefs\n"
    else:
        output += f"**Synthesized:** {total_compacted} insights into {total_created} subsystem briefs\n"

    return output


# ---------------------------------------------------------------------------
