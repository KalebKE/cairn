"""
Cairn Obsidian Export — Export memories as Obsidian-compatible markdown files.

Converts Cairn's SQLite-backed memories into a vault of interlinked markdown
files with YAML frontmatter, organized by event_type into subdirectories.

Usage (CLI):
    cairn export-obsidian [--output-dir ./cairn-vault] [--project PROJECT] [--limit 500]

Usage (Python):
    from cairn.obsidian_export import export_to_obsidian
    result = export_to_obsidian(output_dir="./cairn-vault")
"""

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("cairn.obsidian_export")

# Map event_type values to subdirectory names
_TYPE_TO_DIR = {
    "decision": "decisions",
    "lesson_learned": "lessons",
    "error_pattern": "errors",
    "user_preference": "preferences",
    "task_completion": "tasks",
    "checkpoint": "checkpoints",
    "session_summary": "sessions",
}
_DEFAULT_DIR = "memories"

# Characters not allowed in filenames
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename."""
    sanitized = _UNSAFE_FILENAME_RE.sub("_", name)
    # Collapse multiple underscores
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    # Truncate to reasonable length (keep room for .md extension)
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    return sanitized


def _format_tags(metadata: Dict[str, Any]) -> List[str]:
    """Extract tags from metadata."""
    tags = []
    # Pull from metadata tags field
    raw_tags = metadata.get("tags", [])
    if isinstance(raw_tags, list):
        tags.extend(raw_tags)
    elif isinstance(raw_tags, str):
        tags.extend(t.strip() for t in raw_tags.split(",") if t.strip())
    return tags


def _get_edges_for_node(conn, node_id: str) -> List[Dict[str, str]]:
    """Get all edges (relationships) for a given node_id."""
    rows = conn.execute(
        """SELECT source_id, target_id, edge_type, weight
           FROM edges
           WHERE source_id = ? OR target_id = ?""",
        (node_id, node_id),
    ).fetchall()

    edges = []
    for row in rows:
        source_id, target_id, edge_type, weight = row
        # The "other" node is the one that isn't us
        other_id = target_id if source_id == node_id else source_id
        edges.append({
            "target": other_id,
            "type": edge_type,
            "weight": weight,
            "direction": "outgoing" if source_id == node_id else "incoming",
        })
    return edges


def _edge_type_label(edge_type: str) -> str:
    """Human-readable label for edge types."""
    labels = {
        "related_to": "Related",
        "contradicts": "Contradicts",
        "supersedes": "Supersedes",
        "superseded_by": "Superseded by",
        "evolved_from": "Evolved from",
        "evolved_into": "Evolved into",
        "similar_to": "Similar",
    }
    return labels.get(edge_type, edge_type.replace("_", " ").title())


def _memory_to_markdown(
    node_id: str,
    content: str,
    metadata: Dict[str, Any],
    created_at: str,
    access_count: int,
    ttl_seconds: Optional[int],
    edges: List[Dict[str, str]],
) -> str:
    """Convert a single memory into Obsidian-compatible markdown."""
    event_type = metadata.get("event_type", "memory")
    tags = _format_tags(metadata)
    entity_id = metadata.get("entity_id", "")
    project = metadata.get("project", metadata.get("project_path", ""))
    session_id = metadata.get("session_id", "")
    agent_type = metadata.get("agent_type", "")

    # Compute strength from access_count (normalize to 0-1 range, capped)
    strength = min(1.0, access_count / 20.0) if access_count > 0 else 0.0

    # Build YAML frontmatter
    lines = ["---"]
    lines.append(f"id: {node_id}")
    lines.append(f"type: {event_type}")
    if tags:
        lines.append(f"tags: [{', '.join(tags)}]")
    lines.append(f"created: {created_at}")
    lines.append(f"strength: {strength:.2f}")
    if entity_id:
        lines.append(f"entity: {entity_id}")
    if project:
        lines.append(f"project: {project}")
    if ttl_seconds is not None:
        lines.append(f"ttl_seconds: {ttl_seconds}")
    lines.append("---")
    lines.append("")

    # Main content
    lines.append(content.strip())
    lines.append("")

    # Related section with wikilinks
    if edges:
        lines.append("## Related")
        for edge in edges:
            target_file = _sanitize_filename(edge["target"])
            label = _edge_type_label(edge["type"])
            lines.append(f"- [[{target_file}]] - {label}")
        lines.append("")

    # Metadata section
    meta_items = []
    if session_id:
        meta_items.append(f"- Source: session {session_id}")
    if agent_type:
        meta_items.append(f"- Agent: {agent_type}")
    if access_count > 0:
        meta_items.append(f"- Access count: {access_count}")

    if meta_items:
        lines.append("## Metadata")
        lines.extend(meta_items)
        lines.append("")

    return "\n".join(lines)


def _generate_index(
    stats: Dict[str, int],
    total: int,
    output_dir: Path,
    exported_at: str,
) -> str:
    """Generate the _index.md summary file."""
    lines = ["---"]
    lines.append("title: Cairn Index")
    lines.append(f"generated: {exported_at}")
    lines.append(f"total_memories: {total}")
    lines.append("---")
    lines.append("")
    lines.append("# Cairn Index")
    lines.append("")
    lines.append(f"**Total memories:** {total}")
    lines.append(f"**Exported:** {exported_at}")
    lines.append(f"**Output directory:** `{output_dir}`")
    lines.append("")
    lines.append("## By Type")
    lines.append("")
    lines.append("| Type | Count | Directory |")
    lines.append("|------|-------|-----------|")
    for event_type, count in sorted(stats.items(), key=lambda x: -x[1]):
        dir_name = _TYPE_TO_DIR.get(event_type, _DEFAULT_DIR)
        lines.append(f"| {event_type} | {count} | `{dir_name}/` |")
    lines.append("")
    return "\n".join(lines)


def export_to_obsidian(
    output_dir: str = "./cairn-vault",
    project: Optional[str] = None,
    limit: int = 0,
) -> Dict[str, Any]:
    """Export Cairn memories as Obsidian-compatible markdown files.

    Args:
        output_dir: Root directory for the exported vault.
        project: If set, only export memories for this project.
        limit: Max number of memories to export (0 = all).

    Returns:
        Dict with export statistics.
    """
    from cairn.bridge import _get_store

    db = _get_store()

    # Build query with optional filters
    query_parts = [
        "SELECT node_id, content, metadata, created_at,",
        "       access_count, last_accessed, ttl_seconds",
        "FROM memories",
    ]
    params: list = []
    where_clauses = []

    if project:
        where_clauses.append("(project = ? OR metadata LIKE ?)")
        params.extend([project, f'%"project_path":"%{project}%"%'])

    if where_clauses:
        query_parts.append("WHERE " + " AND ".join(where_clauses))

    query_parts.append("ORDER BY created_at")

    if limit > 0:
        query_parts.append("LIMIT ?")
        params.append(limit)

    sql = "\n".join(query_parts)

    # Execute query directly on the store's connection
    rows = db._conn.execute(sql, params).fetchall()

    # Set up output directory
    vault_dir = Path(output_dir) / "cairn-memories"
    vault_dir.mkdir(parents=True, exist_ok=True)

    exported_at = datetime.now(timezone.utc).isoformat()
    type_stats: Dict[str, int] = {}
    exported_count = 0
    edge_count = 0

    for row in rows:
        node_id = row[0]
        content = row[1]

        # Parse metadata
        meta_raw = row[2]
        if meta_raw:
            try:
                from cairn import json_compat as json
                metadata = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            except Exception:
                metadata = {}
        else:
            metadata = {}

        created_at = row[3] or ""
        access_count = row[4] or 0
        ttl_seconds = row[6]

        # Get edges for this memory
        edges = _get_edges_for_node(db._conn, node_id)
        edge_count += len(edges)

        # Determine subdirectory
        event_type = metadata.get("event_type", "memory")
        dir_name = _TYPE_TO_DIR.get(event_type, _DEFAULT_DIR)
        type_dir = vault_dir / dir_name
        type_dir.mkdir(parents=True, exist_ok=True)

        # Track stats
        type_stats[event_type] = type_stats.get(event_type, 0) + 1

        # Generate markdown
        md_content = _memory_to_markdown(
            node_id=node_id,
            content=content,
            metadata=metadata,
            created_at=created_at,
            access_count=access_count,
            ttl_seconds=ttl_seconds,
            edges=edges,
        )

        # Write file
        filename = _sanitize_filename(node_id) + ".md"
        filepath = type_dir / filename
        filepath.write_text(md_content, encoding="utf-8")
        exported_count += 1

    # Generate index file
    index_content = _generate_index(
        stats=type_stats,
        total=exported_count,
        output_dir=Path(output_dir).resolve(),
        exported_at=exported_at,
    )
    index_path = vault_dir / "_index.md"
    index_path.write_text(index_content, encoding="utf-8")

    result = {
        "output_dir": str(Path(output_dir).resolve()),
        "memories_exported": exported_count,
        "edge_links_created": edge_count,
        "type_breakdown": type_stats,
        "index_file": str(index_path),
        "exported_at": exported_at,
    }

    logger.info(
        "Exported %d memories (%d edges) to %s",
        exported_count,
        edge_count,
        output_dir,
    )

    return result
