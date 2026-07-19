#!/usr/bin/env python3
"""One-shot cleanup script for Cairn memory store.

Phase 1 of Memory Quality & Noise Reduction.
Operates directly on ~/.cairn/cairn.db.

Steps:
1. Delete near-duplicates (same first 80 chars appearing >2 times)
2. Delete junk "decisions" (BROADCAST, BREADCRUMB, task-notification, etc.)
3. Delete test data
4. Delete very short, low-value memories
5. Backfill tags on untagged memories
6. Rebuild FTS5 index
7. Run compaction on lesson_learned, error_pattern, decision types

Usage:
    python scripts/clean_store.py          # dry run
    python scripts/clean_store.py --apply  # actually delete
"""

import os
import re
import sqlite3
import sys
from pathlib import Path

# Ensure cairn is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


CAIRN_HOME = Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))
DB_PATH = CAIRN_HOME / "cairn.db"

# Noise patterns to delete (matched against content start)
NOISE_PATTERNS = [
    r"^\[BROADCAST from",
    r"^\[WORK BREADCRUMB",
    r"^\[SESSION START\]",
    r"^\[SESSION END\]",
    r"^\[ERROR in gnosis_do\]",
    r"^<task-notification>",
    r"^Decision: <task-notification>",
]

# Known test strings to delete
TEST_STRINGS = [
    "Smoke test after cleanup",
    "Test user pref 3",
    "Test auto-capture",
    "Cairn switchover test",
    "Test memory from migration verification",
    "Test auto-capture for verification",
    "Cairn native MCP test",
]

# Session IDs that contain only synthetic/test data
TEST_SESSION_PREFIXES = [
    "test-session",
    "phase2-seed",
    "wf-",
    "stale_test",
]


def backup_db(db_path: Path) -> Path:
    """Create a backup before cleanup."""
    from datetime import datetime
    backups_dir = CAIRN_HOME / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"pre-cleanup-{timestamp}.db"
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(backup_path))
    src.backup(dst)
    dst.close()
    src.close()
    print(f"  Backup created: {backup_path}")
    return backup_path


def step1_delete_near_duplicates(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete near-duplicates: same first 80 chars appearing >2 times."""
    print("\n=== Step 1: Delete near-duplicates ===")

    rows = conn.execute(
        """SELECT SUBSTR(content, 1, 80) AS prefix, COUNT(*) AS cnt,
                  GROUP_CONCAT(node_id, '|') AS ids
           FROM memories
           GROUP BY prefix
           HAVING cnt > 2
           ORDER BY cnt DESC"""
    ).fetchall()

    total_deleted = 0
    for prefix, count, id_str in rows:
        node_ids = id_str.split("|")
        # Keep the first one (oldest), delete the rest
        # Actually, keep the one with highest access_count
        best_id = None
        best_access = -1
        for nid in node_ids:
            row = conn.execute(
                "SELECT access_count FROM memories WHERE node_id = ?", (nid,)
            ).fetchone()
            if row and (row[0] or 0) > best_access:
                best_access = row[0] or 0
                best_id = nid

        to_delete = [nid for nid in node_ids if nid != best_id]
        print(f"  Prefix: {prefix[:60]}... ({count} copies, keeping {best_id[:12]})")

        if not dry_run:
            for nid in to_delete:
                # Get rowid for vec cleanup
                rid_row = conn.execute(
                    "SELECT id FROM memories WHERE node_id = ?", (nid,)
                ).fetchone()
                conn.execute("DELETE FROM memories WHERE node_id = ?", (nid,))
                conn.execute(
                    "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                    (nid, nid)
                )
                if rid_row:
                    try:
                        conn.execute(
                            "DELETE FROM memories_vec WHERE rowid = ?", (rid_row[0],)
                        )
                    except Exception:
                        pass
        total_deleted += len(to_delete)

    if not dry_run:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {total_deleted} duplicates")
    return total_deleted


def step2_delete_noise(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete junk decisions matching noise patterns."""
    print("\n=== Step 2: Delete noise patterns ===")

    all_rows = conn.execute(
        "SELECT node_id, id, content FROM memories"
    ).fetchall()

    compiled = [re.compile(p) for p in NOISE_PATTERNS]
    to_delete = []
    for node_id, rowid, content in all_rows:
        for pattern in compiled:
            if pattern.search(content):
                to_delete.append((node_id, rowid, content[:60]))
                break

    for node_id, rowid, preview in to_delete:
        print(f"  Delete: {preview}...")
        if not dry_run:
            conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            except Exception:
                pass

    if not dry_run and to_delete:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {len(to_delete)} noise memories")
    return len(to_delete)


def step3_delete_test_data(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete known test data."""
    print("\n=== Step 3: Delete test data ===")

    total = 0
    for test_str in TEST_STRINGS:
        rows = conn.execute(
            "SELECT node_id, id FROM memories WHERE content = ?", (test_str,)
        ).fetchall()
        if rows:
            print(f"  Found {len(rows)} copies of: {test_str}")
            if not dry_run:
                for node_id, rowid in rows:
                    conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
                    conn.execute(
                        "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id)
                    )
                    try:
                        conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception:
                        pass
            total += len(rows)

    if not dry_run and total:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {total} test memories")
    return total


def step3b_delete_test_sessions(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete memories from known test/synthetic session IDs."""
    print("\n=== Step 3b: Delete test session memories ===")

    total = 0
    for prefix in TEST_SESSION_PREFIXES:
        rows = conn.execute(
            "SELECT node_id, id, session_id, content FROM memories WHERE session_id LIKE ?",
            (f"{prefix}%",)
        ).fetchall()
        if rows:
            print(f"  Found {len(rows)} memories from sessions matching '{prefix}*'")
            if not dry_run:
                for node_id, rowid, sid, content in rows:
                    conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
                    conn.execute(
                        "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                        (node_id, node_id)
                    )
                    try:
                        conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                    except Exception:
                        pass
            total += len(rows)

    # Also delete memories with empty session_id that are low-value
    # (keep decisions and lessons with empty session — they may be from direct API calls)
    rows = conn.execute(
        """SELECT node_id, id, content, event_type FROM memories
           WHERE (session_id IS NULL OR session_id = '' OR session_id = 'unknown')
           AND event_type IN ('task_completion', 'error_pattern', 'session_summary', 'system_event')"""
    ).fetchall()
    if rows:
        print(f"  Found {len(rows)} low-value memories with empty/unknown session_id")
        if not dry_run:
            for node_id, rowid, content, etype in rows:
                conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
                conn.execute(
                    "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                    (node_id, node_id)
                )
                try:
                    conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
                except Exception:
                    pass
        total += len(rows)

    if not dry_run and total:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {total} test/empty-session memories")
    return total


def step4_delete_short_memories(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete very short, low-value memories (< 20 chars, not user_preference/user_fact)."""
    print("\n=== Step 4: Delete short memories (<20 chars) ===")

    rows = conn.execute(
        """SELECT node_id, id, content, event_type FROM memories
           WHERE LENGTH(content) < 20
           AND (event_type IS NULL OR event_type NOT IN ('user_preference', 'user_fact'))"""
    ).fetchall()

    for node_id, rowid, content, etype in rows:
        print(f"  Delete: [{etype}] {content!r}")
        if not dry_run:
            conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            except Exception:
                pass

    if not dry_run and rows:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {len(rows)} short memories")
    return len(rows)


def step4b_delete_raw_json_errors(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete error_pattern entries that contain raw JSON blobs."""
    print("\n=== Step 4b: Delete raw JSON error blobs ===")


    rows = conn.execute(
        """SELECT node_id, id, content FROM memories
           WHERE event_type = 'error_pattern'
           AND content LIKE 'Error encountered: {"%'"""
    ).fetchall()

    for node_id, rowid, content in rows:
        print(f"  Delete: {content[:60]}...")
        if not dry_run:
            conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            except Exception:
                pass

    if not dry_run and rows:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {len(rows)} raw JSON error blobs")
    return len(rows)


def step4c_delete_stale_preferences(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Delete stale auto-extracted preferences that are too short to be useful."""
    print("\n=== Step 4c: Delete stale short preferences ===")

    rows = conn.execute(
        """SELECT node_id, id, content FROM memories
           WHERE event_type = 'user_preference'
           AND LENGTH(content) < 60
           AND content LIKE '%[Preference]%'"""
    ).fetchall()

    for node_id, rowid, content in rows:
        print(f"  Delete: {content[:60]}...")
        if not dry_run:
            conn.execute("DELETE FROM memories WHERE node_id = ?", (node_id,))
            conn.execute(
                "DELETE FROM edges WHERE source_id = ? OR target_id = ?",
                (node_id, node_id)
            )
            try:
                conn.execute("DELETE FROM memories_vec WHERE rowid = ?", (rowid,))
            except Exception:
                pass

    if not dry_run and rows:
        conn.commit()
    print(f"  {'Would delete' if dry_run else 'Deleted'}: {len(rows)} stale preferences")
    return len(rows)


def step5_backfill_tags(conn: sqlite3.Connection, dry_run: bool) -> int:
    """Backfill tags on untagged memories."""
    print("\n=== Step 5: Backfill tags ===")

    from cairn import json_compat as json
    from cairn.bridge import _extract_tags

    rows = conn.execute(
        "SELECT node_id, content, metadata, project FROM memories"
    ).fetchall()

    updated = 0
    for node_id, content, metadata_json, project in rows:
        meta = json.loads(metadata_json) if metadata_json else {}
        existing_tags = meta.get("tags", [])
        if existing_tags:
            continue

        new_tags = _extract_tags(content, project)
        if not new_tags:
            continue

        meta["tags"] = new_tags
        if not dry_run:
            conn.execute(
                "UPDATE memories SET metadata = ? WHERE node_id = ?",
                (json.dumps(meta), node_id)
            )
        updated += 1

    if not dry_run and updated:
        conn.commit()
    print(f"  {'Would update' if dry_run else 'Updated'}: {updated} memories with tags")
    return updated


def step6_rebuild_fts(conn: sqlite3.Connection, dry_run: bool):
    """Rebuild FTS5 index."""
    print("\n=== Step 6: Rebuild FTS5 index ===")
    if dry_run:
        print("  Would rebuild FTS5 index")
        return
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
        conn.commit()
        print("  FTS5 index rebuilt successfully")
    except Exception as e:
        print(f"  FTS5 rebuild failed: {e}")


def step7_run_compaction(dry_run: bool):
    """Run compaction on key event types."""
    print("\n=== Step 7: Run compaction ===")
    if dry_run:
        print("  Would run compaction on lesson_learned, error_pattern, decision")
        return

    from cairn.bridge import compact

    for event_type in ("lesson_learned", "error_pattern", "decision"):
        print(f"\n  Compacting {event_type}...")
        result = compact(event_type=event_type, similarity_threshold=0.60, min_cluster_size=3)
        # Print just the summary line
        for line in result.split("\n"):
            if "Compacted:" in line or "No clusters" in line or "Only" in line:
                print(f"    {line.strip()}")
                break


def main():
    dry_run = "--apply" not in sys.argv

    if dry_run:
        print("=" * 60)
        print("DRY RUN — pass --apply to actually modify the database")
        print("=" * 60)
    else:
        print("=" * 60)
        print("APPLYING CHANGES — modifying database")
        print("=" * 60)

    if not DB_PATH.exists():
        print(f"Database not found: {DB_PATH}")
        sys.exit(1)

    # Backup before any changes
    if not dry_run:
        backup_db(DB_PATH)

    conn = sqlite3.connect(str(DB_PATH))

    # Load sqlite-vec for vec table cleanup
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        pass

    before = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"\nMemories before cleanup: {before}")

    total_deleted = 0
    total_deleted += step1_delete_near_duplicates(conn, dry_run)
    total_deleted += step2_delete_noise(conn, dry_run)
    total_deleted += step3_delete_test_data(conn, dry_run)
    total_deleted += step3b_delete_test_sessions(conn, dry_run)
    total_deleted += step4_delete_short_memories(conn, dry_run)
    total_deleted += step4b_delete_raw_json_errors(conn, dry_run)
    total_deleted += step4c_delete_stale_preferences(conn, dry_run)
    step5_backfill_tags(conn, dry_run)
    step6_rebuild_fts(conn, dry_run)

    after = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    print(f"\n{'=' * 60}")
    print(f"Memories before: {before}")
    print(f"Memories after deletions: {after}")
    print(f"Total deleted: {total_deleted}")

    conn.close()

    # Compaction runs through bridge (needs its own connection)
    if not dry_run:
        step7_run_compaction(dry_run)

    if dry_run:
        print("\nStep 7 compaction would run on: lesson_learned, error_pattern, decision")
        print("\nDRY RUN complete. Pass --apply to execute.")
    else:
        print("\nCleanup complete!")


if __name__ == "__main__":
    main()
