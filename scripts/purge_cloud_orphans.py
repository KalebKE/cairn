#!/usr/bin/env python3
"""One-time cleanup: delete Supabase memories that were TTL-expired locally
but never propagated to cloud (due to missing _queue_cloud_delete in cleanup_expired).

Usage:
    python3 scripts/purge_cloud_orphans.py          # dry-run (default)
    python3 scripts/purge_cloud_orphans.py --apply   # actually delete
"""

import json
import sqlite3
import sys
from pathlib import Path

# ── Load Supabase credentials ──────────────────────────────────────
secrets_path = Path.home() / ".cairn" / "secrets.json"
if not secrets_path.exists():
    print(f"ERROR: {secrets_path} not found")
    sys.exit(1)

secrets = json.loads(secrets_path.read_text())
sb_url = secrets["supabase_url"]
sb_key = secrets["supabase_key"]

try:
    from supabase import create_client
except ImportError:
    print("ERROR: supabase-py not installed. Run: pip install supabase")
    sys.exit(1)

client = create_client(sb_url, sb_key)

# ── Get all local_ids from local SQLite ────────────────────────────
db_path = Path.home() / ".cairn" / "cairn.db"
conn = sqlite3.connect(str(db_path))
local_ids = {row[0] for row in conn.execute("SELECT id FROM memories").fetchall()}
conn.close()
print(f"Local SQLite: {len(local_ids)} memories")

# ── Get all local_ids from Supabase (paginated) ───────────────────
cloud_rows = []
offset = 0
PAGE_SIZE = 1000
while True:
    result = (
        client.table("memories")
        .select("id, local_id, event_type, priority")
        .range(offset, offset + PAGE_SIZE - 1)
        .execute()
    )
    if not result.data:
        break
    cloud_rows.extend(result.data)
    if len(result.data) < PAGE_SIZE:
        break
    offset += PAGE_SIZE

print(f"Supabase:     {len(cloud_rows)} memories")

# ── Find orphans (in Supabase but not in local) ───────────────────
orphans = [r for r in cloud_rows if r["local_id"] not in local_ids]
print(f"Orphans:      {len(orphans)}")

if not orphans:
    print("No orphans found. Supabase is clean.")
    sys.exit(0)

# Breakdown
from collections import Counter

type_counts = Counter(r["event_type"] for r in orphans)
priority_counts = Counter(r.get("priority", 3) for r in orphans)
print("\nOrphan breakdown by type:")
for t, c in type_counts.most_common():
    print(f"  {t}: {c}")
print("\nOrphan breakdown by priority:")
for p in sorted(priority_counts):
    print(f"  priority {p}: {priority_counts[p]}")

# ── Delete orphans ─────────────────────────────────────────────────
dry_run = "--apply" not in sys.argv

if dry_run:
    print(f"\nDRY RUN: would delete {len(orphans)} orphaned memories from Supabase.")
    print("Run with --apply to execute.")
    sys.exit(0)

orphan_local_ids = [r["local_id"] for r in orphans]
orphan_uuids = [r["id"] for r in orphans]
deleted = 0

# Delete in batches of 50 (Supabase URL length limit)
for i in range(0, len(orphan_uuids), 50):
    batch_uuids = orphan_uuids[i : i + 50]
    batch_local_ids = orphan_local_ids[i : i + 50]

    # Delete embeddings first (foreign key)
    try:
        client.table("memory_embeddings").delete().in_("memory_id", batch_uuids).execute()
    except Exception as e:
        print(f"  Warning: embedding cleanup failed: {e}")

    # Delete memories
    result = client.table("memories").delete().in_("id", batch_uuids).execute()
    batch_deleted = len(result.data) if result.data else 0
    deleted += batch_deleted
    print(f"  Deleted batch {i // 50 + 1}: {batch_deleted} memories")

print(f"\nDone. Deleted {deleted} orphaned memories from Supabase.")
