"""Retrieval feedback for the Cairn bridge (peeled from __init__).

Records relevance feedback signals, batches them, drives the graduation
check that promotes well-reinforced memories, and backfills embeddings.
The store singleton late-binds through the package module.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json

logger = logging.getLogger("cairn.bridge.feedback")


# ---------------------------------------------------------------------------
# Public API -- Feedback
# ---------------------------------------------------------------------------


def record_feedback(
    memory_id: str,
    rating: str,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Record feedback on a surfaced memory."""
    db = _bridge._get_store()
    return db.record_feedback(node_id=memory_id, rating=rating, reason=reason)


def batch_record_feedback(items: List[tuple]) -> int:
    """Record feedback for multiple memories in a single transaction.

    Each item is (node_id, rating, reason). Returns count of updated memories.
    """
    db = _bridge._get_store()
    return db.batch_record_feedback(items)


def _check_graduation(memory_id: str) -> Optional[str]:
    """Check if a memory should graduate or decay based on diff-correlation history.

    Graduation: memory was diff-correlated (positive) in 2+ feedback signals -> promote priority.
    Decay: memory was surfaced 3+ times with zero correlation -> demote priority.

    Reads from the feedback_signals list stored in memory metadata by record_feedback().

    Returns "graduated", "decayed", or None.
    """
    db = _bridge._get_store()
    try:
        row = db._conn.execute(
            "SELECT metadata FROM memories WHERE node_id = ?",
            (memory_id,),
        ).fetchone()

        if not row or not row[0]:
            return None

        meta = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
        signals = meta.get("feedback_signals", [])

        if not signals:
            return None

        diff_positive = sum(
            1 for s in signals
            if s.get("rating") == "helpful" and s.get("reason") and "diff-correlated" in s["reason"]
        )
        surfaced_not_committed = sum(
            1 for s in signals
            if s.get("rating") == "unhelpful" and s.get("reason") and "not committed" in s["reason"]
        )

        if diff_positive >= 2:
            # Graduate: boost priority
            db._conn.execute(
                "UPDATE memories SET priority = MIN(COALESCE(priority, 3) + 1, 5) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "graduated"
        elif surfaced_not_committed >= 3 and diff_positive == 0:
            # Decay: reduce priority
            db._conn.execute(
                "UPDATE memories SET priority = MAX(COALESCE(priority, 3) - 1, 1) WHERE node_id = ?",
                (memory_id,),
            )
            db._conn.commit()
            return "decayed"

        return None
    except Exception as e:
        logger.debug("_check_graduation failed for %s: %s", memory_id[:12], e)
        return None


def backfill_embeddings(batch_size: int = 50) -> dict:
    """Backfill missing embeddings for memories not in memories_vec."""
    db = _bridge._get_store()
    return db.backfill_embeddings(batch_size=batch_size)
