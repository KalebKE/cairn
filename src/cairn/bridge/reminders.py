"""Time-based reminders (experimental) — peeled from bridge/__init__.py (Wave 1)."""
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json
from cairn.exceptions import ValidationError

logger = logging.getLogger("cairn.bridge")


# Regex for parsing human-friendly durations: "1h", "30m", "2d", "1w", "1d12h", "2 hours"
_DURATION_RE = re.compile(
    r"(?:(\d+)\s*w(?:eeks?)?)?\s*"
    r"(?:(\d+)\s*d(?:ays?)?)?\s*"
    r"(?:(\d+)\s*h(?:ours?|rs?)?)?\s*"
    r"(?:(\d+)\s*m(?:in(?:utes?|s?)?)?)?",
)


def parse_duration(text: str) -> timedelta:
    """Parse a human-friendly duration string into a timedelta.

    Supported formats: "1h", "30m", "2d", "1w", "1d12h", "2 hours", "30 minutes".
    Raises ValueError on invalid or zero duration.
    """
    text = text.strip().lower()
    m = _DURATION_RE.fullmatch(text)
    if not m or not any(m.groups()):
        raise ValidationError(f"Invalid duration: {text!r}. Use e.g. '1h', '30m', '2d', '1w', '1d12h'.")
    weeks = int(m.group(1) or 0)
    days = int(m.group(2) or 0)
    hours = int(m.group(3) or 0)
    minutes = int(m.group(4) or 0)
    td = timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes)
    if td.total_seconds() <= 0:
        raise ValidationError("Duration must be positive.")
    return td


def create_reminder(
    text: str,
    duration: str,
    context: Optional[str] = None,
    session_id: Optional[str] = None,
    project: Optional[str] = None,
) -> dict:
    """Create a time-based reminder.

    Stores directly via SQLiteStore.store() to bypass dedup/evolution —
    identical reminder text with different times should create separate entries.
    """
    td = parse_duration(duration)
    now = datetime.now(timezone.utc)
    remind_at = now + td

    meta = {
        "event_type": "reminder",
        "reminder_status": "pending",
        "remind_at": remind_at.isoformat(),
        "created_at_utc": now.isoformat(),
        "notified_out_of_session": False,
    }
    if context:
        meta["context"] = context
    if session_id:
        meta["session_id"] = session_id
    if project:
        meta["project"] = project

    # Include remind_at in content to avoid content-hash dedup
    # (same text at different times = different reminders)
    store_content = f"{text}\n[due: {remind_at.isoformat()}]"

    db = _bridge._get_store()
    node_id = db.store(
        content=store_content,
        session_id=session_id,
        metadata=meta,
        ttl_seconds=None,  # Permanent until dismissed
        skip_inference=True,  # Skip embedding dedup — same text, different times = different reminders
    )

    # Human-readable local time
    try:
        local_str = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
    except Exception as e:
        logger.debug("Timezone conversion failed: %s", e)
        local_str = remind_at.isoformat()

    return {
        "reminder_id": node_id,
        "text": text,
        "remind_at": remind_at.isoformat(),
        "remind_at_local": local_str,
        "duration": duration,
    }


def list_reminders(
    status: Optional[str] = None,
    include_dismissed: bool = False,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List reminders, sorted by overdue first then by remind_at ascending.

    Args:
        entity_id: If provided, only return reminders scoped to this entity.
    """
    db = _bridge._get_store()
    sql = "SELECT node_id, content, metadata, created_at FROM memories WHERE event_type = 'reminder'"
    params: list = []
    if entity_id:
        sql += " AND COALESCE(entity_id, '') = ?"
        params.append(entity_id)
    with db._lock:
        rows = db._conn.execute(sql, params).fetchall()

    now = datetime.now(timezone.utc)
    # Regex to strip the internal [due: ...] suffix from stored content
    _due_suffix_re = re.compile(r"\n\[due: [^\]]+\]$")

    results = []
    for node_id, content, meta_json, created_at in rows:
        try:
            meta = json.loads(meta_json) if isinstance(meta_json, str) else (meta_json or {})
        except (json.JSONDecodeError, TypeError):
            meta = {}

        r_status = meta.get("reminder_status", "pending")

        # Filter out superseded reminders (safety net for Phase 4.5)
        # But keep superseded reminders that are overdue — if the superseding
        # reminder also hasn't fired, the user still needs to be notified.
        if meta.get("superseded") and not include_dismissed and status != "all":
            remind_at_str_check = meta.get("remind_at", "")
            try:
                remind_at_check = datetime.fromisoformat(remind_at_str_check)
                if remind_at_check.tzinfo is None:
                    remind_at_check = remind_at_check.replace(tzinfo=timezone.utc)
                is_overdue_check = now >= remind_at_check and r_status == "pending"
            except (ValueError, TypeError):
                is_overdue_check = False
            if not is_overdue_check:
                continue

        # Filter by status
        if status and status != "all" and r_status != status:
            continue
        if not include_dismissed and not status and r_status == "dismissed":
            continue

        remind_at_str = meta.get("remind_at", "")
        try:
            remind_at = datetime.fromisoformat(remind_at_str)
            if remind_at.tzinfo is None:
                remind_at = remind_at.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            remind_at = now

        is_due = now >= remind_at
        is_overdue = is_due and r_status == "pending"
        time_until = remind_at - now

        try:
            remind_at_local = remind_at.astimezone().strftime("%Y-%m-%d %H:%M %Z")
        except Exception as e:
            logger.debug("Timezone conversion failed: %s", e)
            remind_at_local = remind_at.isoformat()

        # Strip internal [due: ...] suffix for clean display
        clean_text = _due_suffix_re.sub("", content)

        # Compute age since creation for staleness detection
        try:
            created_dt = datetime.fromisoformat(created_at) if created_at else now
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            pending_days = (now - created_dt).days
        except (ValueError, TypeError):
            pending_days = 0

        results.append({
            "id": node_id,
            "text": clean_text,
            "status": r_status,
            "remind_at": remind_at.isoformat(),
            "remind_at_local": remind_at_local,
            "is_due": is_due,
            "is_overdue": is_overdue,
            "pending_days": pending_days,
            "time_until": str(time_until).split(".")[0] if not is_due else "overdue",
            "context": meta.get("context"),
            "created_at": created_at,
        })

    # Sort: overdue first, then by remind_at ascending
    results.sort(key=lambda r: (not r["is_overdue"], r["remind_at"]))
    return results


def dismiss_reminder(reminder_id: str) -> Dict[str, Any]:
    """Dismiss a reminder by updating its status."""
    db = _bridge._get_store()
    node = db.get_node(reminder_id)
    if node is None:
        return {"success": False, "error": f"Reminder {reminder_id} not found"}

    meta = dict(node.metadata or {})
    if meta.get("event_type") != "reminder":
        return {"success": False, "error": f"{reminder_id} is not a reminder"}

    meta["reminder_status"] = "dismissed"
    meta["dismissed_at"] = datetime.now(timezone.utc).isoformat()
    db.update_node(reminder_id, metadata=meta)
    clean_text = re.sub(r"\n\[due: [^\]]+\]$", "", node.content)
    return {"success": True, "dismissed_id": reminder_id, "text": clean_text}


def get_due_reminders(
    mark_fired: bool = False,
    entity_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get all pending reminders that are due now.

    If mark_fired=True, transitions their status from 'pending' to 'fired'.
    If entity_id is provided, only returns reminders scoped to that entity.
    """
    all_reminders = list_reminders(status="pending", entity_id=entity_id)
    due = [r for r in all_reminders if r["is_due"]]

    if mark_fired and due:
        db = _bridge._get_store()
        now_iso = datetime.now(timezone.utc).isoformat()
        for r in due:
            node = db.get_node(r["id"])
            if node:
                meta = dict(node.metadata or {})
                meta["reminder_status"] = "fired"
                meta["fired_at"] = now_iso
                db.update_node(r["id"], metadata=meta)
                r["status"] = "fired"

    return due

