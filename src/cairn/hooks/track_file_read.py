#!/usr/bin/env python3
"""Cairn PreToolUse hook — Track file reads per session.

Fires before Read tool. Records file paths so pre_edit_surface.py can
warn when editing a file that was never read (read-before-write discipline).

Storage: ~/.cairn/session-reads/<session-id>.json (JSON array of paths)
"""
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path


def _cairn_home():
    """Cairn data home, honoring CAIRN_HOME (env-inline; no cairn import)."""
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))


def _log_hook_error(hook_name, error):
    try:
        log_path = _cairn_home() / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        tb = traceback.format_exc()
        data = f"[{timestamp}] {hook_name}: {error}\n{tb}\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    if tool_name != "Read":
        return

    session_id = os.environ.get("SESSION_ID", "")
    if not session_id:
        return

    try:
        input_data = json.loads(os.environ.get("TOOL_INPUT", "{}"))
    except (json.JSONDecodeError, TypeError):
        return

    file_path = input_data.get("file_path", "")
    if not file_path:
        return

    try:
        reads_dir = _cairn_home() / "session-reads"
        reads_dir.mkdir(parents=True, exist_ok=True)

        # Use a sanitized session ID for the filename
        safe_id = session_id.replace("/", "_").replace("..", "_")[:64]
        reads_file = reads_dir / f"{safe_id}.json"

        # Load existing reads
        existing = set()
        if reads_file.exists():
            try:
                existing = set(json.loads(reads_file.read_text()))
            except (json.JSONDecodeError, TypeError):
                existing = set()

        # Add this file path
        existing.add(file_path)

        # Write back
        data = json.dumps(sorted(existing))
        fd = os.open(str(reads_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception as e:
        _log_hook_error("track_file_read", e)


def _log_timing(hook_name, elapsed_ms):
    try:
        log_path = _cairn_home() / "hooks.log"
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        timestamp = datetime.now().isoformat(timespec="seconds")
        data = f"[{timestamp}] {hook_name}: OK ({elapsed_ms:.0f}ms)\n"
        fd = os.open(str(log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("track_file_read", (time.monotonic() - _t0) * 1000)
