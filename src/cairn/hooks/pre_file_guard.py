#!/usr/bin/env python3
"""Cairn PreToolUse hook — File guard for multi-agent coordination.

Triggered on Edit/Write/NotebookEdit. Blocks the tool call if the target file
is claimed by a DIFFERENT agent session. Self-claims are allowed.

Exit code 2 = block the tool call in Claude Code.
Exit code 0 = allow (including fail-open on any error).

Design: Fail-open — Cairn unavailable must never block edits.
"""
import json
import os
import sys
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


def _block_claimed(file_path, owner, owner_task):
    """Print block message and exit with code 2."""
    filename = os.path.basename(file_path)
    print(
        f"\n[FILE-GUARD] BLOCKED: {filename} is claimed by session {owner} ({owner_task}).\n"
        f"  Options:\n"
        f"    1. Wait for the other agent to finish and release\n"
        f"    2. Ask other agent to call cairn_file_release\n"
        f"    3. Force-claim via cairn_file_claim with force=true\n"
        f"    4. The claim expires automatically after 10 minutes of inactivity"
    )
    sys.exit(2)


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return

    try:
        input_data = json.loads(os.environ.get("TOOL_INPUT", "{}"))
    except (json.JSONDecodeError, TypeError):
        return

    file_path = input_data.get("file_path", input_data.get("notebook_path", ""))
    if not file_path:
        return

    # File-claim coordination (check_file/claim_file) was a Pro-only feature,
    # removed in this build. Nothing to enforce — fail open.
    return


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("pre_file_guard", (time.monotonic() - _t0) * 1000)
