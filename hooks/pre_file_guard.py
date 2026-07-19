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


def _log_hook_error(hook_name, error):
    try:
        log_path = Path.home() / ".cairn" / "hooks.log"
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
        log_path = Path.home() / ".cairn" / "hooks.log"
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


def _check_uncommitted_divergence(file_path, session_id):
    """Warn if file on disk diverges from HEAD (uncommitted changes from another session).

    This catches the case where another agent modified a file without committing,
    and the current agent is about to edit (and later commit) those foreign changes.
    """
    import subprocess

    try:
        # Get the repo root for this file
        repo_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=3,
            cwd=os.path.dirname(file_path) or ".",
        )
        if repo_root.returncode != 0:
            return  # Not in a git repo

        # Check if file has uncommitted changes vs HEAD
        diff = subprocess.run(
            ["git", "diff", "--numstat", "HEAD", "--", file_path],
            capture_output=True, text=True, timeout=3,
            cwd=repo_root.stdout.strip(),
        )
        if diff.returncode != 0 or not diff.stdout.strip():
            return  # No diff or error

        parts = diff.stdout.strip().split("\t")
        if len(parts) < 2:
            return

        additions = int(parts[0]) if parts[0] != "-" else 0
        deletions = int(parts[1]) if parts[1] != "-" else 0
        total_changes = additions + deletions

        if total_changes < 10:
            return  # Small diff, not worth warning

        # Check if this session already claimed this file (meaning we made these changes)
        try:
            from cairn.coordination import get_manager
            mgr = get_manager()
            info = mgr.check_file(file_path)
            if info.get("claimed") and info.get("session_id") == session_id:
                return  # We own these changes
        except Exception:
            pass

        filename = os.path.basename(file_path)
        print(
            f"\n[FILE-GUARD] WARNING: {filename} has {total_changes} uncommitted changes "
            f"(+{additions}/-{deletions}) vs HEAD.\n"
            f"  These may be from another agent's work-in-progress.\n"
            f"  If you edit and commit, you may inadvertently include or revert their changes.\n"
            f"  Run: git diff HEAD -- {filename}"
        )
    except Exception:
        pass  # Fail-open: never block on divergence check errors


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    if tool_name not in ("Edit", "Write", "NotebookEdit"):
        return

    session_id = os.environ.get("SESSION_ID", "")

    try:
        input_data = json.loads(os.environ.get("TOOL_INPUT", "{}"))
    except (json.JSONDecodeError, TypeError):
        return

    file_path = input_data.get("file_path", input_data.get("notebook_path", ""))
    if not file_path:
        return

    # Check for uncommitted divergence from HEAD (foreign changes on disk)
    _check_uncommitted_divergence(file_path, session_id)

    try:
        from cairn.coordination import get_manager
        mgr = get_manager()
        info = mgr.check_file(file_path)

        if info.get("claimed"):
            if session_id and info.get("session_id") == session_id:
                # Self-claim — allow
                return

            # Claimed by different session — BLOCK
            # Also blocks when no SESSION_ID (can't prove identity)
            owner = info.get("session_id", "unknown")[:20]
            owner_task = info.get("task") or "unknown task"
            _block_claimed(file_path, owner, owner_task)

        # Unclaimed — if we have a session_id, claim atomically to prevent TOCTOU race
        if session_id:
            result = mgr.claim_file(session_id, file_path, task="pre-edit guard claim")
            if result.get("conflict"):
                # Race lost — another agent claimed between check_file and claim_file
                owner = result["claimed_by"][:20]
                owner_task = result.get("task") or "unknown task"
                _block_claimed(file_path, owner, owner_task)
            # If claim failed for other reasons (session not registered, etc.), allow (fail-open)

        # No session_id + unclaimed → allow (true single-agent, no claims exist)

    except ImportError:
        # Cairn not installed — fail-open
        pass
    except Exception as e:
        # Any error — fail-open, never block when Cairn is unavailable
        _log_hook_error("pre_file_guard", e)


if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("pre_file_guard", (time.monotonic() - _t0) * 1000)
