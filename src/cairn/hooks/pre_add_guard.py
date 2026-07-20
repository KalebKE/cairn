#!/usr/bin/env python3
"""Cairn PreToolUse hook — Git add guard for multi-agent coordination.

BLOCKS `git add .`, `git add -A`, and `git commit -a` unconditionally (exit code 2).
BLOCKS staging files the agent didn't edit (not in own claim list) when coordination is active.
WARNS in solo mode when staging unclaimed files.

Prevents the root cause of mixed commits: broad staging captures pre-existing
dirty worktree changes from other agents or prior sessions.
"""
import json
import os
import re
import shlex
import time
import traceback
from datetime import datetime
from pathlib import Path


def _cairn_home():
    """Cairn data home, honoring CAIRN_HOME (env-inline; no cairn import)."""
    return Path(os.environ.get("CAIRN_HOME", str(Path.home() / ".cairn")))


_COMMIT_VALUE_OPTIONS = {
    "-m",
    "-F",
    "-C",
    "-c",
    "-t",
    "--message",
    "--file",
    "--author",
    "--date",
    "--fixup",
    "--squash",
}


def _token_starts_unclosed_quote(token: str) -> bool:
    return bool(token) and token[0] in {"'", '"'} and token.count(token[0]) % 2 == 1


def _commit_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        try:
            return shlex.split(command, posix=False)
        except ValueError:
            return command.split()


def _token_carries_commit_all_flag(token: str) -> bool:
    if token == "--all":
        return True
    if token.startswith("--"):
        return False
    return token.startswith("-") and "a" in token[1:]


def _commit_has_all_flag(command: str) -> bool:
    tokens = _commit_tokens(command)
    for i in range(len(tokens) - 1):
        if tokens[i] == "git" and tokens[i + 1] == "commit":
            args = tokens[i + 2 :]
            break
    else:
        return False

    skip_next = False
    for token in args:
        if skip_next:
            if _token_starts_unclosed_quote(token):
                return False
            skip_next = False
            continue
        if token == "--":
            return False
        if token in _COMMIT_VALUE_OPTIONS:
            skip_next = True
            continue
        if any(
            token.startswith(f"{option}=")
            for option in _COMMIT_VALUE_OPTIONS
            if option.startswith("--")
        ):
            continue
        if _token_carries_commit_all_flag(token):
            return True
    return False


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


def _is_broad_add(command):
    """Detect git add . / git add -A / git add --all / git commit -a patterns."""
    # git add . or git add -A or git add --all
    if re.search(r"\bgit\s+add\s+(\.\s*$|-A\b|--all\b)", command):
        return True
    # git commit -a / git commit -am / git commit --all
    if _commit_has_all_flag(command):
        return True
    return False


def _extract_add_paths(command):
    """Extract file paths from a git add command. Returns list of paths or None if not a git add."""
    m = re.search(r"\bgit\s+add\s+(.*)", command)
    if not m:
        return None
    args_str = m.group(1).strip()
    # Filter out flags
    paths = []
    for token in args_str.split():
        if token.startswith("-"):
            continue
        paths.append(token)
    return paths


def main():
    tool_name = os.environ.get("TOOL_NAME", "")
    tool_input = os.environ.get("TOOL_INPUT", "{}")

    if tool_name != "Bash":
        return

    try:
        input_data = json.loads(tool_input)
    except (json.JSONDecodeError, TypeError):
        return

    command = input_data.get("command", "")

    # Only trigger on git add or git commit -a
    is_git_add = re.search(r"\bgit\s+add\b", command)
    is_commit_a = _commit_has_all_flag(command)
    if not is_git_add and not is_commit_a:
        return

    session_id = os.environ.get("SESSION_ID", "")

    # --- BLOCK broad staging unconditionally ---
    if _is_broad_add(command):
        lines = [
            "[ADD-GUARD] BLOCKED: broad staging command detected.",
            f"  Command: {command[:120]}",
            "",
            "Never use `git add .`, `git add -A`, or `git commit -a`.",
            "Stage specific files by name: git add <file1> <file2> ...",
        ]


        print("\n".join(lines))
        exit(2)

    # --- For specific git add <files>, check against claims ---
    add_paths = _extract_add_paths(command)
    if not add_paths:
        return  # Not a parseable git add (maybe piped or complex)

    if not session_id:
        return  # No session tracking, can't check claims

    # Per-file claim checking against peers was a Pro-only feature, removed
    # here — the broad-staging block above is the only live guard.
    return



if __name__ == "__main__":
    _t0 = time.monotonic()
    main()
    _log_timing("pre_add_guard", (time.monotonic() - _t0) * 1000)
