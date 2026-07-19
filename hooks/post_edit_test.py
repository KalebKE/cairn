#!/usr/bin/env python3
"""Post-edit hook: auto-run tests for the project that was just modified."""

import json
import os
import shutil
import subprocess
import sys


def _resolve_python() -> str:
    """Resolve the Python interpreter path (same logic as cli.py)."""
    exe = sys.executable
    if exe and os.path.exists(exe) and "venv" not in exe:
        return exe
    which_py = shutil.which("python3")
    if which_py:
        return which_py
    fallback = "/opt/homebrew/bin/python3"
    if os.path.exists(fallback):
        return fallback
    return exe or "python3"


_PYTHON = _resolve_python()

# Project test configs: directory prefix -> (test command, cwd)
# Built-in default: Cairn (this project itself)
_DEFAULT_PROJECTS = {
    os.path.expanduser("~/Projects/cairn/"): {
        "cmd": [_PYTHON, "-m", "pytest", "tests/", "-x", "-q", "--tb=short", "--no-header"],
        "cwd": os.path.expanduser("~/Projects/cairn"),
        "name": "Cairn",
    },
}


def _load_projects():
    """Load project configs: built-in Cairn + optional ~/.cairn/post_edit_projects.json."""
    projects = dict(_DEFAULT_PROJECTS)
    config_path = os.path.expanduser("~/.cairn/post_edit_projects.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                extra = json.load(f)
            for prefix, config in extra.items():
                expanded = os.path.expanduser(prefix)
                if not expanded.endswith("/"):
                    expanded += "/"
                projects[expanded] = {
                    "cmd": config["cmd"],
                    "cwd": os.path.expanduser(config.get("cwd", prefix.rstrip("/"))),
                    "name": config.get("name", os.path.basename(prefix.rstrip("/"))),
                }
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Silently ignore malformed config
    return projects


PROJECTS = _load_projects()

# Files that don't need test runs (docs, configs, etc.)
SKIP_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".lock"}


def get_edited_file():
    """Extract the file path from the hook input."""
    try:
        hook_input = json.loads(sys.stdin.read())
    except (json.JSONDecodeError, EOFError):
        return None

    tool_input = hook_input.get("tool_input", {})
    return tool_input.get("file_path") or tool_input.get("notebook_path")


def should_skip(file_path):
    """Skip non-code files."""
    _, ext = os.path.splitext(file_path)
    return ext.lower() in SKIP_EXTENSIONS


def find_project(file_path):
    """Match edited file to a project."""
    resolved = os.path.realpath(file_path)
    for prefix, config in PROJECTS.items():
        resolved_prefix = os.path.realpath(prefix)
        if resolved.startswith(resolved_prefix):
            return config
    return None


def run_tests(project):
    """Run tests and return formatted output."""
    try:
        result = subprocess.run(
            project["cmd"],
            cwd=project["cwd"],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PATH": f"{os.path.dirname(_PYTHON)}:{os.environ.get('PATH', '')}"},
        )

        output_lines = (result.stdout + result.stderr).strip().split("\n")
        # Keep last 15 lines to stay concise
        tail = output_lines[-15:] if len(output_lines) > 15 else output_lines
        summary = "\n".join(tail)

        if result.returncode == 0:
            return f"[{project['name']}] Tests PASSED\n{summary}"
        else:
            return f"[{project['name']}] Tests FAILED (exit {result.returncode})\n{summary}"

    except subprocess.TimeoutExpired:
        return f"[{project['name']}] Tests TIMED OUT (60s limit)"
    except FileNotFoundError as e:
        return f"[{project['name']}] Test runner not found: {e}"


def main():
    file_path = get_edited_file()
    if not file_path:
        return

    if should_skip(file_path):
        return

    project = find_project(file_path)
    if not project:
        return

    result = run_tests(project)
    if result:
        # Output as user-visible message
        print(result, file=sys.stderr)


if __name__ == "__main__":
    main()
