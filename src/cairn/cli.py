"""Cairn CLI — Memory commands, setup, status, migration, and server management."""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from datetime import timedelta  # noqa: F401 — module-level contract (test_cli.py pins it)
from pathlib import Path
from cairn.paths import cairn_home

logger = logging.getLogger("cairn.cli")


def _use_json(args) -> bool:
    """Check if JSON output requested via --json flag or CAIRN_JSON=1 env var."""
    return getattr(args, "json", False) or os.environ.get("CAIRN_JSON") == "1"


def _parse_event_types_arg(value) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


CAIRN_DIR = cairn_home()
CAIRN_CACHE = Path.home() / ".cache" / "cairn"
GTE_MODEL_DIR = CAIRN_CACHE / "models" / "gte-modernbert-base-onnx"
BGE_MODEL_DIR = CAIRN_CACHE / "models" / "bge-small-en-v1.5-onnx"
MINILM_MODEL_DIR = CAIRN_CACHE / "models" / "all-MiniLM-L6-v2-onnx"
# Primary model dir — gte-modernbert-base; bge/minilm are legacy fallbacks
ONNX_MODEL_DIR = GTE_MODEL_DIR


CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"
SETTINGS_JSON_PATH = Path.home() / ".claude" / "settings.json"
DATA_DIR = Path(__file__).parent / "data"

CAIRN_BEGIN = "<!-- Cairn:BEGIN"
CAIRN_END = "<!-- Cairn:END -->"


def _python_has_cairn(python_path: str) -> bool:
    """Check if a Python interpreter has cairn installed."""
    try:
        result = subprocess.run(
            [python_path, "-c", "import cairn; import mcp"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _resolve_python_path() -> str:
    """Resolve the best Python interpreter path for hooks and MCP configs.

    Priority: first interpreter that can import cairn wins.
    1. sys.executable (even if inside a venv -- that's where cairn lives)
    2. 'python3' from PATH
    3. /opt/homebrew/bin/python3 (macOS Homebrew fallback)
    4. sys.executable as-is (best effort)
    """
    candidates = []

    exe = sys.executable
    if exe and Path(exe).exists():
        candidates.append(exe)

    which_py = shutil.which("python3")
    if which_py and which_py not in candidates:
        candidates.append(which_py)

    fallback = "/opt/homebrew/bin/python3"
    if Path(fallback).exists() and fallback not in candidates:
        candidates.append(fallback)

    for candidate in candidates:
        if _python_has_cairn(candidate):
            return candidate

    # No candidate has cairn -- return sys.executable as best effort
    return exe or "python3"


def _inject_claude_md(*, dry_run: bool = False):
    """Inject or update the Cairn block in ~/.claude/CLAUDE.md (idempotent).

    Selects Pro or Core fragment based on available modules. Never overwrites
    user content — only touches the managed Cairn block between markers.

    Args:
        dry_run: If True, print what would change without writing.
    """
    fragment = (DATA_DIR / "claude-md-fragment.md").read_text()

    if CLAUDE_MD_PATH.exists():
        content = CLAUDE_MD_PATH.read_text()
    else:
        CLAUDE_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
        content = ""

    if CAIRN_BEGIN in content:
        # Replace existing block (upgrade path)
        pattern = re.compile(
            r"<!-- Cairn:BEGIN[^\n]*-->.*?<!-- Cairn:END -->",
            re.DOTALL,
        )
        new_content = pattern.sub(fragment.rstrip(), content)
        if new_content == content:
            print("  CLAUDE.md: Cairn block already up to date")
            return
        if dry_run:
            print("  CLAUDE.md: would update Cairn block (dry-run)")
            return
        CLAUDE_MD_PATH.write_text(new_content)
        print("  CLAUDE.md: Cairn block updated")
    else:
        # First time — back up existing file if it has content
        if content.strip():
            backup_path = CLAUDE_MD_PATH.with_suffix(".md.pre-cairn")
            if not backup_path.exists():
                if dry_run:
                    print(f"  CLAUDE.md: would back up to {backup_path.name} (dry-run)")
                    print("  CLAUDE.md: would append Cairn block (dry-run)")
                    return
                backup_path.write_text(content)
                print(f"  CLAUDE.md: backed up existing file to {backup_path.name}")
        elif dry_run:
            print("  CLAUDE.md: would create with Cairn block (dry-run)")
            return
        separator = "\n" if content and not content.endswith("\n") else ""
        CLAUDE_MD_PATH.write_text(content + separator + fragment)
        print("  CLAUDE.md: Cairn block appended")


def _has_commercial_modules() -> bool:
    """Check if a plugin ships a full hooks manifest (extension seam)."""
    try:
        from cairn.plugins import discover_plugins

        for plugin in discover_plugins():
            if plugin.HOOKS_JSON:
                return True
    except Exception as e:
        logger.debug("Plugin hooks check failed: %s", e)
    return False


def _inject_settings_hooks(hooks_src: Path):
    """Inject Cairn hook entries into ~/.claude/settings.json (idempotent).

    Uses hooks-core.json for core-only installs, or hooks.json (full) when
    commercial modules are available. Supports both old format (single dict
    per event) and new format (list of dicts per event) in hooks.json manifest.
    """
    if _has_commercial_modules():
        hooks_file = "hooks.json"
    else:
        hooks_file = "hooks-core.json"
    manifest = json.loads((DATA_DIR / hooks_file).read_text())

    # Determine the python path: prefer the running interpreter
    python_path = _resolve_python_path()

    if SETTINGS_JSON_PATH.exists():
        try:
            settings = json.loads(SETTINGS_JSON_PATH.read_text())
        except json.JSONDecodeError:
            print("  WARNING: settings.json is malformed, skipping hook injection")
            return
    else:
        SETTINGS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        settings = {}

    if "hooks" not in settings:
        settings["hooks"] = {}

    configured = 0
    skipped = 0
    repaired = 0

    for event, hook_defs in manifest.items():
        # Normalize: old format is a single dict, new format is a list of dicts
        if isinstance(hook_defs, dict):
            hook_defs = [hook_defs]

        for hook_def in hook_defs:
            script = hook_def["script"]
            command = f"{python_path} {hooks_src / script}"

            # Build a unique identifier for this hook (handles "fast_hook.py session_start" etc.)
            # Strip .py and use the full script string for matching
            script_key = script.replace(".py", "").replace(" ", "_")

            # Check if this Cairn hook is already wired (match by script_key in command)
            existing_idx = None
            existing_hook_idx = None
            if event in settings["hooks"]:
                for i, entry in enumerate(settings["hooks"][event]):
                    for j, h in enumerate(entry.get("hooks", [])):
                        cmd = h.get("command", "")
                        if script_key in cmd.replace(".py", "").replace(" ", "_"):
                            existing_idx = i
                            existing_hook_idx = j
                            break
                    if existing_idx is not None:
                        break

            if existing_idx is not None:
                # Hook exists — check if the path is correct
                existing_cmd = settings["hooks"][event][existing_idx]["hooks"][existing_hook_idx]["command"]
                if existing_cmd == command:
                    skipped += 1
                    continue
                # Path changed (broken or outdated) — replace it
                settings["hooks"][event][existing_idx]["hooks"][existing_hook_idx]["command"] = command
                repaired += 1
                continue

            # Build the hook entry
            entry = {
                "hooks": [
                    {
                        "command": command,
                        "timeout": hook_def["timeout"],
                        "type": "command",
                    }
                ],
                "matcher": hook_def.get("matcher", ""),
            }

            if event not in settings["hooks"]:
                settings["hooks"][event] = []
            settings["hooks"][event].append(entry)
            configured += 1

    SETTINGS_JSON_PATH.write_text(json.dumps(settings, indent=2) + "\n")

    if configured > 0:
        print(f"  settings.json: {configured} hook(s) configured")
    if repaired > 0:
        print(f"  settings.json: {repaired} hook(s) repaired (paths updated)")
    if skipped > 0:
        print(f"  settings.json: {skipped} hook(s) already configured")
    if configured == 0 and skipped == 0:
        print("  settings.json: hooks configured")


# ---------------------------------------------------------------------------
# CLI Memory Commands — direct terminal access to Cairn
# ---------------------------------------------------------------------------


def _format_age(created_at) -> str:
    """Format a datetime as relative age string (e.g. '2d ago', '1w ago')."""
    if not created_at:
        return ""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        # Naive datetime — assume UTC
        created_at = created_at.replace(tzinfo=timezone.utc)
    delta = now - created_at
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    days = seconds // 86400
    if days < 7:
        return f"{days}d ago"
    if days < 30:
        return f"{days // 7}w ago"
    return f"{days // 30}mo ago"


def cmd_query(args):
    """Search memories by semantic similarity or exact phrase."""
    query_text = " ".join(args.query_text)
    if not query_text.strip():
        print("Usage: cairn query <search text>", file=sys.stderr)
        sys.exit(1)

    limit = getattr(args, "limit", 10)
    use_json = _use_json(args)
    exact = getattr(args, "exact", False)

    start = time.monotonic()

    if exact:
        # For --json, use the store directly
        if use_json:
            from cairn.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            out = []
            for node in results:
                out.append(
                    {
                        "id": node.id,
                        "content": node.content,
                        "event_type": (node.metadata or {}).get("event_type", "memory"),
                        "created_at": node.created_at.isoformat() if node.created_at else "",
                        "tags": (node.metadata or {}).get("tags", []),
                    }
                )
            print(json.dumps({"results": out, "count": len(out), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            from cairn.bridge import _get_store

            db = _get_store()
            results = db.phrase_search(phrase=query_text, limit=limit)
            elapsed = time.monotonic() - start
            if results:
                from cairn.cli_ui import print_table

                rows = []
                for node in results:
                    etype = (node.metadata or {}).get("event_type", "memory")
                    preview = node.content[:120].replace("\n", " ")
                    age = _format_age(node.created_at)
                    mid = node.id[:12] if node.id else ""
                    rows.append(("--", etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["dim", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')
    else:
        from cairn.bridge import query_structured

        results = query_structured(query_text, limit=limit)
        elapsed = time.monotonic() - start

        if use_json:
            print(json.dumps({"results": results, "count": len(results), "elapsed_s": round(elapsed, 3)}, indent=2))
        else:
            if results:
                from cairn.cli_ui import print_table

                rows = []
                for r in results:
                    relevance = f"{int(r.get('relevance', 0) * 100)}%"
                    etype = r.get("event_type", "memory")
                    preview = r.get("content", "")[:120].replace("\n", " ")
                    age = ""
                    if r.get("created_at"):
                        try:
                            dt = datetime.fromisoformat(r["created_at"])
                            age = _format_age(dt)
                        except (ValueError, TypeError):
                            pass
                    mid = r.get("id", "")[:12]
                    rows.append((relevance, etype, preview, age, mid))
                print_table(
                    None, ["Score", "Type", "Preview", "Age", "ID"], rows, styles=["cyan", "bold", None, "dim", "dim"]
                )
                print(f"\n{len(results)} result(s) ({elapsed:.2f}s)")
            else:
                print(f'No results for "{query_text}" ({elapsed:.2f}s)')


_CLI_TYPE_MAP = {
    "memory": "memory",
    "lesson": "lesson_learned",
    "decision": "decision",
    "error": "error_pattern",
    "task": "task_completion",
    "preference": "user_preference",
}


def cmd_store(args):
    """Store a memory with a specified type."""
    content = " ".join(args.content)
    if not content.strip():
        print("Usage: cairn store <text> [-t TYPE]", file=sys.stderr)
        sys.exit(1)

    cli_type = getattr(args, "type", "memory")
    event_type = _CLI_TYPE_MAP.get(cli_type, cli_type)

    from cairn.bridge import store

    store(content=content, event_type=event_type)

    if _use_json(args):
        print(json.dumps({"status": "ok", "content": content[:200], "type": cli_type}, indent=2))
    else:
        print(f"Stored [{cli_type}]: {content[:80]}")


def cmd_remember(args):
    """Store a permanent user preference."""
    text = " ".join(args.text)
    if not text.strip():
        print("Usage: cairn remember <text>", file=sys.stderr)
        sys.exit(1)

    from cairn.bridge import remember

    remember(text=text)

    if _use_json(args):
        print(json.dumps({"status": "ok", "content": text[:200]}, indent=2))
    else:
        print(f"Remembered: {text[:120]}")


def cmd_timeline(args):
    """Show memory timeline grouped by day."""
    days = getattr(args, "days", 7)
    use_json = _use_json(args)

    if use_json:
        from cairn.bridge import _get_store

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        out = {}
        for day, memories in (data or {}).items():
            out[day] = []
            for m in memories:
                out[day].append(
                    {
                        "id": m.id,
                        "content": m.content[:200],
                        "event_type": (m.metadata or {}).get("event_type", "memory"),
                        "created_at": m.created_at.isoformat() if m.created_at else "",
                    }
                )
        print(json.dumps(out, indent=2))
    else:
        from cairn.bridge import _get_store
        from cairn.cli_ui import print_header, print_table

        db = _get_store()
        data = db.get_timeline(days=days, limit_per_day=20)
        if not data:
            print(f"No memories in the last {days} days.")
            return

        total = sum(len(v) for v in data.values())
        print_header(f"Memory Timeline ({total} memories, last {days} days)")

        for day in sorted(data.keys(), reverse=True):
            memories = data[day]
            rows = []
            for m in memories:
                etype = (m.metadata or {}).get("event_type", "memory")
                preview = m.content[:100].replace("\n", " ")
                time_str = m.created_at.strftime("%H:%M") if m.created_at else ""
                mid = m.id[:12] if m.id else ""
                rows.append((time_str, etype, preview, mid))
            print_table(
                f"{day} ({len(memories)})",
                ["Time", "Type", "Preview", "ID"],
                rows,
                styles=["dim", "bold", None, "dim"],
            )


# ---------------------------------------------------------------------------
# Setup & Doctor
# ---------------------------------------------------------------------------


def _setup_claude_code(errors_ref: list, hooks_src: Path, hooks_only: bool = False, dry_run: bool = False):
    """Claude Code-specific setup: MCP registration, hooks, CLAUDE.md.

    If hooks_only=True, skips MCP server registration entirely. Hooks call
    bridge.py directly (no MCP process needed), saving ~600MB RAM per session.
    """
    if not hooks_only:
        # Register MCP server with Claude Code
        print("  Registering MCP server with Claude Code...")
        python_path = _resolve_python_path()
        try:
            result = subprocess.run(
                ["claude", "mcp", "add", "-s", "user", "cairn", "--", python_path, "-m", "cairn.server.mcp_server"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                print("  MCP server registered successfully")
            else:
                errors_ref.append(1)
                print(f"  ERROR: MCP registration returned code {result.returncode}")
                if result.stderr:
                    print(f"  {result.stderr.strip()}")
                print(f"  Register manually: claude mcp add -s user cairn -- {python_path} -m cairn.server.mcp_server")
        except FileNotFoundError:
            errors_ref.append(1)
            print("  ERROR: 'claude' command not found in PATH.")
            print("  Install Claude Code: https://docs.anthropic.com/en/docs/claude-code")
            print(f"  Or register manually: claude mcp add -s user cairn -- {python_path} -m cairn.server.mcp_server")
        except Exception as e:
            errors_ref.append(1)
            print(f"  ERROR: MCP registration failed: {e}")
            print(f"  Register manually: claude mcp add -s user cairn -- {python_path} -m cairn.server.mcp_server")
    else:
        print("  Skipping MCP server registration (--hooks-only mode)")
        print("  Hooks will call bridge.py directly (~600MB RAM saved per session)")
        print("  Note: cairn_store, cairn_query etc. won't be available as Claude tools")
        print("  To add MCP later: cairn setup --client claude-code")

    # Install hooks
    hooks_dst = Path.home() / ".claude" / "scripts"
    hooks_dst.mkdir(parents=True, exist_ok=True)

    hook_files = ["session_start.py", "session_stop.py", "surface_memories.py", "auto_capture.py"]
    for hook in hook_files:
        src = hooks_src / hook
        dst = hooks_dst / f"cairn-{hook}"
        if src.exists():
            shutil.copy2(src, dst)
            if sys.platform != "win32":
                dst.chmod(0o755)
            print(f"  Installed hook: {dst.name}")
        else:
            print(f"  WARNING: Hook source not found: {src}")

    # Wire hooks into settings.json
    try:
        _inject_settings_hooks(hooks_src)
    except Exception as e:
        errors_ref.append(1)
        print(f"  ERROR: Failed to configure settings.json hooks: {e}")

    # Inject Cairn block into CLAUDE.md
    try:
        _inject_claude_md(dry_run=dry_run)
    except Exception as e:
        print(f"  WARNING: Failed to update CLAUDE.md: {e}")


def _mcp_server_json_snippet() -> str:
    """Return the MCP server JSON config snippet for manual copy-paste."""
    python_path = _resolve_python_path()
    return json.dumps({
        "cairn": {
            "command": python_path,
            "args": ["-m", "cairn.server.mcp_server"],
            "env": {"CAIRN_CLIENT": "{{CLIENT}}"},
        }
    }, indent=2)


def _setup_generic_mcp_client(client_name: str):
    """Print MCP server config for clients that lack a `mcp add` command."""
    snippet = _mcp_server_json_snippet().replace("{{CLIENT}}", client_name)
    print(f"\n  === {client_name.title()} MCP Configuration ===")
    print(f"  Add this to your {client_name} MCP settings:\n")
    print(snippet)
    print("\n  Set the environment variable for client detection:")
    print(f"    export CAIRN_CLIENT={client_name}")
    print(f"\n  NOTE: Hooks are not available for {client_name}.")
    print("  Memory capture requires manual cairn_store calls or MCP tool usage.")
    print("  Session start/stop hooks will not fire automatically.\n")


def _resolve_hooks_src() -> Path:
    """Resolve the hooks source directory.

    Priority:
    1. src/cairn/hooks/ inside the installed package (pip install)
    2. hooks/ at repo root (development checkout)
    """
    pkg_hooks = Path(__file__).parent / "hooks"
    if pkg_hooks.exists() and (pkg_hooks / "fast_hook.py").exists():
        return pkg_hooks
    repo_hooks = Path(__file__).parent.parent.parent / "hooks"
    if repo_hooks.exists() and (repo_hooks / "fast_hook.py").exists():
        return repo_hooks
    return pkg_hooks  # will fail gracefully downstream


def cmd_hooks(args):
    """Manage Claude Code hooks: setup, path, doctor."""
    sub = getattr(args, "hooks_command", None)

    hooks_src = _resolve_hooks_src()
    python_path = _resolve_python_path()

    if sub == "setup":
        print("Cairn hooks setup")
        print(f"  Python:  {python_path}")
        print(f"  Hooks:   {hooks_src}")

        if not (hooks_src / "fast_hook.py").exists():
            print("\n  ERROR: fast_hook.py not found at expected location.")
            print("  Try reinstalling: pip install cairn[server]")
            sys.exit(1)

        try:
            _inject_settings_hooks(hooks_src)
            print("\n  Hooks configured in ~/.claude/settings.json")
        except Exception as e:
            print(f"\n  ERROR: Failed to configure hooks: {e}")
            sys.exit(1)

        try:
            _inject_claude_md()
        except Exception as e:
            print(f"  WARNING: Failed to update CLAUDE.md: {e}")

        print("\n  Done! Restart Claude Code for changes to take effect.")

    elif sub == "path":
        # Machine-readable: just print the path
        print(hooks_src)

    elif sub == "doctor":
        print("Cairn hooks doctor")
        print(f"  Python:     {python_path}")
        print(f"  Hooks dir:  {hooks_src}")

        # Check fast_hook.py exists
        fh = hooks_src / "fast_hook.py"
        if fh.exists():
            print(f"  fast_hook:  OK ({fh})")
        else:
            print(f"  fast_hook:  MISSING ({fh})")

        # Check settings.json has hooks
        if SETTINGS_JSON_PATH.exists():
            try:
                settings = json.loads(SETTINGS_JSON_PATH.read_text())
                hooks = settings.get("hooks", {})
                events_with_cairn = 0
                broken_paths = []
                for event, entries in hooks.items():
                    for entry in entries:
                        for h in entry.get("hooks", []):
                            cmd = h.get("command", "")
                            if "cairn" in cmd.lower() or "fast_hook" in cmd:
                                events_with_cairn += 1
                                # Check if the path in the command exists
                                parts = cmd.split()
                                if len(parts) >= 2:
                                    py_path = parts[0]
                                    script_path = parts[1]
                                    if not Path(py_path).exists():
                                        broken_paths.append(f"{event}: Python not found: {py_path}")
                                    if not Path(script_path).exists():
                                        broken_paths.append(f"{event}: Script not found: {script_path}")

                print(f"  settings:   {events_with_cairn} Cairn hook events configured")
                if broken_paths:
                    print(f"  BROKEN:     {len(broken_paths)} path issue(s)")
                    for bp in broken_paths:
                        print(f"    - {bp}")
                    print("\n  Fix with: cairn hooks setup")
                else:
                    print("  paths:      All OK")
            except json.JSONDecodeError:
                print("  settings:   MALFORMED (~/.claude/settings.json)")
        else:
            print("  settings:   NOT FOUND (~/.claude/settings.json)")
            print("\n  Fix with: cairn hooks setup")

    else:
        print("Usage: cairn hooks {setup|path|doctor}")
        print()
        print("  setup   Configure hooks in ~/.claude/settings.json")
        print("  path    Print the hooks directory path")
        print("  doctor  Check hook configuration health")


def _setup_cursor(errors_ref: list, hooks_src: Path):
    """Cursor-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("cursor")


def _setup_windsurf(errors_ref: list, hooks_src: Path):
    """Windsurf-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("windsurf")


def _setup_cline(errors_ref: list, hooks_src: Path):
    """Cline-specific setup: print MCP config for manual paste."""
    _setup_generic_mcp_client("cline")


def _setup_codex(errors_ref: list, hooks_src: Path):
    """OpenAI Codex CLI setup: merge MCP server into ~/.codex/config.toml."""
    print("  Configuring OpenAI Codex CLI...")
    config_path = Path.home() / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    python_path = _resolve_python_path()

    # Read existing TOML content (preserve manually since tomllib is read-only)
    lines = []
    if config_path.exists():
        try:
            lines = config_path.read_text().splitlines(keepends=True)
        except OSError as e:
            errors_ref.append(e)
            print(f"  ERROR: Could not read {config_path}: {e}")
            return

    # Check if cairn is already configured
    content = "".join(lines)
    if "mcp_servers.cairn" in content:
        print(f"  cairn already configured in {config_path}")
        return

    # Build the TOML block to insert
    toml_block = (
        '\n[mcp_servers.cairn]\n'
        f'command = "{python_path}"\n'
        'args = ["-m", "cairn.server.mcp_server"]\n'
    )

    # Insert before the first [projects.*] section if present, otherwise append
    insert_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("[projects."):
            insert_idx = i
            break

    if insert_idx is not None:
        lines.insert(insert_idx, toml_block + "\n")
    else:
        lines.append(toml_block)

    try:
        config_path.write_text("".join(lines))
        print(f"  Wrote MCP config to {config_path}")
        print("  Restart Codex CLI to activate Cairn.")
        print("  NOTE: Hooks (auto-capture, memory surfacing) are only available with Claude Code.")
    except OSError as e:
        errors_ref.append(e)
        print(f"  ERROR: Could not write {config_path}: {e}")


def _setup_antigravity(errors_ref: list, hooks_src: Path):
    """Antigravity IDE setup: write MCP config to ~/.gemini/antigravity/mcp_config.json."""
    print("  Configuring Antigravity IDE...")
    config_path = Path.home() / ".gemini" / "antigravity" / "mcp_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    python_path = _resolve_python_path()
    mcp_entry = {
        "mcpServers": {
            "cairn": {
                "command": python_path,
                "args": ["-m", "cairn.server.mcp_server"],
            }
        }
    }

    # Read or create config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            config = {}
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    if "cairn" in config.get("mcpServers", {}):
        print(f"  cairn already configured in {config_path}")
        return

    config["mcpServers"]["cairn"] = {
        "command": python_path,
        "args": ["-m", "cairn.server.mcp_server"],
    }

    try:
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"  Wrote MCP config to {config_path}")
        print("  Restart Antigravity to activate Cairn.")
        print("  NOTE: Hooks (auto-capture, memory surfacing) are only available with Claude Code.")
    except OSError as e:
        errors_ref.append(e)
        print(f"  ERROR: Could not write {config_path}: {e}")


def _setup_venv(errors_ref: list, hooks_src: Path):
    """Venv setup: print MCP and CLI paths for manual client configuration."""
    python_path = _resolve_python_path()
    cairn_bin = shutil.which("cairn") or str(Path(python_path).parent / "cairn")

    print("\n  Cairn venv configuration:")
    print(f"  Python:  {python_path}")
    print(f"  CLI:     {cairn_bin}")
    print("\n  MCP server (stdio):")
    print(f"    command: {python_path}")
    print('    args:    ["-m", "cairn.server.mcp_server"]')
    print("\n  JSON config block (copy into your client):")
    config = json.dumps({
        "cairn": {
            "command": python_path,
            "args": ["-m", "cairn.server.mcp_server"],
        }
    }, indent=2)
    for line in config.splitlines():
        print(f"    {line}")


def _setup_claude_desktop(errors_ref: list, hooks_src: Path, dry_run: bool = False):
    """Claude Desktop setup: inject MCP entry into claude_desktop_config.json."""
    # Determine config path
    if sys.platform == "darwin":
        config_path = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            errors_ref.append(1)
            print("  ERROR: APPDATA not set, cannot find Claude Desktop config")
            return
        config_path = Path(appdata) / "Claude" / "claude_desktop_config.json"

    python_path = _resolve_python_path()
    mcp_entry = {
        "command": python_path,
        "args": ["-m", "cairn.server.mcp_server"],
    }

    # Read or create config
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: Could not parse existing config ({e}), creating new")
            config = {}
    else:
        config = {}

    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Check if already configured and up to date
    existing = config["mcpServers"].get("cairn")
    if existing and existing.get("command") == python_path:
        print("  Claude Desktop: cairn already configured")
    else:
        config["mcpServers"]["cairn"] = mcp_entry
        if dry_run:
            print(f"  Claude Desktop: would write MCP entry to {config_path} (dry-run)")
        else:
            # Back up existing config
            if config_path.exists():
                backup = config_path.with_suffix(".json.bak")
                if not backup.exists():
                    shutil.copy2(config_path, backup)
                    print(f"  Backed up config to {backup.name}")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  Claude Desktop: MCP server registered in {config_path}")

    # Inject CLAUDE.md (reuse existing function)
    try:
        _inject_claude_md(dry_run=dry_run)
    except Exception as e:
        print(f"  WARNING: Failed to update CLAUDE.md: {e}")


def cmd_setup(args):
    """Set up Cairn: create dirs, download model, initialize DB. Optionally configure a client."""
    # ── Python version check ──────────────────────────────────────────
    if sys.version_info < (3, 11):
        print(f"ERROR: Cairn requires Python 3.11 or higher (you have {sys.version_info.major}.{sys.version_info.minor}).")
        print("Install Python 3.11+: https://www.python.org/downloads/")
        sys.exit(1)

    client = getattr(args, "client", None)
    hooks_only = getattr(args, "hooks_only", False)
    dry_run = getattr(args, "dry_run", False)
    errors = []
    download_model = getattr(args, "download_model", False)

    # --hooks-only implies claude-code client
    if hooks_only and client is None:
        client = "claude-code"

    # ── Auto-detect Claude Code if --client not specified ─────────────
    if client is None and shutil.which("claude"):
        client = "claude-code"
        print("Setting up Cairn (Claude Code detected)...")
    elif client is None:
        print("Setting up Cairn...")
        print("  NOTE: Claude Code CLI not found in PATH.")
        print("  Skipping MCP registration and hooks. To add them later:")
        print("    cairn setup --client claude-code")
        print()
    else:
        print("Setting up Cairn...")

    # Track what we did for the summary
    steps_done = []
    steps_skipped = []
    files_modified = []

    # 1. Create directories with restricted permissions
    CAIRN_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    (CAIRN_DIR / "graphs").mkdir(exist_ok=True, mode=0o700)
    print(f"  Created {CAIRN_DIR}")
    steps_done.append("Storage directory")

    # 2. Models. --download-model eagerly fetches the intended embedder +
    # reranker now (verified). Without it, setup does NOT download a fallback
    # (the old all-MiniLM default was exactly the silent-wrong-model footgun) —
    # the intended model auto-downloads on first use instead.
    from cairn.embedding import INTENDED_EMBEDDING_MODEL
    from cairn.model_download import download_model as _dl_model
    from cairn.model_download import is_model_present
    from cairn.reranker import INTENDED_RERANKER_MODEL

    if download_model:
        try:
            _dl_model(INTENDED_EMBEDDING_MODEL)
            _dl_model(INTENDED_RERANKER_MODEL)
            steps_done.append(f"Models ({INTENDED_EMBEDDING_MODEL} + {INTENDED_RERANKER_MODEL})")
        except Exception as e:
            errors.append(e)
            print(f"  ERROR: model download failed: {e}")
    else:
        if is_model_present(INTENDED_EMBEDDING_MODEL):
            print(f"  ONNX model: {INTENDED_EMBEDDING_MODEL} already present")
            steps_done.append("Embedding model (already present)")
        elif (BGE_MODEL_DIR / "model.onnx").exists():
            print(f"  ONNX model: bge-small-en-v1.5 (legacy 384-dim) at {BGE_MODEL_DIR}")
            print("  TIP: 'cairn setup --download-model' then 'cairn migrate-embeddings' to upgrade")
            steps_done.append("Embedding model (already present, legacy)")
        else:
            print(f"  Embedding model ({INTENDED_EMBEDDING_MODEL}, ~570MB) will download "
                  "automatically on first use.")
            print("  TIP: run 'cairn setup --download-model' to fetch it now.")
            steps_done.append("Embedding model (deferred to first use)")

    # 4. Create default config
    config_path = CAIRN_DIR / "config.json"
    if not config_path.exists():
        from cairn.embedding import INTENDED_EMBEDDING_MODEL
        from cairn.reranker import INTENDED_RERANKER_MODEL

        config = {
            "storage_path": str(CAIRN_DIR),
            "model_dir": str(ONNX_MODEL_DIR),
            # Intended models recorded at store creation — a secondary anchor
            # so drift ("running X, this store expects Y") is auditable.
            "embedding_model": INTENDED_EMBEDDING_MODEL,
            "reranker_model": INTENDED_RERANKER_MODEL,
            "version": "0.1.0",
            "entity_scoping": {"enabled": False},
        }
        config_path.write_text(json.dumps(config, indent=2))
        config_path.chmod(0o600)
        print(f"  Created config at {config_path}")
    steps_done.append("Config file")

    # 5. Client-specific setup
    hooks_src = _resolve_hooks_src()
    _CLIENT_SETUP = {
        "cursor": _setup_cursor,
        "windsurf": _setup_windsurf,
        "cline": _setup_cline,
        "codex": _setup_codex,
        "antigravity": _setup_antigravity,
        "venv": _setup_venv,
    }
    if client == "claude-code":
        _setup_claude_code(errors, hooks_src, hooks_only=hooks_only, dry_run=dry_run)
        if hooks_only:
            steps_done.append("MCP server registration (skipped — hooks-only)")
        else:
            steps_done.append("MCP server registration")
        steps_done.append("Hooks (settings.json)")
        steps_done.append("CLAUDE.md instructions")
        files_modified.append("~/.claude/settings.json (hook entries)")
        files_modified.append("~/.claude/CLAUDE.md (Cairn instruction block)")
        if not hooks_only:
            files_modified.append("~/.claude.json (MCP server entry)")
    elif client == "claude-desktop":
        _setup_claude_desktop(errors, hooks_src, dry_run=dry_run)
        steps_done.append("Claude Desktop MCP registration")
        steps_done.append("CLAUDE.md instructions")
        if sys.platform == "darwin":
            config_display = "~/Library/Application Support/Claude/claude_desktop_config.json"
        else:
            config_display = "%APPDATA%/Claude/claude_desktop_config.json"
        files_modified.append(f"{config_display} (MCP server entry)")
        files_modified.append("~/.claude/CLAUDE.md (Cairn instruction block)")
    elif client in _CLIENT_SETUP:
        _CLIENT_SETUP[client](errors, hooks_src)
        steps_done.append(f"MCP config snippet ({client})")
        steps_skipped.append(f"Hooks (not available for {client})")
    else:
        steps_skipped.append("MCP server registration (no client specified)")
        steps_skipped.append("Hooks (no client specified)")
        python_path = _resolve_python_path()
        print("\n  MCP server ready. Add to your client:")
        print(f"    Command: {python_path} -m cairn.server.mcp_server")
        print("    Transport: stdio")

    # ── Summary ───────────────────────────────────────────────────────
    print()
    if errors:
        print(f"Cairn setup completed with {len(errors)} error(s).")
        for step in steps_done:
            print(f"  [OK] {step}")
        for err in errors:
            print(f"  [FAIL] {err}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        print("\nRun 'cairn doctor' to diagnose issues.")
        sys.exit(1)
    else:
        print("Cairn setup complete!")
        try:
            from cairn.telemetry import track_event
            track_event("setup_complete", {"client": client or "none"})
        except Exception:
            pass
        for step in steps_done:
            print(f"  [OK] {step}")
        for step in steps_skipped:
            print(f"  [SKIP] {step}")
        if files_modified:
            print("\n  Files modified outside ~/.cairn/:")
            for f in files_modified:
                print(f"    {f}")
        print(f"\n  Storage: {CAIRN_DIR}")
        print("  Run 'cairn doctor' to verify.")

        # GitHub star ask -- always show on first setup
        print()
        print("  If Cairn is useful, please star us on GitHub:")
        print("    https://github.com/TracqiTechnology/cairn")


def cmd_status(args):
    """Show Cairn status: memory count, store size, model status."""
    use_json = _use_json(args)
    data = {}

    # SQLite database (primary backend)
    db_path = CAIRN_DIR / "cairn.db"
    if db_path.exists():
        import sqlite3

        size_mb = db_path.stat().st_size / (1024 * 1024)
        data["backend"] = "sqlite"
        data["database"] = str(db_path)
        data["size_mb"] = round(size_mb, 2)
        try:
            conn = sqlite3.connect(str(db_path), timeout=30)
            count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            data["memories"] = count
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                data["vector_search"] = True
            except Exception:
                data["vector_search"] = False
            conn.close()
        except Exception as e:
            data["error"] = str(e)
    else:
        store_path = CAIRN_DIR / "store.jsonl"
        if store_path.exists():
            size_mb = store_path.stat().st_size / (1024 * 1024)
            with open(store_path) as f:
                line_count = sum(1 for _ in f)
            data["backend"] = "jsonl"
            data["store"] = str(store_path)
            data["memories"] = line_count
            data["size_mb"] = round(size_mb, 2)
        else:
            data["backend"] = None
            data["memories"] = 0

    # Model — resolve through the runtime path (honors CAIRN_ONNX_MODEL_DIR and
    # picks up gte/legacy/fallback correctly) instead of hard-probing two dirs.
    try:
        from cairn.embedding import _get_onnx_model_dir, get_embedding_model_info

        resolved_dir = _get_onnx_model_dir()
        data["model"] = get_embedding_model_info().get("model_name") if resolved_dir else None
        if resolved_dir:
            mp = Path(resolved_dir) / "model.onnx"
            if mp.exists():
                data["model_size_mb"] = round(mp.stat().st_size / (1024 * 1024), 0)
    except Exception:
        data["model"] = None
    try:
        import cairn.reranker as _rr

        data["reranker"] = _rr._resolve_reranker_model()[0]
    except Exception:
        pass
    try:
        from cairn.model_health import model_health_warnings

        mw = model_health_warnings()
        if mw:
            data["model_warnings"] = mw
    except Exception:
        pass

    # Profile
    profile_path = CAIRN_DIR / "profile.json"
    data["has_profile"] = profile_path.exists()

    # Config version
    config_path = CAIRN_DIR / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            data["version"] = config.get("version", "unknown")
        except Exception:
            pass

    if use_json:
        print(json.dumps(data, indent=2, default=str))
        return

    # Rich/plain output (existing behavior preserved)
    from cairn.cli_ui import print_header, print_kv

    print_header("Cairn Status")
    kv: list[tuple[str, str]] = []

    if data.get("backend") == "sqlite":
        kv.append(("Backend", "SQLite"))
        kv.append(("Database", data.get("database", "")))
        kv.append(("Size", f"{data.get('size_mb', 0):.2f} MB"))
        kv.append(("Memories", str(data.get("memories", 0))))
        if data.get("vector_search"):
            kv.append(("Vector search", "enabled (sqlite-vec)"))
        else:
            kv.append(("Vector search", "text-only fallback"))
        if "error" in data:
            kv.append(("Error", data["error"]))
    elif data.get("backend") == "jsonl":
        kv.append(("Backend", "JSONL (legacy)"))
        kv.append(("Store", data.get("store", "")))
        kv.append(("Memories", str(data.get("memories", 0))))
        kv.append(("Size", f"{data.get('size_mb', 0):.2f} MB"))
        kv.append(("Tip", "Run 'cairn migrate-db' to upgrade to SQLite"))
    else:
        kv.append(("Store", "not initialized"))
        kv.append(("Memories", "0"))

    if data.get("model"):
        model_label = data["model"]
        if data.get("model_size_mb"):
            model_label += f" ONNX ({data['model_size_mb']:.0f} MB)"
        kv.append(("Model", model_label))
        if data["model"] == "all-MiniLM-L6-v2":
            kv.append(("Tip", "Run 'cairn setup --download-model' to upgrade to gte-modernbert-base"))
    else:
        kv.append(("Model", "not downloaded"))
        kv.append(("Tip", "Run 'cairn setup' to download"))

    # Legacy graphs
    graphs_dir = CAIRN_DIR / "graphs"
    if graphs_dir.exists():
        graph_files = list(graphs_dir.glob("*.json"))
        if graph_files:
            kv.append(("Legacy graphs", f"{len(graph_files)} files (run 'cairn migrate-db' to convert)"))

    if data.get("has_profile"):
        kv.append(("Profile", str(CAIRN_DIR / "profile.json")))

    if data.get("version"):
        kv.append(("Version", data["version"]))

    print_kv(kv)

    print()


def cmd_reingest(args):
    """Reingest JSONL entries into the SQLite database."""
    store_path = CAIRN_DIR / "store.jsonl"
    pre_sqlite = CAIRN_DIR / "store.jsonl.pre-sqlite"
    # Check both current and backed-up JSONL
    if pre_sqlite.exists() and not store_path.exists():
        store_path = pre_sqlite
    if not store_path.exists():
        print(f"No JSONL store found at {CAIRN_DIR}")
        print("  Nothing to reingest (SQLite is the primary store now)")
        return

    from cairn.bridge import reingest

    result = reingest(store_path=store_path)

    print("\nReingest complete:")
    print(f"  Ingested:   {result.get('ingested', 0)}")
    print(f"  Duplicates: {result.get('duplicates', 0)}")
    print(f"  Skipped:    {result.get('skipped', 0)}")
    print(f"  Errors:     {result.get('errors', 0)}")
    print(f"  Total:      {result.get('total', 0)}")

    from cairn.bridge import status as cairn_status

    s = cairn_status()
    print(f"\nNode count: {s.get('node_count', 0)}")


def cmd_rollup(args):
    """LLM-assisted episodic rollup: synthesize aging task_completions /
    session_summaries into durable project_history memories."""
    from cairn.bridge import _get_store
    from cairn.embedding import generate_embedding, is_embedding_degraded
    from cairn.rollup import rollup_pending, find_pending_windows

    # Guard: never roll up while embeddings are degraded — the synthesis memory
    # would get an unqueryable hash vector.
    generate_embedding("rollup health probe")
    if is_embedding_degraded():
        print("Embedding backend degraded (hash fallback) — rollup skipped.")
        sys.exit(1)

    db = _get_store()
    min_age = getattr(args, "min_age_days", 30)
    dry_run = getattr(args, "dry_run", False)

    pending = find_pending_windows(db, min_age_days=min_age)
    if not pending:
        print("No pending rollup windows.")
        return
    print(f"Pending windows: {len(pending)}")
    result = rollup_pending(db, min_age_days=min_age, dry_run=dry_run,
                            max_windows=getattr(args, "max_windows", 6))
    for w in result["windows"]:
        line = f"  {w['window']}  {w['project'] or '(no project)'}  rows={w['count']}  -> {w['status']}"
        print(line)
        if dry_run and w.get("synthesis"):
            print("    --- synthesis preview ---")
            for ln in w["synthesis"].splitlines():
                print(f"    {ln}")
    print(f"{'Would roll' if dry_run else 'Rolled'}: {result['windows_rolled']} window(s), "
          f"{result['rows_rolled']} row(s); llm-skipped: {result['skipped_llm']}")


def cmd_link(args):
    """Discover and link related memories (vector-similarity edge builder)."""
    from cairn.bridge import discover_connections
    print(discover_connections(
        lookback_hours=getattr(args, "hours", 48),
        dry_run=getattr(args, "dry_run", False),
    ))


def cmd_gc(args):
    """Storage garbage collection: re-embed hash-tainted rows, drain the
    write-only cloud_delete_queue, and VACUUM to return freed pages."""
    from cairn.bridge import _get_store
    db = _get_store()

    r = db.reembed_hash_tainted()
    if r.get("skipped_degraded"):
        print("  reembed: skipped (embedding backend degraded)")
    else:
        print(f"  reembed: {r['reembedded']} re-embedded, {r['remaining']} remaining")

    with db._lock:
        drained = db._conn.execute("DELETE FROM cloud_delete_queue").rowcount
        db._commit()
    print(f"  cloud_delete_queue drained: {drained} row(s) (no cloud sync in this build)")

    size_before = os.path.getsize(db.db_path) if hasattr(db, "db_path") and os.path.exists(str(db.db_path)) else 0
    with db._lock:
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db._conn.execute("VACUUM")
    size_after = os.path.getsize(db.db_path) if size_before else 0
    if size_before:
        print(f"  VACUUM: {size_before/1048576:.2f} MB -> {size_after/1048576:.2f} MB")
    else:
        print("  VACUUM: done")


def cmd_consolidate(args):
    """Run memory consolidation: deduplicate and prune old entries."""
    prune_days = getattr(args, "prune_days", 30)
    print(f"Running Cairn consolidation (prune_days={prune_days})...")

    from cairn.bridge import _get_store, deduplicate

    db = _get_store()
    node_count_before = db.node_count()
    print(f"  Nodes before: {node_count_before}")

    # Run deduplication via bridge
    result = deduplicate()
    merged = result.get("merged", 0) if isinstance(result, dict) else 0

    # Prune expired
    expired = db.cleanup_expired()

    # Evict old low-access entries if requested
    evicted = 0
    if prune_days > 0:
        evicted = db.evict_lru(count=0)  # 0 = only expired

    node_count_after = db.node_count()

    print("\nConsolidation complete:")
    print(f"  Duplicates merged: {merged}")
    print(f"  Expired pruned:    {expired}")
    print(f"  Evicted:           {evicted}")
    print(f"  Nodes after:       {node_count_after}")


def cmd_migrate_home(args):
    """Migrate the legacy ~/.omega data dir to ~/.cairn (omega.db -> cairn.db,
    markers, config, backups, and the ONNX model cache). Non-destructive: the
    legacy dir is left in place and an existing cairn.db is never overwritten."""
    from cairn._compat import migrate_home, needs_home_migration
    if not needs_home_migration():
        if (CAIRN_DIR / "cairn.db").exists():
            print(f"Cairn store already present at {CAIRN_DIR / 'cairn.db'} — nothing to migrate.")
        else:
            print("No legacy ~/.omega store found — nothing to migrate.")
        return
    print("Migrating ~/.omega -> ~/.cairn ...")
    migrate_home(verbose=True)
    print("Done. Verify with `cairn status`, then remove ~/.omega when satisfied.")


def cmd_migrate_embeddings(args):
    """Rebuild the vec table + all embeddings for the current model/dimension.

    Safe by construction: backs up the DB first, refuses to run while MCP
    servers are live (their in-process model would write old-dim vectors),
    and verifies count parity before declaring success.
    """
    import sqlite3 as _sqlite3
    from datetime import datetime as _dt

    db_path = CAIRN_DIR / "cairn.db"
    if not db_path.exists():
        print(f"No store at {db_path} — nothing to migrate.")
        return 1

    # 1. No live writers: a running server holds the OLD model and would
    #    produce wrong-dim vectors the rebuilt table rejects.
    from cairn.server.pid_registry import list_registered_pids

    live = list_registered_pids()
    if live and not getattr(args, "force", False):
        print(f"Refusing: {len(live)} live Cairn server(s) registered "
              f"(pids: {[e.get('pid') for e in live]}).")
        print("Close those sessions (or pass --force if you know they are stale) and retry.")
        return 1

    from cairn.embedding import get_embedding_model_info
    from cairn.sqlite_store import EMBEDDING_DIM

    print(f"Target: dim={EMBEDDING_DIM}")

    # 2. Backup via the online-backup API (consistent even mid-WAL).
    backups_dir = CAIRN_DIR / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    ts = _dt.now().strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"pre-migrate-embeddings-{ts}.db"
    src = _sqlite3.connect(str(db_path))
    dst = _sqlite3.connect(str(backup_path))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    print(f"Backup: {backup_path}")

    # 3. Rebuild under the mismatch bypass (the guard would refuse the open).
    os.environ["CAIRN_ALLOW_DIM_MISMATCH"] = "1"
    try:
        from cairn.sqlite_store import SQLiteStore

        store = SQLiteStore(db_path=db_path)
        try:
            before = store.node_count()
            info = get_embedding_model_info()
            stats = store.rebuild_vec_table()
            vec_count = store._conn.execute(
                "SELECT COUNT(*) FROM memories_vec"
            ).fetchone()[0]
            after = store.node_count()
        finally:
            store.close()
    finally:
        os.environ.pop("CAIRN_ALLOW_DIM_MISMATCH", None)

    print(f"Model: {info.get('model_name')} | memories: {before} -> {after} | "
          f"re-embedded: {stats.get('updated')} (failed: {stats.get('failed')}) | "
          f"vec rows: {vec_count}")
    if after != before or stats.get("failed"):
        print("WARNING: count parity or re-embed failures — inspect before trusting. "
              f"Backup retained at {backup_path}")
        return 1

    # 4. Reopen WITHOUT the bypass: the guard itself verifies the new dim.
    store = SQLiteStore(db_path=db_path)
    store.close()
    print(f"Migration complete: store is {EMBEDDING_DIM}-dim. Restart any Cairn "
          f"sessions so servers pick up the new model.")
    return 0


def cmd_migrate_db(args):
    """Migrate from JSON graphs + JSONL to SQLite backend."""
    force = getattr(args, "force", False)
    from cairn.migrate_to_sqlite import migrate

    report = migrate(force=force)
    if report.get("warnings"):
        for w in report["warnings"]:
            print(f"  WARNING: {w}")


def cmd_backup(args):
    """Back up cairn.db to ~/.cairn/backups/ with timestamp."""
    db_path = CAIRN_DIR / "cairn.db"
    if not db_path.exists():
        print("No cairn.db found — nothing to back up.")
        return

    backups_dir = CAIRN_DIR / "backups"
    backups_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup_path = backups_dir / f"cairn-{timestamp}.db"

    import sqlite3
    from cairn.crypto import secure_connect

    src = sqlite3.connect(str(db_path), timeout=30)
    dst = secure_connect(backup_path)
    src.backup(dst)
    dst.close()
    src.close()

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup saved: {backup_path} ({size_mb:.2f} MB)")

    # Rotate — keep only the 5 most recent backups
    backups = sorted(backups_dir.glob("cairn-*.db"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[5:]:
        old.unlink()
        print(f"  Rotated old backup: {old.name}")


def cmd_compact(args):
    """Cluster and summarize related memories to reduce noise."""
    event_type = getattr(args, "type", "lesson_learned")
    threshold = getattr(args, "threshold", 0.60)
    dry_run = getattr(args, "dry_run", False)

    print(f"Compacting {event_type} (threshold={threshold}, dry_run={dry_run})...")

    from cairn.bridge import compact

    result = compact(
        event_type=event_type,
        similarity_threshold=threshold,
        dry_run=dry_run,
    )
    print(result)


def cmd_stats(args):
    """Show memory type distribution and health summary."""
    use_json = _use_json(args)
    use_card = getattr(args, "card", False)

    if use_card:
        from cairn.bridge import stats_card_data
        from cairn.cli_ui import print_stats_card

        data = stats_card_data()
        if use_json:
            print(json.dumps(data, indent=2, default=str))
        else:
            print_stats_card(data)
        return

    from cairn.bridge import type_stats, status as cairn_status

    stats = type_stats()
    health = cairn_status()

    if use_json:
        print(json.dumps({"types": stats, "health": health}, indent=2, default=str))
        return

    from cairn.cli_ui import print_bar_chart, print_header, print_kv

    total = sum(stats.values())
    print_header("Cairn Stats")
    print_kv(
        [
            ("Memories", str(total)),
            ("DB size", f"{health.get('db_size_mb', 0):.2f} MB"),
            ("Edges", str(health.get("edge_count", 0))),
            ("Backend", health.get("backend", "unknown")),
        ]
    )
    print()
    items = sorted(stats.items(), key=lambda x: -x[1])
    print_bar_chart(items, title="Type Distribution", total=total)


def cmd_activity(args):
    """Show recent session activity: sessions, tasks, insights, claims."""
    days = getattr(args, "days", 7)
    use_json = _use_json(args)

    from cairn.bridge import get_activity_summary

    data = get_activity_summary(days=days)

    if use_json:
        print(json.dumps(data, indent=2, default=str))
        return

    from cairn.cli_ui import print_header, print_section, print_table

    print_header(f"Cairn Activity (last {days} days)")

    # Sessions
    print_section("Active Sessions")
    if data["sessions"]:
        rows = []
        for s in data["sessions"]:
            project = s.get("project") or ""
            rows.append(
                (
                    s.get("session_id") or "",
                    project.split("/")[-1] or project,
                    (s.get("task") or "")[:50],
                    (s.get("started_at") or "")[:19],
                    s.get("status") or "",
                )
            )
        print_table(
            None,
            ["Session", "Project", "Task", "Started", "Status"],
            rows,
            styles=["cyan", "bold", None, "dim", "green"],
        )
    else:
        print("  No active sessions")

    # Tasks
    print_section("Open Tasks")
    if data["tasks"]:
        rows = []
        for t in data["tasks"]:
            progress = f"{t.get('progress', 0)}%" if t.get("status") == "in_progress" else ""
            rows.append(
                (
                    str(t.get("id", "")),
                    t.get("title", "")[:50],
                    t.get("status", ""),
                    progress,
                    t.get("created_at", "")[:19],
                )
            )
        print_table(
            None,
            ["ID", "Title", "Status", "Progress", "Created"],
            rows,
            styles=["dim", "bold", "yellow", "cyan", "dim"],
        )
    else:
        print("  No open tasks")

    # Recent Insights
    print_section("Recent Insights")
    if data["insights"]:
        rows = []
        for i in data["insights"]:
            rows.append(
                (
                    i.get("type", ""),
                    i.get("preview", "")[:80],
                    i.get("created_at", "")[:19],
                    i.get("id", ""),
                )
            )
        print_table(None, ["Type", "Preview", "Created", "ID"], rows, styles=["bold", None, "dim", "dim"])
    else:
        print("  No recent insights")

    # Claims
    print_section("Active Claims")
    if data["claims"]:
        rows = []
        for c in data["claims"]:
            rows.append(
                (
                    c.get("type", ""),
                    c.get("path", ""),
                    c.get("session", ""),
                )
            )
        print_table(None, ["Type", "Path/Branch", "Session"], rows, styles=["bold", None, "dim"])
    else:
        print("  No active claims")


def _send_notification(text: str, context: str = None):
    """Send a macOS notification via osascript. Best-effort."""
    try:
        text_escaped = text.replace('"', '\\"')
        subtitle = ""
        if context:
            ctx_escaped = context[:80].replace('"', '\\"')
            subtitle = f' subtitle "{ctx_escaped}"'
        script = f'display notification "{text_escaped}" with title "Cairn Reminder"{subtitle} sound name "Glass"'
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=5,
        )
    except Exception as e:
        logger.debug("macOS notification failed: %s", e)


def cmd_remind(args):
    """Manage reminders: set, list, check, dismiss."""
    sub = getattr(args, "remind_command", None)

    if sub == "set":
        text = " ".join(args.text)
        duration = args.duration
        context = getattr(args, "context", None)
        if not text.strip():
            print("Usage: cairn remind set <text> -d <duration>", file=sys.stderr)
            sys.exit(1)

        from cairn.bridge import create_reminder

        try:
            result = create_reminder(text=text, duration=duration, context=context)
            print(f"Reminder set: {result['text']}")
            print(f"  Due at: {result['remind_at_local']}")
            print(f"  ID: {result['reminder_id']}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif sub == "list":
        from cairn.bridge import list_reminders

        status = getattr(args, "status", None)
        include_dismissed = status in ("dismissed", "all")
        reminders = list_reminders(status=status, include_dismissed=include_dismissed)

        if not reminders:
            print("No reminders found.")
            return

        print(f"Reminders ({len(reminders)} found):\n")
        for r in reminders:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"  [{r['status']}]{overdue} {r['text']}")
            print(f"    Due: {r['remind_at_local']} | Time: {r['time_until']}")
            if r.get("context"):
                print(f"    Context: {r['context'][:120]}")
            print(f"    ID: {r['id']}")

    elif sub == "check":
        from cairn.bridge import get_due_reminders

        notify = getattr(args, "notify", False)
        due = get_due_reminders(mark_fired=True)

        if not due:
            print("No reminders due.")
            return

        for r in due:
            overdue = " [OVERDUE]" if r.get("is_overdue") else ""
            print(f"[REMINDER]{overdue} {r['text']}")
            if r.get("context"):
                print(f"  Context: {r['context'][:120]}")
            print(f"  ID: {r['id']}")

            if notify:
                _send_notification(r["text"], r.get("context"))

    elif sub == "dismiss":
        reminder_id = args.reminder_id
        from cairn.bridge import dismiss_reminder

        result = dismiss_reminder(reminder_id)
        if result.get("success"):
            print(f"Dismissed: {result.get('text', reminder_id)}")
        else:
            print(f"Error: {result.get('error')}", file=sys.stderr)
            sys.exit(1)

    else:
        print("Usage: cairn remind {set,list,check,dismiss}", file=sys.stderr)
        sys.exit(1)


def cmd_logs(args):
    """Show recent entries from ~/.cairn/hooks.log."""
    hooks_log = CAIRN_DIR / "hooks.log"
    if not hooks_log.exists():
        print("No hooks.log found — no hook errors recorded.")
        return

    n = getattr(args, "lines", 50)
    lines = hooks_log.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    print(f"--- Last {len(recent)} lines from {hooks_log} ---\n")
    for line in recent:
        print(line)


def cmd_validate(args):
    """Validate cairn.db integrity: SQLite PRAGMA + FTS5 checks."""
    from cairn.cli_ui import print_header, print_section, print_status_line, print_summary, print_table

    db_path = CAIRN_DIR / "cairn.db"
    if not db_path.exists():
        print("No cairn.db found.")
        return

    import sqlite3

    conn = sqlite3.connect(str(db_path), timeout=30)
    errors = 0

    print_header("Cairn Validate")

    # SQLite integrity check
    print_section("SQLite Integrity")
    result = conn.execute("PRAGMA integrity_check").fetchone()[0]
    if result == "ok":
        print_status_line("ok", "PRAGMA integrity_check passed")
    else:
        errors += 1
        print_status_line("fail", result)

    # FTS5 integrity
    print_section("FTS5 Index")
    try:
        conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
        print_status_line("ok", "FTS5 integrity check passed")
    except Exception as e:
        errors += 1
        print_status_line("fail", f"FTS5 integrity: {e}")
        if getattr(args, "repair", False):
            print("  Attempting rebuild...")
            try:
                conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
                conn.commit()
                print_status_line("ok", "FTS5 index rebuilt")
                errors -= 1
            except Exception as rebuild_err:
                print_status_line("fail", f"Rebuild failed: {rebuild_err}")

    # Row counts (allowlist — these names are used in f-string SQL)
    print_section("Table Counts")
    _VALID_TABLES = frozenset(
        [
            "memories",
            "edges",
            "entity_index",
        ]
    )
    table_rows = []
    for tbl in sorted(_VALID_TABLES):
        try:
            # SECURITY: tbl from _VALID_TABLES hardcoded frozenset, not user input
            count = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            table_rows.append((tbl, str(count)))
        except Exception as e:
            logger.debug("Table count failed for %s: %s", tbl, e)
    print_table(None, ["Table", "Count"], table_rows)

    conn.close()
    print()
    print_summary(errors, 0)
    sys.exit(1 if errors > 0 else 0)


_PLIST_LABEL = "com.cairn.mcp-daemon"
_PLIST_DEST = Path.home() / "Library" / "LaunchAgents" / f"{_PLIST_LABEL}.plist"
_DEFAULT_HTTP_PORT = 8377
_DEFAULT_HTTP_HOST = "127.0.0.1"

_JP_PLIST_LABEL = "com.cairn.jit-proxy-daemon"
_JP_PLIST_DEST = Path.home() / "Library" / "LaunchAgents" / f"{_JP_PLIST_LABEL}.plist"
_JP_HTTP_PORT = 8378
_JP_HTTP_HOST = "127.0.0.1"


def cmd_serve(args):
    """Run the Cairn MCP server. Supports stdio (default) and HTTP daemon mode."""
    import asyncio

    subcmd = getattr(args, "serve_command", None)

    if subcmd == "install":
        _serve_install(args)
        return
    elif subcmd == "uninstall":
        _serve_uninstall(args)
        return
    elif subcmd == "status":
        _serve_status(args)
        return
    elif subcmd == "migrate-config":
        _serve_migrate_config(args)
        return
    elif subcmd == "restore-config":
        _serve_restore_config(args)
        return

    # Default: run the MCP server
    if getattr(args, "no_condensed", False):
        os.environ["CAIRN_CONDENSED"] = "0"

    if getattr(args, "daemon", False):
        os.environ["CAIRN_TRANSPORT"] = "http"

    try:
        from cairn.server.mcp_server import main
    except SystemExit:
        return

    asyncio.run(main())


def _serve_install(args):
    """Generate launchd plist and load the daemon."""
    plist_template = (DATA_DIR / "com.cairn.mcp-daemon.plist").read_text()

    python_path = _resolve_python_path()
    cairn_home = str(CAIRN_DIR)

    # Resolve PYTHONPATH so cairn package is importable
    try:
        import cairn
        pythonpath = str(Path(cairn.__file__).parent.parent)
    except Exception:
        pythonpath = ""

    # Ensure log directory exists
    log_dir = CAIRN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = (
        plist_template
        .replace("__PYTHON_PATH__", python_path)
        .replace("__CAIRN_HOME__", cairn_home)
        .replace("__PYTHONPATH__", pythonpath)
    )

    _PLIST_DEST.parent.mkdir(parents=True, exist_ok=True)
    _PLIST_DEST.write_text(plist_content)
    print(f"Plist written to {_PLIST_DEST}")

    # Load the daemon
    result = subprocess.run(
        ["launchctl", "load", str(_PLIST_DEST)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Daemon loaded. It will start automatically on login.")
        print(f"\nVerify: curl http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/health")
        print("\nTo use with Claude Code, run: cairn serve migrate-config")
    else:
        print(f"launchctl load failed: {result.stderr.strip()}")
        sys.exit(1)


def _serve_uninstall(args):
    """Unload and remove the daemon plist."""
    if _PLIST_DEST.exists():
        subprocess.run(
            ["launchctl", "unload", str(_PLIST_DEST)],
            capture_output=True, text=True,
        )
        _PLIST_DEST.unlink()
        print("Daemon unloaded and plist removed.")
        print("\nTo restore stdio config, run: cairn serve restore-config")
    else:
        print("No daemon plist found. Nothing to uninstall.")


def _serve_status(args):
    """Check daemon status via launchd and health endpoint."""
    import urllib.request
    import urllib.error

    # Check launchd
    result = subprocess.run(
        ["launchctl", "list", _PLIST_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Daemon: not loaded (launchd)")
    else:
        lines = result.stdout.strip().split("\n")
        print("Daemon: loaded (launchd)")
        for line in lines:
            if "PID" in line or '"PID"' in line:
                print(f"  {line.strip()}")

    # Check health endpoint
    url = f"http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            print(f"\nHealth: {data.get('status', 'unknown')}")
            print(f"  PID: {data.get('pid')}")
            print(f"  RSS: {data.get('rss_mb')} MB")
            print(f"  Uptime: {data.get('uptime_s')}s")
            print(f"  Tools: {data.get('tool_count')}")
    except (urllib.error.URLError, OSError):
        print(f"\nHealth: unreachable ({url})")


def _serve_migrate_config(args):
    """Migrate ~/.claude.json cairn entries from stdio to http."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        print("No ~/.claude.json found.")
        return

    content = claude_json.read_text()
    config = json.loads(content)

    # Create backup
    backup = claude_json.with_suffix(".json.bak")
    backup.write_text(content)
    print(f"Backup saved to {backup}")

    # Trailing slash on purpose: the Starlette mount 307-redirects /mcp
    # to /mcp/, and not every MCP client re-POSTs through a redirect.
    url = f"http://{_DEFAULT_HTTP_HOST}:{_DEFAULT_HTTP_PORT}/mcp/"
    changed = 0

    def _migrate_entry(servers):
        nonlocal changed
        entry = servers.get("cairn")
        if not isinstance(entry, dict):
            return
        # No "type" key means stdio: it is the implied default for
        # command-based entries.
        if entry.get("type", "stdio") == "stdio":
            servers["cairn"] = {"type": "http", "url": url}
            changed += 1

    # User scope (`claude mcp add -s user`) lives at the top level, not
    # under projects — the fleet's registration is usually here.
    _migrate_entry(config.get("mcpServers", {}))
    for proj_config in config.get("projects", {}).values():
        _migrate_entry(proj_config.get("mcpServers", {}))

    if changed > 0:
        claude_json.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Migrated {changed} project(s) from stdio to http.")
        print(f"MCP endpoint: {url}")
        print("\nRestart Claude Code terminals to use the daemon.")
    else:
        print("No stdio cairn entries found to migrate.")


def _serve_restore_config(args):
    """Restore ~/.claude.json from backup."""
    claude_json = Path.home() / ".claude.json"
    backup = claude_json.with_suffix(".json.bak")

    if not backup.exists():
        print("No backup found at ~/.claude.json.bak")
        return

    backup_content = backup.read_text()
    claude_json.write_text(backup_content)
    print("Restored ~/.claude.json from backup.")
    print("Restart Claude Code terminals to use stdio mode.")


def cmd_proxy(args):
    """Manage jit-proxy daemon."""
    subcmd = getattr(args, "proxy_command", None)

    if subcmd == "install":
        _jp_install(args)
    elif subcmd == "uninstall":
        _jp_uninstall(args)
    elif subcmd == "status":
        _jp_status(args)
    elif subcmd == "migrate-config":
        _jp_migrate_config(args)
    elif subcmd == "restore-config":
        _jp_restore_config(args)
    else:
        print("Usage: cairn proxy {install|uninstall|status|migrate-config|restore-config}")


def _jp_install(args):
    """Install jit-proxy launchd daemon."""
    plist_template = (DATA_DIR / "com.cairn.jit-proxy-daemon.plist").read_text()

    python_path = _resolve_python_path()
    cairn_home = str(CAIRN_DIR)

    try:
        import cairn
        pythonpath = str(Path(cairn.__file__).parent.parent)
    except Exception:
        pythonpath = ""

    # Capture current PATH so backends (npx, uvx, x-twitter-mcp-server) are findable
    current_path = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")

    log_dir = CAIRN_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    plist_content = (
        plist_template
        .replace("__PYTHON_PATH__", python_path)
        .replace("__CAIRN_HOME__", cairn_home)
        .replace("__PYTHONPATH__", pythonpath)
        .replace("__PATH__", current_path)
    )

    _JP_PLIST_DEST.parent.mkdir(parents=True, exist_ok=True)
    _JP_PLIST_DEST.write_text(plist_content)
    print(f"Plist written to {_JP_PLIST_DEST}")

    result = subprocess.run(
        ["launchctl", "load", str(_JP_PLIST_DEST)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("jit-proxy daemon loaded. It will start automatically on login.")
        print(f"\nVerify: curl http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/health")
        print("\nTo use with Claude Code, run: cairn proxy migrate-config")
    else:
        print(f"launchctl load failed: {result.stderr.strip()}")
        sys.exit(1)


def _jp_uninstall(args):
    """Unload and remove jit-proxy daemon."""
    if _JP_PLIST_DEST.exists():
        subprocess.run(
            ["launchctl", "unload", str(_JP_PLIST_DEST)],
            capture_output=True, text=True,
        )
        _JP_PLIST_DEST.unlink()
        print("jit-proxy daemon unloaded and plist removed.")
        print("\nTo restore stdio config, run: cairn proxy restore-config")
    else:
        print("No jit-proxy daemon plist found. Nothing to uninstall.")


def _jp_status(args):
    """Check jit-proxy daemon status."""
    import urllib.request
    import urllib.error

    result = subprocess.run(
        ["launchctl", "list", _JP_PLIST_LABEL],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Daemon: not loaded")
    else:
        print("Daemon: loaded")
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                print(f"  {line.strip()}")

    url = f"http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/health"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
            print("\nHealth: OK")
            print(f"  PID: {data.get('pid')}")
            print(f"  RSS: {data.get('rss_mb')} MB")
            print(f"  Uptime: {data.get('uptime_s')}s")
            print(f"  Tools: {data.get('tool_count')}")
            backends = data.get("backends", {})
            for name, status in backends.items():
                connected = "connected" if status.get("connected") else "idle"
                print(f"  Backend {name}: {connected}")
    except Exception:
        print(f"\nHealth: unreachable ({url})")


def _jp_migrate_config(args):
    """Migrate ~/.claude.json jit-proxy entry from stdio to http."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        print("No ~/.claude.json found.")
        return

    content = claude_json.read_text()
    config = json.loads(content)

    # Backup
    backup = claude_json.with_suffix(".json.bak")
    backup.write_text(content)
    print(f"Backup saved to {backup}")

    url = f"http://{_JP_HTTP_HOST}:{_JP_HTTP_PORT}/mcp"
    changed = 0

    # Global mcpServers (top-level)
    servers = config.get("mcpServers", {})
    if "jit-proxy" in servers:
        entry = servers["jit-proxy"]
        if entry.get("type") == "stdio":
            servers["jit-proxy"] = {
                "type": "http",
                "url": url,
            }
            changed += 1

    # Also check per-project entries (in case user moved it)
    projects = config.get("projects", {})
    for proj_path, proj_config in projects.items():
        proj_servers = proj_config.get("mcpServers", {})
        if "jit-proxy" in proj_servers:
            entry = proj_servers["jit-proxy"]
            if entry.get("type") == "stdio":
                proj_servers["jit-proxy"] = {
                    "type": "http",
                    "url": url,
                }
                changed += 1

    if changed > 0:
        claude_json.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Migrated {changed} jit-proxy entry/entries from stdio to http.")
        print(f"MCP endpoint: {url}")
        print("\nRestart Claude Code terminals to use the daemon.")
    else:
        print("No stdio jit-proxy entries found to migrate.")


def _jp_restore_config(args):
    """Restore ~/.claude.json from backup."""
    claude_json = Path.home() / ".claude.json"
    backup = claude_json.with_suffix(".json.bak")

    if not backup.exists():
        print("No backup found at ~/.claude.json.bak")
        return

    backup_content = backup.read_text()
    claude_json.write_text(backup_content)
    print("Restored ~/.claude.json from backup.")
    print("Restart Claude Code terminals to apply.")


class _DoctorReport:
    """Accumulates doctor check results and prints them as they land.

    Carries the ok/fail/warn/section behaviour that ``cmd_doctor`` used to
    express as nested closures, so each diagnostic can live in its own
    ``_doctor_check_*`` function while sharing one result tally.
    """

    def __init__(self, use_json: bool):
        from cairn.cli_ui import print_section, print_status_line
        self.use_json = use_json
        self.checks = []
        self.errors = 0
        self.warnings = 0
        self._print_section = print_section
        self._print_status_line = print_status_line

    def ok(self, msg):
        self.checks.append({"status": "ok", "message": msg})
        if not self.use_json:
            self._print_status_line("ok", msg)

    def fail(self, msg):
        self.errors += 1
        self.checks.append({"status": "fail", "message": msg})
        if not self.use_json:
            self._print_status_line("fail", msg)

    def warn(self, msg):
        self.warnings += 1
        self.checks.append({"status": "warn", "message": msg})
        if not self.use_json:
            self._print_status_line("warn", msg)

    def section(self, title):
        if not self.use_json:
            self._print_section(title)


def _doctor_check_imports(report) -> bool:
    """Package + entry-point imports. Returns False if cairn itself is fatal."""
    report.section("Package Import")
    try:
        import cairn

        report.ok(f"cairn {cairn.__version__} imported")
    except Exception as e:
        report.fail(f"Cannot import cairn: {e}")
        return False

    try:
        from cairn.bridge import status as _s, auto_capture as _ac, query as _q  # noqa: F811,F401

        report.ok("cairn.bridge imported (status, auto_capture, query)")
    except Exception as e:
        report.fail(f"Cannot import cairn.bridge: {e}")

    try:
        from cairn.server.handlers import HANDLERS

        report.ok(f"cairn.server.handlers: {len(HANDLERS)} handlers registered")
    except Exception as e:
        report.fail(f"Cannot import handlers: {e}")

    try:
        from cairn.server.tool_schemas import TOOL_SCHEMAS

        report.ok(f"cairn.server.tool_schemas: {len(TOOL_SCHEMAS)} tools defined")
    except Exception as e:
        report.fail(f"Cannot import tool_schemas: {e}")
    return True


def _doctor_check_model(report) -> None:
    """Embedding model identity/files + a live generation probe at the expected dim."""
    from cairn.sqlite_store import EMBEDDING_DIM

    report.section("Embedding Model")
    # Resolve identity through the same path the runtime uses (honors
    # CAIRN_ONNX_MODEL_DIR overrides and cairn.json sidecars) rather than
    # hard-coding the two legacy directories.
    try:
        from cairn.embedding import _get_onnx_model_dir, get_embedding_model_info

        resolved_dir = _get_onnx_model_dir()
        minfo = get_embedding_model_info()
    except Exception as e:
        report.fail(f"Model resolution failed: {e}")
        resolved_dir = None
        minfo = {}

    if resolved_dir:
        model_path = Path(resolved_dir) / "model.onnx"
        model_mb = model_path.stat().st_size / (1024 * 1024) if model_path.exists() else 0
        report.ok(
            f"Resolved model: {minfo.get('model_name', '?')} "
            f"({model_mb:.0f} MB at {resolved_dir})"
        )
        sc = minfo.get("sidecar") or {}
        if sc.get("source") == "sidecar":
            report.ok(
                f"Sidecar config: dim={sc.get('dim', '?')} pooling={sc.get('pooling')} "
                f"prefixes={'yes' if sc.get('query_prefix') or sc.get('doc_prefix') else 'no'}"
            )
        if minfo.get("model_name") == "all-MiniLM-L6-v2":
            report.warn("Using legacy fallback model. Run 'cairn setup --download-model' to upgrade")
        tokenizer_path = Path(resolved_dir) / "tokenizer.json"
        if tokenizer_path.exists():
            report.ok("tokenizer.json present")
        else:
            report.fail(f"tokenizer.json not found at {resolved_dir}")
    else:
        report.fail(
            f"model.onnx not found at {GTE_MODEL_DIR}, {BGE_MODEL_DIR}, or {MINILM_MODEL_DIR}"
        )

    try:
        from cairn.embedding import generate_embedding, get_embedding_info

        info = get_embedding_info()
        if info.get("onnx_available"):
            report.ok("ONNX Runtime available")
        else:
            report.warn("ONNX Runtime not available, will use fallback")

        emb = generate_embedding("test embedding")
        from cairn.embedding import is_embedding_degraded, get_active_backend
        backend = get_active_backend() or "hash-fallback"
        if len(emb) != EMBEDDING_DIM:
            report.fail(f"Embedding dimension wrong: {len(emb)} (expected {EMBEDDING_DIM})")
        elif is_embedding_degraded() or backend in ("hash", "hash-fallback"):
            # Do NOT report this as OK — a hash-fallback backend means semantic
            # search is silently broken: stored ONNX vectors won't match hashed
            # query vectors. This is the failure mode that went unnoticed for
            # months. Fix: install the ONNX model, then re-embed the store.
            report.fail(f"Embeddings DEGRADED to {backend} — semantic search is broken. "
                        f"The ONNX model failed to load; queries and stored vectors will "
                        f"not match. Install the model and re-embed the store.")
        else:
            report.ok(f"Embedding generation works ({EMBEDDING_DIM}-dim, backend={backend})")
    except Exception as e:
        report.fail(f"Embedding generation failed: {e}")

    # Reranker identity — the model that actually resolves, not the one we
    # assume. (bge-reranker-v2-m3 was documented as the reranker for months
    # while the resolver silently fell back to ms-marco-MiniLM-L-6-v2.)
    report.section("Reranker Model")
    try:
        import cairn.reranker as _rr

        rr_name, rr_default_dir = _rr._resolve_reranker_model()
        rr_dir = _rr._get_model_dir()  # None when files absent (pre-download)
        rr_onnx = Path(rr_dir) / "model.onnx" if rr_dir else None
        if rr_onnx and rr_onnx.exists():
            rr_mb = rr_onnx.stat().st_size / (1024 * 1024)
            report.ok(f"Resolved reranker: {rr_name} ({rr_mb:.0f} MB at {rr_dir})")
        else:
            report.warn(
                f"Resolved reranker: {rr_name} — model files not on disk at "
                f"{rr_default_dir} (downloads on first use unless "
                f"CAIRN_RERANKER_AUTODOWNLOAD=0)"
            )
        # Drift: running a non-intended reranker without an explicit override.
        if rr_name != _rr.INTENDED_RERANKER_MODEL and not os.environ.get("CAIRN_RERANKER_MODEL"):
            report.warn(
                f"Reranker '{rr_name}' is not the intended "
                f"'{_rr.INTENDED_RERANKER_MODEL}' (and no CAIRN_RERANKER_MODEL override)"
            )
        if os.environ.get("CAIRN_CROSS_ENCODER", "1") == "0":
            report.warn("Cross-encoder disabled via CAIRN_CROSS_ENCODER=0")
    except Exception as e:
        report.warn(f"Reranker identity check failed: {e}")

    # LLM provider — used by the optional rollup / distillation / query-expansion
    # features. Bring a key from any major provider (see cairn.llm). Optional:
    # with no key these features no-op and core memory is unaffected.
    report.section("LLM Provider (optional)")
    try:
        import cairn.llm as _llm

        provider = os.environ.get("CAIRN_LLM_PROVIDER", "anthropic")
        if provider not in _llm.list_providers():
            report.warn(
                f"CAIRN_LLM_PROVIDER='{provider}' is unknown. Known: "
                f"{', '.join(_llm.list_providers())}"
            )
        else:
            key = _llm._get_api_key(provider)
            has_key = bool(key) and key not in ("none", "ollama")
            keyless_ok = provider == "ollama"
            fast = os.environ.get("CAIRN_LLM_MODEL_FAST") or _llm.get_model_map()[provider]["fast"]
            if has_key or keyless_ok:
                report.ok(f"LLM provider: {provider} (fast model: {fast})")
            else:
                report.warn(
                    f"LLM provider '{provider}' has no API key configured — "
                    f"rollup/distillation/query-expansion will no-op. Set the "
                    f"provider's key env or CAIRN_LLM_API_KEY."
                )
    except Exception as e:
        report.warn(f"LLM provider check failed: {e}")


def _doctor_check_database(report):
    """Open a read-only probe connection; returns it (or None) for later checks."""
    # Use a single lightweight read-only connection with short busy_timeout
    # to avoid blocking when the MCP server holds a WAL write lock.
    report.section("Database")
    db_path = CAIRN_DIR / "cairn.db"
    _doctor_conn = None
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        report.ok(f"cairn.db exists ({size_mb:.2f} MB)")
        try:
            import sqlite3 as _sqlite3
            _doctor_conn = _sqlite3.connect(str(db_path), timeout=5)
            _doctor_conn.execute("PRAGMA busy_timeout=5000")
            _doctor_conn.execute("PRAGMA query_only=ON")
            try:
                import sqlite_vec
                _doctor_conn.enable_load_extension(True)
                sqlite_vec.load(_doctor_conn)
                _doctor_conn.enable_load_extension(False)
                vec_enabled = True
            except Exception:
                vec_enabled = False
            mem_count = _doctor_conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            report.ok(f"Database accessible: {mem_count} memories, {size_mb:.2f} MB")
            if vec_enabled:
                report.ok("sqlite-vec enabled (vector search)")
            else:
                report.warn("sqlite-vec not available (text-only search)")
        except Exception as e:
            report.fail(f"Database check failed: {e}")
    else:
        report.warn("cairn.db not found (will be created on first use)")
    return _doctor_conn


def _doctor_check_mcp(report, client) -> None:
    """MCP server registration (Claude Code CLI, or generic availability)."""
    check_claude = client == "claude-code" or shutil.which("claude")
    if check_claude:
        report.section("MCP Server (Claude Code)")
        try:
            result = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True, timeout=5)
            if "cairn" in result.stdout:
                report.ok("cairn registered in Claude Code")
            else:
                report.fail("cairn NOT registered in Claude Code")
                if not report.use_json:
                    print("    Run: claude mcp add -s user cairn -- python3 -m cairn.server.mcp_server")
        except FileNotFoundError:
            report.warn("Claude Code CLI not found (cannot verify MCP registration)")
        except Exception as e:
            report.warn(f"MCP check failed: {e}")
    else:
        report.section("MCP Server")
        python_path = _resolve_python_path()
        report.ok(f"MCP server available: {python_path} -m cairn.server.mcp_server")


def _doctor_check_claude_desktop(report) -> None:
    """Claude Desktop config registration."""
    report.section("Claude Desktop")
    if sys.platform == "darwin":
        desktop_config = Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    else:
        appdata = os.environ.get("APPDATA", "")
        desktop_config = Path(appdata) / "Claude" / "claude_desktop_config.json" if appdata else None

    if desktop_config and desktop_config.exists():
        try:
            dc = json.loads(desktop_config.read_text(encoding="utf-8"))
            servers = dc.get("mcpServers", {})
            if "cairn" in servers:
                entry = servers["cairn"]
                cmd = entry.get("command", "")
                if cmd and Path(cmd).exists():
                    report.ok(f"Claude Desktop: cairn configured (python: {cmd})")
                elif cmd:
                    report.warn(f"Claude Desktop: cairn configured but python not found: {cmd}")
                else:
                    report.warn("Claude Desktop: cairn entry has no command")
            else:
                report.warn("Claude Desktop: cairn not registered")
                if not report.use_json:
                    print("    Run: cairn setup --client claude-desktop")
        except (json.JSONDecodeError, OSError) as e:
            report.warn(f"Claude Desktop: cannot read config: {e}")
    elif desktop_config:
        report.ok("Claude Desktop: config not found (not installed or not configured)")
    else:
        report.ok("Claude Desktop: skipped (APPDATA not set)")


def _doctor_check_fts5(report, conn) -> None:
    """FTS5 index population, drift, and integrity."""
    report.section("FTS5 Index")
    if not conn:
        return
    db_path = CAIRN_DIR / "cairn.db"
    try:
        fts_count = conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0]
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if fts_count > 0:
            report.ok(f"FTS5 index populated ({fts_count} entries, {mem_count} memories)")
            if abs(fts_count - mem_count) > mem_count * 0.1:
                report.warn(f"FTS5 index drift: {fts_count} vs {mem_count} memories (>10% mismatch)")
        else:
            report.warn("FTS5 index empty (text search will use slower LIKE fallback)")
        # Integrity check (requires write access; use separate connection)
        try:
            import sqlite3 as _sqlite3
            _fts_conn = _sqlite3.connect(str(db_path), timeout=5)
            _fts_conn.execute("PRAGMA busy_timeout=5000")
            _fts_conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('integrity-check')")
            report.ok("FTS5 integrity check passed")
            _fts_conn.close()
        except Exception as fts_err:
            if "readonly" in str(fts_err) or "locked" in str(fts_err):
                report.ok("FTS5 index readable (integrity check skipped, DB busy)")
            else:
                report.fail(f"FTS5 integrity check failed: {fts_err}")
                if not report.use_json:
                    print("    Fix: INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    except Exception as e:
        report.warn(f"FTS5 check skipped: {e}")


def _doctor_check_vec(report, conn) -> None:
    """Vector index count + orphan detection."""
    report.section("Vector Index")
    if not conn:
        return
    try:
        vec_count = conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        report.ok(f"Vec index: {vec_count} embeddings, {mem_count} memories")
        if vec_count > mem_count:
            orphans = vec_count - mem_count
            report.warn(f"Vec index has ~{orphans} potential orphaned embeddings (run 'cairn consolidate' to clean)")
    except Exception as e:
        report.warn(f"Vec table not available: {e}")


def _doctor_check_memory_quality(report, conn) -> None:
    """Feedback-score coverage + review-flagged memories."""
    report.section("Memory Quality")
    if not conn:
        return
    try:
        rows = conn.execute("SELECT metadata FROM memories WHERE metadata LIKE '%feedback_score%'").fetchall()
        if rows:
            scores = []
            flagged = 0
            for (meta_str,) in rows:
                try:
                    meta = json.loads(meta_str)
                    scores.append(meta.get("feedback_score", 0))
                    if meta.get("flagged_for_review"):
                        flagged += 1
                except Exception as e:
                    logger.debug("Feedback metadata parse failed: %s", e)
            if scores:
                avg = sum(scores) / len(scores)
                report.ok(f"{len(scores)} memories with feedback (avg score: {avg:.2f})")
                if flagged > 0:
                    report.warn(f"{flagged} memory(ies) flagged for review (score <= -3)")
        else:
            report.ok("No feedback signals recorded yet")
    except Exception as e:
        report.warn(f"Quality check skipped: {e}")


def _doctor_check_hook_log(report) -> None:
    """Recent hook errors from hooks.log."""
    report.section("Hook Health")
    hooks_log = CAIRN_DIR / "hooks.log"
    if hooks_log.exists():
        try:
            lines = hooks_log.read_text().strip().split("\n")
            error_lines = [line for line in lines if line.startswith("[") and ": OK " not in line]
            if error_lines:
                recent = error_lines[-5:]
                report.warn(f"{len(error_lines)} hook error(s) in log, last {len(recent)}:")
                if not report.use_json:
                    for line in recent:
                        print(f"    {line[:120]}")
            else:
                report.ok("No hook errors in log")
        except Exception as e:
            report.warn(f"Cannot read hooks.log: {e}")
    else:
        report.ok("No hooks.log (no errors recorded)")


def _doctor_check_hooks_config(report, client) -> None:
    """SessionStart/Stop/PostToolUse hook wiring in settings.json."""
    check_hooks = client == "claude-code" or SETTINGS_JSON_PATH.exists()
    if not check_hooks:
        return
    report.section("Hooks (Claude Code)")
    if SETTINGS_JSON_PATH.exists():
        try:
            settings = json.loads(SETTINGS_JSON_PATH.read_text())
            hooks = settings.get("hooks", {})
            expected_events = ["SessionStart", "Stop", "PostToolUse"]
            for event in expected_events:
                found = False
                for entry in hooks.get(event, []):
                    for h in entry.get("hooks", []):
                        if "cairn" in h.get("command", ""):
                            found = True
                            cmd_parts = h["command"].split()
                            if cmd_parts and not Path(cmd_parts[0]).exists():
                                report.warn(f"{event} hook references {cmd_parts[0]} which doesn't exist")
                            break
                if found:
                    report.ok(f"{event} hook configured")
                else:
                    report.warn(f"{event} hook not configured")
        except Exception as e:
            report.warn(f"Cannot read settings.json: {e}")
    else:
        report.warn("settings.json not found (hooks not configured)")


def _doctor_check_surfacing(report) -> None:
    """Read-path health: surface hook wired but daemon socket gone = silent death."""
    # Catch the silent-death mode where the PostToolUse surface hook is wired
    # but the daemon socket is gone, so nothing ever surfaces and nothing errors.
    report.section("Read Path (Surfacing)")
    surface_wired = False
    if SETTINGS_JSON_PATH.exists():
        try:
            _settings = json.loads(SETTINGS_JSON_PATH.read_text())
            for _entry in _settings.get("hooks", {}).get("PostToolUse", []):
                for _h in _entry.get("hooks", []):
                    if "surface_memories" in _h.get("command", ""):
                        surface_wired = True
        except Exception:
            pass
    _hook_sock = CAIRN_DIR / "hook.sock"
    if surface_wired:
        if _hook_sock.exists():
            report.ok("Surface hook wired and daemon hook.sock present")
        else:
            report.fail("Surface hook is WIRED but hook.sock is MISSING — memory "
                        "surfacing is silently dead (the daemon isn't serving the hook "
                        "socket). Restart the MCP daemon.")
    else:
        report.ok("Surface hook not wired (explicit-query mode)")


def _doctor_check_query_expansion(report) -> None:
    """Query expansion enabled but no LLM provider = silent no-op."""
    try:
        from cairn.query_expansion import is_expansion_enabled
        if is_expansion_enabled():
            _provider = os.environ.get("CAIRN_LLM_PROVIDER", "anthropic")
            _has_llm = bool(
                os.environ.get("CAIRN_LLM_BASE_URL")
                or os.environ.get("ANTHROPIC_API_KEY")
                or _provider not in ("", "anthropic")
            )
            if _has_llm:
                report.ok(f"Query expansion enabled (LLM provider: {_provider})")
            else:
                report.warn("Query expansion is enabled but no LLM provider is configured "
                            "(CAIRN_LLM_PROVIDER/ANTHROPIC_API_KEY unset) — expansion "
                            "silently no-ops. Configure a provider or set "
                            "CAIRN_QUERY_EXPANSION=0.")
        else:
            report.ok("Query expansion disabled (CAIRN_QUERY_EXPANSION=0)")
    except Exception as e:
        report.warn(f"Query expansion check failed: {e}")


def _doctor_check_retrieval_trend(report) -> None:
    """Retrieval-quality MRR trend from eval-history.csv (silent-degradation signal)."""
    try:
        _hist = CAIRN_DIR / "logs" / "eval-history.csv"
        if _hist.exists():
            _rows = [ln.split(",") for ln in _hist.read_text().strip().splitlines()[1:] if ln]
            _mrrs = [float(r[1]) for r in _rows if len(r) >= 2]
            if len(_mrrs) >= 5:
                _prev = sorted(_mrrs[:-1])
                _median = _prev[len(_prev) // 2]
                if _median > 0 and _mrrs[-1] < 0.9 * _median:
                    report.warn(f"Retrieval MRR dropped: latest {_mrrs[-1]:.3f} vs median "
                                f"{_median:.3f} — investigate embeddings/scoring regressions")
                else:
                    report.ok(f"Retrieval MRR trend healthy (latest {_mrrs[-1]:.3f}, "
                              f"median {_median:.3f}, n={len(_mrrs)})")
            else:
                report.ok(f"Retrieval eval history: {len(_mrrs)} run(s) (trend needs 5)")
        else:
            report.ok("Retrieval eval history: none yet (nightly job will create it)")
    except Exception as e:
        report.warn(f"Eval trend check failed: {e}")


def _doctor_check_maintenance(report) -> None:
    """Maintenance freshness — stale markers mean the nightly jobs stopped running."""
    _maint_intervals = {"last-consolidate": 3, "last-compact": 3,
                        "last-backup": 7, "last-doctor": 7}
    _stale_maint = []
    for _marker, _days in _maint_intervals.items():
        _mp = CAIRN_DIR / _marker
        if _mp.exists():
            try:
                _age_days = (time.time() - _mp.stat().st_mtime) / 86400
                if _age_days > _days * 2:
                    _stale_maint.append(f"{_marker} {_age_days:.0f}d old (expected <{_days}d)")
            except Exception:
                pass
    if _stale_maint:
        report.warn("Maintenance appears stale (not running?): " + "; ".join(_stale_maint))
    else:
        report.ok("Maintenance markers fresh (or none yet)")


def _doctor_check_environment(report) -> None:
    """Python path + home + platform."""
    report.section("Environment")
    python_path = _resolve_python_path()
    if Path(python_path).exists():
        report.ok(f"Python: {python_path}")
    else:
        report.fail(f"Python path does not exist: {python_path}")

    report.ok(f"Cairn home: {CAIRN_DIR}")
    report.ok(f"Platform: {sys.platform}")


def _doctor_check_claude_md(report) -> None:
    """CLAUDE.md Cairn block + pre-Cairn backup."""
    if CLAUDE_MD_PATH.exists():
        claude_content = CLAUDE_MD_PATH.read_text()
        if CAIRN_BEGIN in claude_content:
            report.ok("CLAUDE.md: Cairn block installed")
            backup = CLAUDE_MD_PATH.with_suffix(".md.pre-cairn")
            if backup.exists():
                report.ok(f"CLAUDE.md: pre-Cairn backup at {backup.name}")
        else:
            report.warn("CLAUDE.md exists but has no Cairn block (run 'cairn setup' to add)")
    else:
        report.warn("CLAUDE.md not found (run 'cairn setup' to create)")


def cmd_doctor(args):
    """Verify Cairn installation: import, model, database, MCP, hooks.

    Orchestrates the individual ``_doctor_check_*`` diagnostics, each of which
    records ok/fail/warn results on a shared :class:`_DoctorReport`.
    """
    from cairn.cli_ui import print_header, print_summary

    use_json = _use_json(args)
    report = _DoctorReport(use_json)

    if not use_json:
        print_header("Cairn Doctor")

    # Package import is fatal if cairn itself won't load — bail with a terse tally.
    if not _doctor_check_imports(report):
        if use_json:
            print(json.dumps({"checks": report.checks, "errors": report.errors, "warnings": report.warnings}, indent=2))
        else:
            print(f"\n{report.errors} error(s), {report.warnings} warning(s)")
        sys.exit(1)

    client = getattr(args, "client", None)

    _doctor_check_model(report)
    _doctor_conn = _doctor_check_database(report)
    _doctor_check_mcp(report, client)
    _doctor_check_claude_desktop(report)
    _doctor_check_fts5(report, _doctor_conn)
    _doctor_check_vec(report, _doctor_conn)
    _doctor_check_memory_quality(report, _doctor_conn)
    _doctor_check_hook_log(report)
    _doctor_check_hooks_config(report, client)
    _doctor_check_surfacing(report)
    _doctor_check_query_expansion(report)
    _doctor_check_retrieval_trend(report)
    _doctor_check_maintenance(report)
    _doctor_check_environment(report)
    _doctor_check_claude_md(report)

    # Cleanup
    if _doctor_conn:
        _doctor_conn.close()

    # Summary
    if use_json:
        print(json.dumps({"checks": report.checks, "errors": report.errors, "warnings": report.warnings}, indent=2))
    else:
        print()
        print_summary(report.errors, report.warnings)

    sys.exit(1 if report.errors > 0 else 0)


def cmd_export_obsidian(args):
    """Export memories as Obsidian-compatible markdown files."""
    from cairn.obsidian_export import export_to_obsidian

    output_dir = getattr(args, "output_dir", "./cairn-vault")
    project = getattr(args, "project", None)
    limit = getattr(args, "limit", 0)

    result = export_to_obsidian(
        output_dir=output_dir,
        project=project,
        limit=limit,
    )

    print(f"Exported {result['memories_exported']} memories "
          f"({result['edge_links_created']} edges) to {result['output_dir']}")
    print(f"Index file: {result['index_file']}")


def cmd_eval_context_packet(args):
    """Evaluate task-aware context packet quality."""
    from dataclasses import asdict
    from cairn.evaluation.context_packet_eval import (
        format_packet_report,
        run_context_packet_evaluation,
    )

    report = run_context_packet_evaluation(
        sample_size=args.sample_size,
        budget_tokens=args.budget_tokens,
        mode=args.mode,
        seed=args.seed,
        output_path=args.output,
        probe_cache_path=getattr(args, "probe_cache", None),
    )
    if _use_json(args):
        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(format_packet_report(report))
        if args.output:
            print(f"\nSaved JSON report to {args.output}")


def cmd_backfill_context_packet(args):
    """Backfill graph edges from context packet misses."""
    from cairn.bridge import _get_store
    from cairn.evaluation.context_packet_eval import backfill_packet_miss_report

    manifest = backfill_packet_miss_report(
        _get_store(),
        args.report,
        similarity_threshold=args.threshold,
        max_connections_per_source=args.max_connections_per_source,
        max_edges=args.max_edges,
        dry_run=not getattr(args, "apply", False),
        output_path=args.output,
        event_types=_parse_event_types_arg(getattr(args, "event_types", None)),
    )
    if _use_json(args):
        print(json.dumps(manifest, indent=2, default=str))
    else:
        action = "Would create" if manifest.get("dry_run") else "Created"
        print(
            f"{action} {manifest.get('created', 0)} packet-miss edge(s) "
            f"from {manifest.get('eligible_misses', 0)} eligible miss(es)."
        )
        if args.output:
            print(f"Saved manifest to {args.output}")


def cmd_diagnose_context_packet(args):
    """Explain context packet eval misses without mutating memory state."""
    from cairn.bridge import _get_store
    from cairn.evaluation.context_packet_eval import (
        diagnose_context_packet_report,
        format_packet_diagnosis,
    )

    result = diagnose_context_packet_report(
        _get_store(),
        args.report,
        limit=args.limit,
        include_hits=getattr(args, "include_hits", False),
        output_path=args.output,
    )
    if _use_json(args):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_packet_diagnosis(result))
        if args.output:
            print(f"\nSaved diagnosis JSON to {args.output}")


def cmd_maintain_context_packet(args):
    """Run eval plus capped packet-miss maintenance."""
    from cairn.bridge import _get_store
    from cairn.evaluation.context_packet_eval import run_context_packet_maintenance_loop

    result = run_context_packet_maintenance_loop(
        _get_store(),
        artifact_prefix=args.artifact_prefix,
        sample_size=args.sample_size,
        budget_tokens=args.budget_tokens,
        mode=args.mode,
        seed=args.seed,
        probe_cache_path=getattr(args, "probe_cache", None),
        similarity_threshold=args.threshold,
        max_connections_per_source=args.max_connections_per_source,
        max_edges=args.max_edges,
        event_types=_parse_event_types_arg(getattr(args, "event_types", None)),
        apply=getattr(args, "apply", False),
        re_eval=getattr(args, "re_eval", False),
    )
    if _use_json(args):
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"Before report: {result.get('before_report')}")
        print(f"Backfill manifest: {result.get('backfill_manifest')}")
        if result.get("after_report"):
            print(f"After report: {result.get('after_report')}")


def cmd_eval_retrieval(args):
    """Evaluate retrieval quality with probe queries."""
    from cairn.evaluation.retrieval_eval import format_report, run_evaluation

    sample_size = getattr(args, "sample_size", 20)
    top_k = getattr(args, "top_k", 5)
    judge = getattr(args, "judge", False)
    model = getattr(args, "model", "claude-haiku-4-5-20251001")
    seed = getattr(args, "seed", 42)
    output_path = getattr(args, "output", None)
    use_json = _use_json(args)

    # --- v2 paths: frozen judged probe sets (non-self-referential) ---------
    if getattr(args, "build_probes", False):
        from cairn.evaluation.retrieval_eval import snapshot_db
        from cairn.evaluation.probe_set import build_probe_set
        from cairn.sqlite_store import SQLiteStore

        live = CAIRN_DIR / "cairn.db"
        if not live.exists():
            print("No cairn.db found — nothing to build probes from.")
            sys.exit(1)
        snap = snapshot_db(str(live))
        print(f"Building probe set against snapshot {snap} (size={sample_size}, seed={seed})...")
        store = SQLiteStore(db_path=snap)
        try:
            payload = build_probe_set(store, size=sample_size, seed=seed, top_k=10,
                                      from_query_log=getattr(args, "from_query_log", False))
        finally:
            store.close()
        print(f"Probe set: {payload['path']}")
        print(f"  probes: {payload['probe_count']}  skipped: {len(payload['skipped'])}  sha: {payload['content_sha256'][:12]}")
        if payload["probe_count"] == 0:
            print("  WARNING: 0 probes built — is an LLM provider configured (CAIRN_LLM_PROVIDER/ANTHROPIC_API_KEY)?")
            sys.exit(1)
        return

    probes_path = getattr(args, "probes", None)
    if probes_path:
        from cairn.evaluation.retrieval_eval import (
            compare_ab, format_report_v2, run_evaluation_v2, write_history_row,
        )

        ab_spec = getattr(args, "ab", None)
        if ab_spec:
            # --ab "VAR=VAL_A|VAL_B": paired A/B over one env knob
            try:
                var, vals = ab_spec.split("=", 1)
                val_a, val_b = vals.split("|", 1)
            except ValueError:
                print("Error: --ab expects VAR=VAL_A|VAL_B (e.g. CAIRN_CROSS_ENCODER=1|0)")
                sys.exit(1)
            va = (f"{var}={val_a}", {var: val_a})
            vb = (f"{var}={val_b}", {var: val_b})
            print(f"Paired A/B on probe set {probes_path}: {va[0]} vs {vb[0]} (top-{top_k})...")
            cmp = compare_ab(probes_path, va, vb, top_k=top_k)
            if use_json:
                print(json.dumps(cmp, indent=2, default=str))
            else:
                d = cmp["delta"]
                st = cmp["sign_test"]
                la, lb = cmp["labels"]
                print(f"\n| metric | {la} | {lb} | delta (B-A) |")
                print("|--------|------|------|-------------|")
                for m in ("mrr", "ndcg_at_k", "precision_at_k", "hit_rate"):
                    print(f"| {m} | {cmp['a'][m]:.3f} | {cmp['b'][m]:.3f} | {d[m]:+.3f} |")
                print(f"\nSign test (per-probe RR): B wins {st['b_wins']}, "
                      f"A wins {st['a_wins']}, p={st['p_value']:.4f}")
            if output_path:
                Path(output_path).write_text(json.dumps(cmp, indent=2, default=str))
                print(f"\nA/B report saved to {output_path}")
            return

        report = run_evaluation_v2(probes_path, top_k=top_k)
        csv_path = write_history_row(report)
        if use_json:
            from dataclasses import asdict
            print(json.dumps(asdict(report), indent=2, default=str))
        else:
            print(format_report_v2(report))
            print(f"\nHistory row appended to {csv_path}")
        if output_path:
            from dataclasses import asdict
            Path(output_path).write_text(json.dumps(asdict(report), indent=2, default=str))
        return

    if judge:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            print("Error: --judge requires the 'anthropic' package. Install with: pip install anthropic")
            sys.exit(1)
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("Error: --judge requires ANTHROPIC_API_KEY environment variable")
            sys.exit(1)

    print(f"Running retrieval evaluation ({sample_size} probes, top-{top_k}, mode={'judge' if judge else 'basic'})...")

    report = run_evaluation(
        sample_size=sample_size,
        top_k=top_k,
        judge=judge,
        model=model,
        seed=seed,
        output_path=output_path,
    )

    if use_json:
        from dataclasses import asdict

        print(json.dumps(asdict(report), indent=2, default=str))
    else:
        print(format_report(report))

    if output_path:
        print(f"\nReport saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="cairn",
        description="Cairn — Persistent memory for AI coding agents",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Memory commands ---
    query_parser = subparsers.add_parser("query", help="Search memories by semantic similarity or exact phrase")
    query_parser.add_argument("query_text", nargs="+", help="Search text")
    query_parser.add_argument("--exact", action="store_true", help="Use FTS5 exact phrase search instead of semantic")
    query_parser.add_argument("--limit", type=int, default=10, help="Max results (default: 10)")
    query_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    store_parser = subparsers.add_parser("store", help="Store a memory with a specified type")
    store_parser.add_argument("content", nargs="+", help="Memory content")
    store_parser.add_argument(
        "-t",
        "--type",
        default="memory",
        choices=["memory", "lesson", "decision", "error", "task", "preference"],
        help="Memory type (default: memory)",
    )
    store_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    remember_parser = subparsers.add_parser("remember", help="Store a permanent user preference")
    remember_parser.add_argument("text", nargs="+", help="Preference text")
    remember_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    timeline_parser = subparsers.add_parser("timeline", help="Show memory timeline grouped by day")
    timeline_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    timeline_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    # --- Admin commands ---
    setup_parser = subparsers.add_parser("setup", help="Set up Cairn: download model, initialize DB")
    setup_parser.add_argument(
        "--download-model",
        action="store_true",
        help="Eagerly download the intended embedder + reranker now (verified). "
        "Without this, the embedder auto-downloads on first use.",
    )
    setup_parser.add_argument(
        "--client", choices=["claude-code", "claude-desktop", "cursor", "windsurf", "cline", "codex", "antigravity", "venv"], help="Configure a specific client (MCP registration, hooks)"
    )
    setup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing any files",
    )
    setup_parser.add_argument(
        "--hooks-only",
        action="store_true",
        help="Configure hooks and CLAUDE.md WITHOUT MCP server (saves ~600MB RAM per session)",
    )

    status_parser = subparsers.add_parser("status", help="Show memory count, store size, model status")
    status_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    doctor_parser = subparsers.add_parser("doctor", help="Verify installation: import, model, database")
    doctor_parser.add_argument("--client", choices=["claude-code", "claude-desktop", "cursor", "windsurf", "cline", "codex", "antigravity", "venv"], help="Include client-specific checks (MCP, hooks)")
    doctor_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")

    subparsers.add_parser("migrate-home", help="Migrate legacy ~/.omega data dir to ~/.cairn (non-destructive)")
    migrate_emb_parser = subparsers.add_parser(
        "migrate-embeddings",
        help="Rebuild vec table + re-embed the store for the current model/dimension",
    )
    migrate_emb_parser.add_argument(
        "--force", action="store_true",
        help="proceed even if live Cairn servers are registered",
    )
    migrate_db_parser = subparsers.add_parser("migrate-db", help="Migrate JSON graphs to SQLite backend")
    migrate_db_parser.add_argument("--force", action="store_true", help="Overwrite existing SQLite database")
    subparsers.add_parser("reingest", help="Load store.jsonl entries into graph system")
    consolidate_parser = subparsers.add_parser("consolidate", help="Deduplicate, prune, and optimize memory")
    consolidate_parser.add_argument(
        "--prune-days", type=int, default=30, help="Prune entries older than N days with 0 access (default: 30)"
    )
    rollup_parser = subparsers.add_parser(
        "rollup", help="Synthesize aging episodic memories into durable project_history (LLM)"
    )
    rollup_parser.add_argument("--min-age-days", type=int, default=30,
                               help="Only roll rows older than N days (default: 30)")
    rollup_parser.add_argument("--max-windows", type=int, default=6,
                               help="Max project-month windows per run (default: 6)")
    rollup_parser.add_argument("--dry-run", action="store_true",
                               help="Print the synthesis without writing anything")
    link_parser = subparsers.add_parser(
        "link", help="Discover and link related memories (vector-similarity edges)"
    )
    link_parser.add_argument("--hours", type=int, default=48,
                             help="Lookback window in hours (default: 48)")
    link_parser.add_argument("--dry-run", action="store_true")
    subparsers.add_parser(
        "gc", help="Storage GC: re-embed hash-tainted rows, drain dead queues, VACUUM"
    )
    subparsers.add_parser("backup", help="Back up cairn.db to ~/.cairn/backups/ (keeps last 5)")
    compact_parser = subparsers.add_parser("compact", help="Cluster and summarize related memories")
    compact_parser.add_argument(
        "-t",
        "--type",
        default="lesson_learned",
        choices=["lesson_learned", "decision", "error_pattern", "task_completion"],
        help="Event type to compact (default: lesson_learned)",
    )
    compact_parser.add_argument("--threshold", type=float, default=0.60, help="Similarity threshold (default: 0.60)")
    compact_parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be compacted without changing data"
    )
    stats_parser = subparsers.add_parser("stats", help="Show memory type distribution and health summary")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")
    stats_parser.add_argument("--card", action="store_true", help="Show a shareable stats card")
    activity_parser = subparsers.add_parser("activity", help="Show recent session activity overview")
    activity_parser.add_argument("--days", type=int, default=7, help="Number of days to show (default: 7)")
    activity_parser.add_argument("--json", action="store_true", help="Output as JSON (also: CAIRN_JSON=1)")
    logs_parser = subparsers.add_parser("logs", help="Show recent hook errors from hooks.log")
    logs_parser.add_argument("-n", "--lines", type=int, default=50, help="Number of lines to show (default: 50)")
    validate_parser = subparsers.add_parser("validate", help="Validate cairn.db integrity (SQLite + FTS5)")
    validate_parser.add_argument("--repair", action="store_true", help="Attempt to repair FTS5 index if corrupted")
    serve_parser = subparsers.add_parser("serve", help="Run MCP server (stdio or HTTP daemon)")
    serve_parser.add_argument("--daemon", action="store_true", help="Run as HTTP daemon (CAIRN_TRANSPORT=http)")
    serve_parser.add_argument("--no-condensed", action="store_true", help="Disable condensed mode (expose all tools individually instead of meta-tools)")
    serve_sub = serve_parser.add_subparsers(dest="serve_command", help="Daemon management")
    serve_sub.add_parser("install", help="Install launchd daemon and load it")
    serve_sub.add_parser("uninstall", help="Unload and remove launchd daemon")
    serve_sub.add_parser("status", help="Check daemon status and health")
    serve_sub.add_parser("migrate-config", help="Migrate ~/.claude.json from stdio to http")
    serve_sub.add_parser("restore-config", help="Restore ~/.claude.json from backup")

    # --- Proxy commands (jit-proxy daemon) ---
    proxy_parser = subparsers.add_parser("proxy", help="Manage jit-proxy daemon")
    proxy_sub = proxy_parser.add_subparsers(dest="proxy_command", help="Proxy daemon management")
    proxy_sub.add_parser("install", help="Install jit-proxy launchd daemon")
    proxy_sub.add_parser("uninstall", help="Unload and remove jit-proxy daemon")
    proxy_sub.add_parser("status", help="Check jit-proxy daemon status and health")
    proxy_sub.add_parser("migrate-config", help="Migrate ~/.claude.json jit-proxy from stdio to http")
    proxy_sub.add_parser("restore-config", help="Restore ~/.claude.json from backup")

    # --- Hooks commands ---
    hooks_parser = subparsers.add_parser("hooks", help="Manage Claude Code hooks")
    hooks_sub = hooks_parser.add_subparsers(dest="hooks_command", help="Hook subcommands")
    hooks_sub.add_parser("setup", help="Configure hooks in ~/.claude/settings.json")
    hooks_sub.add_parser("path", help="Print the hooks directory path")
    hooks_sub.add_parser("doctor", help="Check hook configuration health")

    # --- Reminder commands (experimental) ---
    remind_parser = subparsers.add_parser("remind", help="Manage time-based reminders (experimental)")
    remind_sub = remind_parser.add_subparsers(dest="remind_command", help="Reminder subcommands")

    remind_set_parser = remind_sub.add_parser("set", help="Set a new reminder")
    remind_set_parser.add_argument("text", nargs="+", help="Reminder text")
    remind_set_parser.add_argument("-d", "--duration", required=True, help="Duration: 1h, 30m, 2d, 1w, 1d12h")
    remind_set_parser.add_argument("--context", help="Optional context for the reminder")

    remind_list_parser = remind_sub.add_parser("list", help="List reminders")
    remind_list_parser.add_argument(
        "--status",
        choices=["pending", "fired", "dismissed", "all"],
        help="Filter by status (default: pending + fired)",
    )

    remind_check_parser = remind_sub.add_parser("check", help="Check for due reminders")
    remind_check_parser.add_argument("--notify", action="store_true", help="Send macOS notification for due reminders")

    remind_dismiss_parser = remind_sub.add_parser("dismiss", help="Dismiss a reminder")
    remind_dismiss_parser.add_argument("reminder_id", help="Reminder ID to dismiss")

    # --- Obsidian export ---
    obsidian_parser = subparsers.add_parser("export-obsidian", help="Export memories as Obsidian-compatible markdown files")
    obsidian_parser.add_argument("--output-dir", default="./cairn-vault", help="Root directory for exported vault (default: ./cairn-vault)")
    obsidian_parser.add_argument("--project", help="Only export memories for this project")
    obsidian_parser.add_argument("--limit", type=int, default=0, help="Max memories to export (default: all)")

    # --- Evaluation commands ---
    eval_parser = subparsers.add_parser("eval-retrieval", help="Evaluate retrieval quality with probe queries")
    eval_parser.add_argument("--sample-size", type=int, default=20, help="Number of memories to probe (default: 20)")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Results per probe (default: 5)")
    eval_parser.add_argument("--judge", action="store_true", help="Use LLM to generate queries and score relevance")
    eval_parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="LLM model for judge mode")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling (default: 42)")
    eval_parser.add_argument("--output", help="Save JSON report to this path")
    eval_parser.add_argument("--json", action="store_true", help="Output as JSON to stdout (also: CAIRN_JSON=1)")
    eval_parser.add_argument("--build-probes", action="store_true",
                             help="Build a frozen judged probe set (v2, needs LLM provider) and exit")
    eval_parser.add_argument("--probes", help="Run eval v2 against a frozen probe-set JSON (provider-free)")
    eval_parser.add_argument("--from-query-log", action="store_true",
                             help="With --build-probes: source topics from real logged queries (replay)")
    eval_parser.add_argument("--ab", metavar="VAR=VAL_A|VAL_B",
                             help="Paired A/B over one env knob on the probe set "
                                  "(e.g. CAIRN_CROSS_ENCODER=1|0) + sign test")

    packet_eval_parser = subparsers.add_parser("eval-context-packet", help="Evaluate task-aware context packet quality")
    packet_eval_parser.add_argument("--sample-size", type=int, default=20, help="Number of memories to probe (default: 20)")
    packet_eval_parser.add_argument("--budget-tokens", type=int, default=800, help="Packet budget in approximate tokens")
    packet_eval_parser.add_argument("--mode", default="before_edit", choices=["before_edit", "planning", "debug", "review", "command"])
    packet_eval_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    packet_eval_parser.add_argument("--output", help="Save JSON report to this path")
    packet_eval_parser.add_argument("--probe-cache", help="Load/save fixed packet probe cache")
    packet_eval_parser.add_argument("--json", action="store_true", help="Output as JSON")

    packet_backfill_parser = subparsers.add_parser("backfill-context-packet", help="Preview or apply context packet miss edge backfill")
    packet_backfill_parser.add_argument("--report", required=True, help="Context packet eval JSON report")
    packet_backfill_parser.add_argument("--threshold", type=float, default=0.72, help="Similarity threshold")
    packet_backfill_parser.add_argument("--max-connections-per-source", type=int, default=1)
    packet_backfill_parser.add_argument("--max-edges", type=int, default=10)
    packet_backfill_parser.add_argument("--event-types", help="Comma-separated source event types to consider")
    packet_backfill_parser.add_argument("--apply", action="store_true", help="Write edges. Requires Cairn Pro.")
    packet_backfill_parser.add_argument("--output", help="Save backfill manifest JSON")
    packet_backfill_parser.add_argument("--json", action="store_true", help="Output as JSON")

    packet_diag_parser = subparsers.add_parser("diagnose-context-packet", help="Explain context packet misses from an eval report")
    packet_diag_parser.add_argument("--report", required=True, help="Context packet eval JSON report")
    packet_diag_parser.add_argument("--limit", type=int, default=10, help="Max probes to diagnose")
    packet_diag_parser.add_argument("--include-hits", action="store_true", help="Diagnose hits as well as misses")
    packet_diag_parser.add_argument("--output", help="Save diagnosis JSON")
    packet_diag_parser.add_argument("--json", action="store_true", help="Output as JSON")

    packet_maint_parser = subparsers.add_parser("maintain-context-packet", help="Run packet eval plus capped miss backfill. Requires Cairn Pro.")
    packet_maint_parser.add_argument("--artifact-prefix", required=True, help="Prefix for before/backfill/after artifacts")
    packet_maint_parser.add_argument("--sample-size", type=int, default=20)
    packet_maint_parser.add_argument("--budget-tokens", type=int, default=800)
    packet_maint_parser.add_argument("--mode", default="before_edit", choices=["before_edit", "planning", "debug", "review", "command"])
    packet_maint_parser.add_argument("--seed", type=int, default=42)
    packet_maint_parser.add_argument("--probe-cache", help="Load/save fixed packet probe cache")
    packet_maint_parser.add_argument("--threshold", type=float, default=0.72)
    packet_maint_parser.add_argument("--max-connections-per-source", type=int, default=1)
    packet_maint_parser.add_argument("--max-edges", type=int, default=10)
    packet_maint_parser.add_argument("--event-types", help="Comma-separated source event types to consider")
    packet_maint_parser.add_argument("--apply", action="store_true", help="Write edges")
    packet_maint_parser.add_argument("--re-eval", action="store_true", help="Run an after eval when --apply is used")
    packet_maint_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    commands = {
        "query": cmd_query,
        "store": cmd_store,
        "remember": cmd_remember,
        "timeline": cmd_timeline,
        "setup": cmd_setup,
        "status": cmd_status,
        "doctor": cmd_doctor,
        "migrate-home": cmd_migrate_home,
        "migrate-embeddings": cmd_migrate_embeddings,
        "migrate-db": cmd_migrate_db,
        "reingest": cmd_reingest,
        "consolidate": cmd_consolidate,
        "rollup": cmd_rollup,
        "link": cmd_link,
        "gc": cmd_gc,
        "backup": cmd_backup,
        "compact": cmd_compact,
        "stats": cmd_stats,
        "activity": cmd_activity,
        "logs": cmd_logs,
        "validate": cmd_validate,
        "serve": cmd_serve,
        "proxy": cmd_proxy,
        "hooks": cmd_hooks,
        "remind": cmd_remind,
        "eval-retrieval": cmd_eval_retrieval,
        "eval-context-packet": cmd_eval_context_packet,
        "backfill-context-packet": cmd_backfill_context_packet,
        "diagnose-context-packet": cmd_diagnose_context_packet,
        "maintain-context-packet": cmd_maintain_context_packet,
        "export-obsidian": cmd_export_obsidian,
    }

    # Wire plugin CLI commands (cairn-pro, etc.)
    try:
        from cairn.plugins import discover_plugins
        for plugin in discover_plugins():
            for cmd_name, setup_func in getattr(plugin, "CLI_COMMANDS", []):
                if cmd_name not in commands:
                    try:
                        setup_func(subparsers)
                        commands[cmd_name] = getattr(plugin, f"cmd_{cmd_name}", None)
                    except Exception as e:
                        print(f"Warning: plugin CLI command '{cmd_name}' failed: {e}", file=sys.stderr)
    except Exception as e:
        logger.debug("Plugin CLI registration failed: %s", e)

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
