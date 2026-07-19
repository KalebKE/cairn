"""Post-install configuration for Claude Desktop.

Finds Claude Desktop's config, backs it up, and injects the Cairn
MCP server entry so Cairn tools appear on next Claude Desktop restart.

Cross-platform: supports macOS and Windows.

Usage: python configure_claude.py [--install-dir PATH]
  --install-dir: Path to Cairn install directory
    macOS default:   ~/Library/Cairn
    Windows default: %LOCALAPPDATA%\\Cairn
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("cairn-configure")


def get_claude_config_path() -> Path:
    """Return the path to Claude Desktop's config file."""
    if sys.platform == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    # Windows
    appdata = os.environ.get("APPDATA")
    if not appdata:
        raise RuntimeError("APPDATA environment variable not set")
    return Path(appdata) / "Claude" / "claude_desktop_config.json"


def backup_config(config_path: Path) -> Path | None:
    """Create a backup of the existing config. Returns backup path or None."""
    if not config_path.exists():
        return None
    backup_path = config_path.with_suffix(".json.bak")
    shutil.copy2(config_path, backup_path)
    print(f"  Backed up config to {backup_path}")
    return backup_path


def _python_path(install_dir: Path) -> Path:
    """Return the platform-appropriate path to the bundled Python."""
    if sys.platform == "darwin":
        return install_dir / "python" / "bin" / "python3"
    return install_dir / "python" / "python.exe"


def build_mcp_entry(install_dir: Path) -> dict:
    """Build the Cairn MCP server entry for Claude Desktop config."""
    return {
        "command": str(_python_path(install_dir)),
        "args": ["-m", "cairn.server.mcp_server"],
        "env": {
            "CAIRN_HOME": str(Path.home() / ".cairn"),
        },
    }


def inject_mcp_config(config_path: Path, install_dir: Path) -> bool:
    """Merge Cairn MCP server entry into Claude Desktop config.

    Returns True if config was written and verified, False otherwise.
    """
    # Read existing config or start fresh
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            log.info("Read existing config: %d keys", len(config))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Could not parse existing config (%s), starting fresh", e)
            config = {}
    else:
        log.info("No existing config file, creating new")
        config = {}

    # Ensure mcpServers key exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Add/update Cairn entry
    mcp_entry = build_mcp_entry(install_dir)
    config["mcpServers"]["cairn"] = mcp_entry
    log.info("MCP entry: command=%s", mcp_entry["command"])

    # Write config
    config_path.parent.mkdir(parents=True, exist_ok=True)
    new_content = json.dumps(config, indent=2, ensure_ascii=False) + "\n"
    try:
        config_path.write_text(new_content, encoding="utf-8")
    except OSError as e:
        log.error("FAILED to write config: %s", e)
        print(f"  ERROR: Could not write config: {e}")
        return False

    # Verify the write succeeded by reading back
    try:
        written = json.loads(config_path.read_text(encoding="utf-8"))
        if "cairn" in written.get("mcpServers", {}):
            log.info("Verified: cairn present in config")
            print(f"  Configured Claude Desktop: {config_path}")
            return True
        else:
            log.error("Verification failed: cairn not in written config")
            print("  ERROR: Config written but cairn entry missing")
            return False
    except (json.JSONDecodeError, OSError) as e:
        log.error("Verification read-back failed: %s", e)
        print(f"  ERROR: Could not verify config: {e}")
        return False


def remove_mcp_config(config_path: Path) -> None:
    """Remove Cairn MCP server entry from Claude Desktop config (for uninstall)."""
    if not config_path.exists():
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    servers = config.get("mcpServers", {})
    if "cairn" in servers:
        del servers["cairn"]
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("  Removed Cairn from Claude Desktop config")


def inject_claude_md(install_dir: Path) -> None:
    """Inject Cairn instruction block into ~/.claude/CLAUDE.md.

    Uses markers to safely update without touching user content.
    Creates a .pre-cairn backup on first append.
    """
    CAIRN_BEGIN = "<!-- Cairn:BEGIN"
    CAIRN_END = "<!-- Cairn:END -->"

    claude_md = Path.home() / ".claude" / "CLAUDE.md"

    # Find the fragment file — try Pro first, fall back to Core
    python_path = _python_path(install_dir)
    try:
        result = subprocess.run(
            [str(python_path), "-c", "import cairn; print(cairn.__file__)"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            cairn_pkg = Path(result.stdout.strip()).parent
            pro_fragment = cairn_pkg / "data" / "claude-md-fragment-pro.md"
            core_fragment = cairn_pkg / "data" / "claude-md-fragment.md"
            fragment_path = pro_fragment if pro_fragment.exists() else core_fragment
        else:
            log.warning("Could not locate cairn package, skipping CLAUDE.md injection")
            return
    except Exception as e:
        log.warning("Could not locate cairn package: %s", e)
        return

    if not fragment_path.exists():
        log.warning("Fragment file not found: %s", fragment_path)
        return

    fragment = fragment_path.read_text(encoding="utf-8")

    if claude_md.exists():
        content = claude_md.read_text(encoding="utf-8")
    else:
        claude_md.parent.mkdir(parents=True, exist_ok=True)
        content = ""

    import re
    if CAIRN_BEGIN in content:
        # Replace existing block
        pattern = re.compile(
            r"<!-- Cairn:BEGIN[^\n]*-->.*?<!-- Cairn:END -->",
            re.DOTALL,
        )
        new_content = pattern.sub(fragment.rstrip(), content)
        if new_content == content:
            print("  CLAUDE.md: Cairn block already up to date")
            return
        claude_md.write_text(new_content, encoding="utf-8")
        print("  CLAUDE.md: Cairn block updated")
    else:
        # First time — back up if file has content
        if content.strip():
            backup = claude_md.with_suffix(".md.pre-cairn")
            if not backup.exists():
                backup.write_text(content, encoding="utf-8")
                print(f"  CLAUDE.md: backed up to {backup.name}")
        separator = "\n" if content and not content.endswith("\n") else ""
        claude_md.write_text(content + separator + fragment, encoding="utf-8")
        print("  CLAUDE.md: Cairn block appended")


def run_cairn_setup(install_dir: Path) -> None:
    """Run `cairn setup` to initialize Cairn data directory."""
    python_path = _python_path(install_dir)
    try:
        result = subprocess.run(
            [str(python_path), "-m", "cairn", "setup"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("  Cairn setup complete")
        else:
            print(f"  WARNING: cairn setup returned code {result.returncode}")
            if result.stderr:
                print(f"  {result.stderr[:200]}")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"  WARNING: Could not run cairn setup: {e}")


def _setup_logging(install_dir: Path) -> None:
    """Configure logging to both console and a log file in the install dir."""
    log_file = install_dir / "configure_claude.log"
    try:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[
                logging.FileHandler(str(log_file), encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
        log.info("Logging to %s", log_file)
    except OSError:
        # Fall back to console-only if we can't write the log file
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        log.warning("Could not create log file at %s", log_file)


def main():
    parser = argparse.ArgumentParser(description="Configure Claude Desktop for Cairn")
    if sys.platform == "darwin":
        default_install_dir = Path.home() / "Library" / "Cairn"
    else:
        default_install_dir = Path(os.environ.get("LOCALAPPDATA", "")) / "Cairn"

    parser.add_argument(
        "--install-dir",
        type=Path,
        default=default_install_dir,
        help="Cairn install directory",
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Remove Cairn config (for uninstaller)",
    )
    args = parser.parse_args()

    _setup_logging(args.install_dir)
    log.info("Platform: %s, Python: %s", sys.platform, sys.version)
    log.info("Install dir: %s", args.install_dir)
    log.info("APPDATA: %s", os.environ.get("APPDATA", "<not set>"))

    config_path = get_claude_config_path()
    log.info("Config path: %s (exists: %s)", config_path, config_path.exists())

    if args.uninstall:
        remove_mcp_config(config_path)
        return

    print("Configuring Cairn for Claude Desktop...")
    backup_config(config_path)
    success = inject_mcp_config(config_path, args.install_dir)
    run_cairn_setup(args.install_dir)

    # Inject Cairn instructions into CLAUDE.md
    try:
        inject_claude_md(args.install_dir)
    except Exception as e:
        log.warning("Could not inject CLAUDE.md: %s", e)
        print(f"  WARNING: Could not update CLAUDE.md: {e}")

    if success:
        print("Done! Restart Claude Desktop to use Cairn.")
        log.info("Configuration completed successfully")
        sys.exit(0)
    else:
        print("WARNING: Could not configure Claude Desktop automatically.")
        print("Run 'Repair Cairn Config' from the Start Menu to fix this.")
        log.error("Configuration FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
