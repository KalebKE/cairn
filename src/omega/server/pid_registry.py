"""PID registry for local OMEGA server processes.

The registry is intentionally local-only. It supports startup cleanup and
database-lock diagnostics without exposing process metadata outside the user's
machine.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

_REGISTRY_DIR = Path.home() / ".omega" / "pids"


def _registry_dir() -> Path:
    _REGISTRY_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        os.chmod(_REGISTRY_DIR, 0o700)
    except OSError:
        pass
    return _REGISTRY_DIR


def _pid_file(pid: int | None = None) -> Path:
    return _registry_dir() / f"{pid or os.getpid()}.json"


def _read_entry(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_entry(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    tmp.replace(path)


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _parent_pid(pid: int) -> int | None:
    if pid == os.getpid():
        return os.getppid()
    try:
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except Exception:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        return int(text.splitlines()[0].strip())
    except ValueError:
        return None


def register_pid(transport: str = "stdio", port: int | None = None) -> dict[str, Any]:
    """Register the current process for lock diagnostics."""
    pid = os.getpid()
    now = time.time()
    entry: dict[str, Any] = {
        "pid": pid,
        "ppid": os.getppid(),
        "transport": transport,
        "port": port,
        "registered_at": now,
        "updated_at": now,
    }
    _write_entry(_pid_file(pid), entry)
    return entry


def unregister_pid(pid: int | None = None) -> None:
    """Remove a process from the local registry."""
    try:
        _pid_file(pid).unlink(missing_ok=True)
    except OSError:
        pass


def list_registered_pids(clean_stale: bool = True) -> list[dict[str, Any]]:
    """Return live registered OMEGA server processes."""
    entries: list[dict[str, Any]] = []
    for path in _registry_dir().glob("*.json"):
        entry = _read_entry(path)
        try:
            pid = int(entry.get("pid")) if entry else int(path.stem)
        except (TypeError, ValueError):
            if clean_stale:
                path.unlink(missing_ok=True)
            continue
        if not _process_exists(pid):
            if clean_stale:
                path.unlink(missing_ok=True)
            continue
        if entry is None:
            entry = {"pid": pid}
        entry["ppid_current"] = _parent_pid(pid)
        entries.append(entry)
    return entries


def kill_orphaned_servers() -> int:
    """Terminate registered stdio server processes whose parent exited."""
    killed = 0
    current_pid = os.getpid()
    for entry in list_registered_pids(clean_stale=True):
        pid = int(entry.get("pid", 0))
        if pid <= 0 or pid == current_pid:
            continue
        if entry.get("transport") == "http":
            continue
        ppid = entry.get("ppid_current")
        if ppid != 1:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
            unregister_pid(pid)
        except ProcessLookupError:
            unregister_pid(pid)
        except PermissionError:
            continue
        except OSError:
            continue
    return killed


def format_lock_diagnostic() -> str:
    """Format registered process details for database lock errors."""
    entries = list_registered_pids(clean_stale=True)
    if not entries:
        return "No registered OMEGA server processes found"
    parts = []
    for entry in entries[:8]:
        pid = entry.get("pid")
        transport = entry.get("transport", "unknown")
        port = entry.get("port")
        ppid = entry.get("ppid_current", entry.get("ppid"))
        label = f"pid={pid} transport={transport} ppid={ppid}"
        if port:
            label += f" port={port}"
        parts.append(label)
    suffix = "" if len(entries) <= 8 else f" (+{len(entries) - 8} more)"
    return "Registered OMEGA processes: " + "; ".join(parts) + suffix


__all__ = [
    "format_lock_diagnostic",
    "kill_orphaned_servers",
    "list_registered_pids",
    "register_pid",
    "unregister_pid",
]
