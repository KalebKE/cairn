"""Constraint enforcement for the Cairn bridge (peeled from __init__).

Loads, checks, lists, and persists project/session constraints stored as
files under ``CAIRN_HOME/constraints``. The constraints directory is
late-bound through the package module so a relocated CAIRN_HOME resolves.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import cairn.bridge as _bridge
from cairn import json_compat as json

logger = logging.getLogger("cairn.bridge.constraints")


def _load_constraints(project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load constraint rules for a project from ~/.cairn/constraints/.

    Loads global.json first, then <project-name>.json if project is given.
    Returns merged list of rule dicts.
    """
    rules: List[Dict[str, Any]] = []
    if not _bridge.CONSTRAINTS_DIR.exists():
        return rules

    # Global constraints
    global_file = _bridge.CONSTRAINTS_DIR / "global.json"
    if global_file.exists():
        try:
            data = json.loads(global_file.read_text())
            for r in data.get("rules", []):
                r["source"] = "global"
                rules.append(r)
        except Exception as e:
            logger.debug(f"Failed to load global constraints: {e}")

    # Project-specific constraints
    if project:
        proj_name = Path(project).name
        proj_file = _bridge.CONSTRAINTS_DIR / f"{proj_name}.json"
        if proj_file.exists():
            try:
                data = json.loads(proj_file.read_text())
                for r in data.get("rules", []):
                    r["source"] = proj_name
                    rules.append(r)
            except Exception as e:
                logger.debug(f"Failed to load {proj_name} constraints: {e}")

    return rules


def check_constraints(file_path: str, project: Optional[str] = None) -> List[Dict[str, Any]]:
    """Check a file path against loaded constraint rules.

    Returns list of matching constraints with severity and message.
    """
    import fnmatch

    rules = _load_constraints(project)
    if not rules:
        return []

    matches = []
    filename = os.path.basename(file_path)

    for rule in rules:
        pattern = rule.get("pattern", "")
        if not pattern:
            continue
        # Match against filename or full path
        if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(file_path, pattern):
            matches.append(
                {
                    "pattern": pattern,
                    "constraint": rule.get("constraint", ""),
                    "severity": rule.get("severity", "warn"),
                    "source": rule.get("source", "unknown"),
                }
            )

    return matches


def list_constraints(project: Optional[str] = None) -> Dict[str, Any]:
    """List all loaded constraint rules for a project."""
    rules = _load_constraints(project)
    return {
        "count": len(rules),
        "rules": rules,
        "constraints_dir": str(_bridge.CONSTRAINTS_DIR),
    }


def save_constraints(
    rules: List[Dict[str, Any]],
    project: Optional[str] = None,
) -> Dict[str, Any]:
    """Save constraint rules to the appropriate file.

    If project is given, saves to <project-name>.json, else global.json.
    """
    _bridge.CONSTRAINTS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)

    if project:
        target = _bridge.CONSTRAINTS_DIR / f"{Path(project).name}.json"
    else:
        target = _bridge.CONSTRAINTS_DIR / "global.json"

    # Clean source field from rules before saving
    clean_rules = []
    for r in rules:
        clean = {k: v for k, v in r.items() if k != "source"}
        clean_rules.append(clean)

    data = {"rules": clean_rules}
    target.write_text(json.dumps(data, indent=2))

    return {"saved": str(target), "count": len(clean_rules)}
