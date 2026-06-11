"""Regression test for the Mar 2026 restructure bug.

In commit b3edc0a the public repo stripped all Pro modules, which moved
license handling from ``omega.license`` to ``omega_platform.license``
(the latter ships in the Pro wheel). Several call sites in the public
CLI and server were left behind still importing ``omega.license``, a
module that no longer exists anywhere. Pro customers who ran
``omega activate <key>`` without the Pro wheel installed saw the cryptic
error "License activation requires omega-pro." with no path forward.

This test scans the public ``omega`` package for any lingering
``from omega.license import ...`` statements so the bug cannot come
back. It is pure AST, runs in milliseconds, and has no runtime deps.
"""

from __future__ import annotations

import ast
import pathlib


def test_no_stale_omega_license_imports() -> None:
    src_root = pathlib.Path(__file__).resolve().parent.parent / "src" / "omega"
    offenders: list[str] = []

    for path in src_root.rglob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            # Skip files that cannot be parsed; unrelated to this invariant.
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "omega.license":
                rel = path.relative_to(src_root.parent.parent)
                offenders.append(f"{rel}:{node.lineno}")

    assert not offenders, (
        "Found stale imports of the removed `omega.license` module. "
        "This module was replaced by explicit Pro activation/status helpers in "
        "`omega_platform.license` (shipped in the Pro wheel) in the Mar 2026 "
        "architecture restructure (commit b3edc0a). Do not use local license "
        "checks to unlock Core behavior; use plugin capabilities instead.\n\n"
        "Offenders:\n" + "\n".join(f"  {o}" for o in offenders)
    )
