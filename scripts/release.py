#!/usr/bin/env python3.11
"""Cut a cairn release: bump the version, commit, tag, and push.

The tag push triggers .github/workflows/release.yml, which builds the wheel +
sdist, publishes to PyPI via OIDC trusted publishing, and creates the GitHub
Release. This script does NOT publish or create the release itself, so there is
exactly one publisher and no double-publish risk. Use `git tag`/push directly
if you prefer; this just automates the bump + preflight checks.

Use when you don't want to wait for or trust GitHub Actions runners.

Usage:
    python3.11 scripts/release.py <version>            # publish for real
    python3.11 scripts/release.py <version> --dry-run  # build + verify only
    python3.11 scripts/release.py <version> --skip-confirm  # CI-like, no prompts

Pre-flight:
    - Working tree clean on main, up to date with origin
    - Version not already tagged
(PyPI credentials are not needed here — the workflow publishes via OIDC.)
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYPROJECT = REPO / "pyproject.toml"
INIT_PY = REPO / "src" / "cairn" / "__init__.py"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO, check=True, **kw)


def step(name: str) -> None:
    print(f"\n=== {name} ===")


def confirm(prompt: str, skip: bool) -> None:
    if skip:
        return
    answer = input(f"\n{prompt} [y/N] ").strip().lower()
    if answer != "y":
        sys.exit("Aborted.")


def preflight(version: str) -> None:
    step("Pre-flight")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"Version must be X.Y.Z, got {version!r}")

    # No PyPI token needed here anymore: publishing is done by the release
    # workflow via OIDC trusted publishing, triggered by the tag this pushes.

    # Only block on uncommitted changes to files this script will modify.
    tracked_targets = ["pyproject.toml", "src/cairn/__init__.py"]
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--"] + tracked_targets,
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        sys.exit(f"Uncommitted changes to release-target files:\n{dirty}")

    branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if branch != "main":
        sys.exit(f"Not on main, on {branch!r}")

    run(["git", "fetch", "origin", "main"])
    behind = subprocess.run(
        ["git", "rev-list", "--count", "HEAD..origin/main"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if behind != "0":
        sys.exit(f"Local main is {behind} commits behind origin/main. Pull first.")

    tag = f"v{version}"
    existing = subprocess.run(
        ["git", "tag", "-l", tag], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if existing:
        sys.exit(f"Tag {tag} already exists locally.")

    print(f"  OK: version={version}, branch=main, clean, no tag {tag}")


def bump_version(version: str) -> None:
    step(f"Bumping version to {version}")
    for path, pattern, replacement in [
        (PYPROJECT, r'^version = "[^"]+"', f'version = "{version}"'),
        (INIT_PY, r'^__version__ = "[^"]+"', f'__version__ = "{version}"'),
    ]:
        text = path.read_text()
        if not re.search(pattern, text, flags=re.MULTILINE):
            sys.exit(f"Pattern not found in {path}: {pattern!r}")
        new = re.sub(pattern, replacement, text, count=1, flags=re.MULTILINE)
        if new == text:
            print(f"  unchanged {path.relative_to(REPO)}: already at {version}")
        else:
            path.write_text(new)
            print(f"  updated {path.relative_to(REPO)}")


def build() -> tuple[Path, Path]:
    step("Building wheel + sdist")
    dist = REPO / "dist"
    if dist.exists():
        for f in dist.iterdir():
            f.unlink()
    run([sys.executable, "-m", "build", "--wheel", "--sdist"])
    # Package builds as cairn-*, not cairn_memory-* (a former name).
    wheels = list(dist.glob("cairn-*.whl"))
    sdists = list(dist.glob("cairn-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        sys.exit(f"Expected 1 wheel + 1 sdist, got {wheels=} {sdists=}")
    return wheels[0], sdists[0]


def verify(wheel: Path, expected_version: str) -> None:
    step("Verifying wheel in clean venv")
    with tempfile.TemporaryDirectory(prefix="cairn-mem-verify-") as tmp:
        env_dir = Path(tmp) / "venv"
        venv.create(str(env_dir), with_pip=True)
        py = env_dir / "bin" / "python3.11"
        if not py.exists():
            py = env_dir / "bin" / "python"
        run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])
        proc = subprocess.run(
            [str(py), "-c", "import cairn; print(cairn.__version__)"],
            capture_output=True, text=True, check=True,
        )
        installed = proc.stdout.strip()
        if installed != expected_version:
            sys.exit(f"Wheel reports {installed!r}, expected {expected_version!r}")
        print(f"  OK: installed and reported version={installed}")


def git_commit_tag_push(version: str) -> None:
    step("Committing + tagging + pushing")
    run(["git", "add", "pyproject.toml", "src/cairn/__init__.py"])
    staged = subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if staged:
        run(["git", "commit", "-m", f"chore: release v{version}"])
    else:
        print("  no version-file changes to commit (idempotent re-run)")
    existing = subprocess.run(
        ["git", "tag", "-l", f"v{version}"], cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    if not existing:
        run(["git", "tag", f"v{version}"])
    run(["git", "push", "origin", "main"])
    run(["git", "push", "origin", f"v{version}"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="Version X.Y.Z")
    ap.add_argument("--dry-run", action="store_true", help="Build + verify only; do not publish or push")
    ap.add_argument("--skip-confirm", action="store_true", help="Skip interactive confirmation")
    args = ap.parse_args()

    preflight(args.version)
    bump_version(args.version)
    wheel, sdist = build()
    verify(wheel, args.version)

    if args.dry_run:
        print(f"\nDRY RUN: built {wheel.name} + {sdist.name}; would commit + push v{args.version}")
        print("Reverting version bump...")
        run(["git", "checkout", "--", "pyproject.toml", "src/cairn/__init__.py"])
        return 0

    # PyPI publish + GitHub release are done by .github/workflows/release.yml,
    # triggered by the tag push below. This script only bumps + tags + pushes,
    # so there is exactly one publisher (no double-publish).
    confirm(f"Commit + tag v{args.version} + push (triggers PyPI publish workflow)?", args.skip_confirm)
    git_commit_tag_push(args.version)

    step("Done")
    print(f"cairn {args.version} tagged + pushed. The release workflow now builds,")
    print("publishes to PyPI, and creates the GitHub release.")
    print("  Actions: https://github.com/KalebKE/cairn/actions")
    print(f"  PyPI:    https://pypi.org/project/cairn/{args.version}/ (after the workflow)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
