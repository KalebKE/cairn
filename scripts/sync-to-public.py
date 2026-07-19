#!/usr/bin/env python3
"""Sync all source files from private Cairn repo to the public cairn-public repo.

Since the single-package merge, ALL source and test files are synced.
Pro features are gated by optional extras, not by file exclusion.

Usage:
    python3 scripts/sync-to-public.py                  # dry-run (default)
    python3 scripts/sync-to-public.py --apply           # actually copy files
    python3 scripts/sync-to-public.py --apply --test    # copy + run tests
    python3 scripts/sync-to-public.py --report          # full sync report
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

# ── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PRIVATE_ROOT = SCRIPT_DIR.parent
PUBLIC_ROOT = PRIVATE_ROOT.parent / "cairn-public"
MANIFEST_PATH = PRIVATE_ROOT / "sync-manifest.yaml"


def load_manifest() -> dict:
    """Load and validate the sync manifest."""
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)
    with open(MANIFEST_PATH) as f:
        return yaml.safe_load(f)


def handle_file_to_dir_transitions(manifest: dict, dry_run: bool) -> list[str]:
    """Delete old single files in public when private has converted them to directories."""
    actions = []
    for entry in manifest.get("file_to_dir_transitions", []):
        path = entry.split("#")[0].strip()
        old_file = PUBLIC_ROOT / (path + ".py")
        new_dir = PRIVATE_ROOT / path
        if old_file.exists() and old_file.is_file() and new_dir.exists() and new_dir.is_dir():
            if dry_run:
                actions.append(f"WOULD DELETE {old_file.relative_to(PUBLIC_ROOT)} (replaced by directory)")
            else:
                old_file.unlink()
                actions.append(f"DELETED {old_file.relative_to(PUBLIC_ROOT)} (replaced by directory)")
    return actions



# File patterns to skip during sync
_SKIP_NAMES = {'.DS_Store'}
_SKIP_SUFFIXES = {'.db', '.pyc'}


def _should_skip(path: Path) -> bool:
    return path.name in _SKIP_NAMES or path.suffix in _SKIP_SUFFIXES

def resolve_sync_files(manifest: dict) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for all files to sync."""
    pairs = []
    for f in manifest.get("sync_src", []):
        name = f.split("#")[0].strip()
        src = PRIVATE_ROOT / "src/cairn" / name
        dst = PUBLIC_ROOT / "src/cairn" / name
        if src.is_dir():
            # Recursively add all files from the directory
            for child in sorted(src.rglob("*")):
                if child.is_file() and "__pycache__" not in str(child) and not _should_skip(child):
                    rel = child.relative_to(PRIVATE_ROOT / "src/cairn")
                    pairs.append((child, PUBLIC_ROOT / "src/cairn" / rel))
        elif src.exists():
            pairs.append((src, dst))
    for f in manifest.get("sync_tests", []):
        name = f.split("#")[0].strip()
        src = PRIVATE_ROOT / "tests" / name
        dst = PUBLIC_ROOT / "tests" / name
        if src.exists():
            pairs.append((src, dst))
    return pairs


def _build_excluded_prefixes(manifest: dict) -> list[Path]:
    """Build list of path prefixes that should never be synced (private_only)."""
    prefixes = []
    for entry in manifest.get("private_only", []):
        name = entry.split("#")[0].strip()
        prefixes.append(PRIVATE_ROOT / name)
    return prefixes


def _is_excluded(path: Path, excluded_prefixes: list[Path]) -> bool:
    """Check if a path falls under any excluded prefix."""
    for prefix in excluded_prefixes:
        try:
            path.relative_to(prefix)
            return True
        except ValueError:
            continue
    return False


def resolve_sync_top(manifest: dict) -> list[tuple[Path, Path]]:
    """Return (src, dst) pairs for top-level files/dirs to sync."""
    excluded = _build_excluded_prefixes(manifest)
    pairs = []
    for entry in manifest.get("sync_top", []):
        name = entry.split("#")[0].strip()
        src = PRIVATE_ROOT / name
        dst = PUBLIC_ROOT / name
        if src.is_dir():
            for child in sorted(src.rglob("*")):
                if child.is_file() and "__pycache__" not in str(child) and not _should_skip(child) and not _is_excluded(child, excluded):
                    rel = child.relative_to(PRIVATE_ROOT)
                    pairs.append((child, PUBLIC_ROOT / rel))
        elif src.exists() and not _is_excluded(src, excluded):
            pairs.append((src, dst))
    return pairs


def sync_file(src: Path, dst: Path, dry_run: bool) -> str:
    """Copy src to dst. Returns status string."""
    if not src.exists():
        return "SKIP (missing)"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        try:
            if src.read_bytes() == dst.read_bytes():
                return "identical"
        except (IsADirectoryError, PermissionError):
            pass

    if dry_run:
        return "WOULD COPY"

    shutil.copy2(src, dst)
    return "copied"


def main():
    parser = argparse.ArgumentParser(
        description="Sync files from private Cairn to public cairn-public",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python3 scripts/sync-to-public.py              # Dry run — see what would change
  python3 scripts/sync-to-public.py --report      # Full sync report
  python3 scripts/sync-to-public.py --apply       # Actually sync files
  python3 scripts/sync-to-public.py --apply --test # Sync + run public tests
""")
    parser.add_argument("--apply", action="store_true",
                        help="Actually copy files (default: dry-run)")
    parser.add_argument("--test", action="store_true",
                        help="Run tests in cairn-public after sync")
    parser.add_argument("--report", action="store_true",
                        help="Full sync report")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show per-file status")
    args = parser.parse_args()

    dry_run = not args.apply

    if not PRIVATE_ROOT.exists() or not PUBLIC_ROOT.exists():
        print(f"ERROR: Repos not found at expected paths")
        print(f"  Private: {PRIVATE_ROOT} (exists: {PRIVATE_ROOT.exists()})")
        print(f"  Public:  {PUBLIC_ROOT} (exists: {PUBLIC_ROOT.exists()})")
        sys.exit(1)

    manifest = load_manifest()

    print("=" * 64)
    print("  Cairn Single-Package Sync")
    print("=" * 64)
    print(f"  Private repo:  {PRIVATE_ROOT}")
    print(f"  Public repo:   {PUBLIC_ROOT}")
    mode = "DRY RUN" if dry_run else "APPLY"
    print(f"  Mode:          {mode}")
    print()

    # ── 0. File-to-directory transitions ─────────────────────────────────
    transitions = handle_file_to_dir_transitions(manifest, dry_run)
    if transitions:
        print(f"[0/4] File-to-directory transitions ({len(transitions)})")
        for t in transitions:
            print(f"  {t}")
        print()

    # ── 1. Source + test files ───────────────────────────────────────────
    sync_pairs = resolve_sync_files(manifest)
    print(f"[1/4] Source & test files ({len(sync_pairs)} files)")

    copied = 0
    identical = 0
    for src, dst in sync_pairs:
        status = sync_file(src, dst, dry_run)
        if status in ("copied", "WOULD COPY"):
            copied += 1
            rel = src.relative_to(PRIVATE_ROOT)
            marker = "~" if dry_run else ">"
            print(f"  {marker} {rel}")
        elif status == "identical":
            identical += 1
            if args.verbose:
                rel = src.relative_to(PRIVATE_ROOT)
                print(f"  = {rel}")

    print(f"  Result: {copied} to sync, {identical} already identical")
    print()

    # ── 2. Top-level files ──────────────────────────────────────────────
    top_pairs = resolve_sync_top(manifest)
    print(f"[2/4] Top-level files ({len(top_pairs)} files)")

    top_copied = 0
    top_identical = 0
    for src, dst in top_pairs:
        status = sync_file(src, dst, dry_run)
        if status in ("copied", "WOULD COPY"):
            top_copied += 1
            rel = src.relative_to(PRIVATE_ROOT)
            marker = "~" if dry_run else ">"
            print(f"  {marker} {rel}")
        elif status == "identical":
            top_identical += 1
            if args.verbose:
                rel = src.relative_to(PRIVATE_ROOT)
                print(f"  = {rel}")

    print(f"  Result: {top_copied} to sync, {top_identical} already identical")
    print()

    # ── 3. Public-only preservation ──────────────────────────────────────
    public_only = manifest.get("public_only", [])
    print(f"[3/4] Public-only files ({len(public_only)} — never touched)")
    for f in public_only:
        public_path = PUBLIC_ROOT / f
        status = "OK" if public_path.exists() else "MISSING"
        print(f"  {status}: {f}")
    print()

    # ── 4. Tests (optional) ──────────────────────────────────────────────
    if args.test and not dry_run:
        print("[4/4] Running cairn-public tests...")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short", "tests/"],
            cwd=PUBLIC_ROOT,
        )
        if result.returncode != 0:
            print("\n  FAIL: Tests failed!")
            sys.exit(1)
        print("  OK: All tests passed")
    else:
        print("[4/4] Tests: skipped (use --apply --test)")
    print()

    # ── Report mode ──────────────────────────────────────────────────────
    if args.report:
        private_only = manifest.get("private_only", [])
        print("=" * 64)
        print("  SYNC REPORT")
        print("=" * 64)
        print(f"  Source & tests (sync):   {len(sync_pairs):>3} files")
        print(f"  Top-level (sync):        {len(top_pairs):>3} files")
        print(f"  Private-only (blocked):  {len(private_only):>3} entries")
        print(f"  Public-only (kept):      {len(public_only):>3} entries")
        print(f"  File→dir transitions:    {len(transitions):>3}")
        print()
        total_to_sync = copied + top_copied
        total_identical = identical + top_identical
        if total_to_sync > 0:
            print(f"  STATUS: {total_to_sync} files to sync, {total_identical} already identical")
        else:
            print(f"  STATUS: ALL {total_identical} files identical — repos in sync")
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 64)
    if dry_run:
        print("  DRY RUN complete. Use --apply to sync files.")
    else:
        print("  Sync complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
