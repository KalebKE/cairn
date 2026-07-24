#!/usr/bin/env python3
"""Upload the model files to the pinned GitHub Release as flat-named assets.

The manifest (models_manifest.py) references each file by a flat asset name
(e.g. gte-modernbert-base-onnx__model.onnx) because GitHub release asset names
can't contain slashes. This copies each local model file to its asset name and
uploads it to MODEL_RELEASE_TAG, creating the release if needed.

Run once after generating the manifest (and again only when a model changes).
Requires `gh` authenticated with write access to MODEL_REPO.

    python scripts/upload_model_assets.py            # create/upload
    python scripts/upload_model_assets.py --clobber  # replace existing assets
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from cairn.model_download import models_root  # noqa: E402
from cairn.models_manifest import MODEL_RELEASE_TAG, MODEL_REPO, MODELS  # noqa: E402


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd))
    return subprocess.run(cmd, check=True)


def _release_exists() -> bool:
    r = subprocess.run(
        ["gh", "release", "view", MODEL_RELEASE_TAG, "--repo", MODEL_REPO],
        capture_output=True,
    )
    return r.returncode == 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clobber", action="store_true", help="overwrite existing assets")
    args = ap.parse_args()

    root = models_root()
    # Verify every file exists locally before touching the release.
    missing = []
    for spec in MODELS.values():
        for f in spec["files"]:
            if not (root / spec["dir"] / f["name"]).exists():
                missing.append(f"{spec['dir']}/{f['name']}")
    if missing:
        sys.exit("Missing local model files (download them first):\n  " + "\n  ".join(missing))

    if not _release_exists():
        _run([
            "gh", "release", "create", MODEL_RELEASE_TAG, "--repo", MODEL_REPO,
            "--title", f"Model assets ({MODEL_RELEASE_TAG})",
            "--notes", "Version-pinned, sha256-verified ONNX model assets for cairn "
                       "(see src/cairn/models_manifest.py). Not tied to a package version.",
        ])

    with tempfile.TemporaryDirectory(prefix="cairn-assets-") as tmp:
        staged = []
        for spec in MODELS.values():
            for f in spec["files"]:
                src = root / spec["dir"] / f["name"]
                dst = Path(tmp) / f["asset"]  # flat asset name
                shutil.copyfile(src, dst)
                staged.append(str(dst))
        cmd = ["gh", "release", "upload", MODEL_RELEASE_TAG, "--repo", MODEL_REPO, *staged]
        if args.clobber:
            cmd.append("--clobber")
        _run(cmd)

    print(f"\nUploaded {sum(len(s['files']) for s in MODELS.values())} assets to "
          f"{MODEL_REPO}@{MODEL_RELEASE_TAG}.")


if __name__ == "__main__":
    main()
