#!/usr/bin/env python3
"""Regenerate src/cairn/models_manifest.py from the models on disk.

Run this after adding/updating a model in ~/.cache/cairn/models. It computes
the sha256 + size of each downloadable file and writes the manifest module.
Then upload the same files as assets to the MODEL_RELEASE_TAG GitHub release
(scripts/upload_model_assets.py).
"""

import hashlib
import os
import pprint
from pathlib import Path

CACHE = Path(os.path.expanduser("~/.cache/cairn/models"))

# model_name -> (local_dir, [downloaded files], sidecar_key, hf_repo, hf_subpath_for_onnx)
# hf_* give a secondary mirror; the sha256 still gates every download, so a
# drifted mirror fails loudly instead of installing silently.
SPEC = {
    "gte-modernbert-base": {
        "dir": "gte-modernbert-base-onnx",
        "files": ["model.onnx", "config.json", "tokenizer_config.json", "tokenizer.json"],
        "sidecar": "gte",
        "hf_repo": "Alibaba-NLP/gte-modernbert-base",
        "hf_onnx_subdir": "onnx",  # model.onnx lives under onnx/ on HF
    },
    "ms-marco-MiniLM-L-6-v2": {
        "dir": "ms-marco-MiniLM-L-6-v2-onnx",
        "files": ["model.onnx", "config.json", "tokenizer.json"],
        "sidecar": None,
        "hf_repo": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "hf_onnx_subdir": "onnx",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    models = {}
    for name, spec in SPEC.items():
        d = spec["dir"]
        hf_base = f"https://huggingface.co/{spec['hf_repo']}/resolve/main/"
        entry = {"dir": d, "sidecar": spec["sidecar"], "files": []}
        for fn in spec["files"]:
            p = CACHE / d / fn
            if not p.exists():
                raise SystemExit(f"missing {p} — download the model first")
            hf_path = f"{spec['hf_onnx_subdir']}/{fn}" if fn == "model.onnx" else fn
            entry["files"].append({
                "name": fn,
                "asset": f"{d}__{fn}",  # flat asset name for the GH release
                "sha256": sha256(p),
                "size": p.stat().st_size,
                "hf_url": hf_base + hf_path,
            })
        models[name] = entry
    return models


HEADER = '''"""Version-pinned, checksum-verified model manifest.

Models are too big to bundle in a wheel (gte-modernbert-base ~569MB), so they
download from GitHub Release assets on a PUBLIC repo, pinned by release tag and
verified by sha256. The sha256 is the guarantee that a download IS the intended
model — the missing integrity check is part of how the wrong model ran silently.

Assets live on ONE release (MODEL_RELEASE_TAG), decoupled from the package
version so a patch release does not re-upload ~600MB. HuggingFace URLs are a
secondary mirror; the sha256 gates every download either way. Regenerate with
scripts/gen_model_manifest.py.
"""

MODEL_REPO = "KalebKE/cairn"
MODEL_RELEASE_TAG = "models-v1"
ASSET_BASE = f"https://github.com/{MODEL_REPO}/releases/download/{MODEL_RELEASE_TAG}/"

MODELS = '''


def main() -> None:
    models = build()
    out = Path(__file__).resolve().parent.parent / "src" / "cairn" / "models_manifest.py"
    out.write_text(HEADER + pprint.pformat(models, indent=1, sort_dicts=False, width=100) + "\n")
    print(f"wrote {out}")
    for name, e in models.items():
        total = sum(f["size"] for f in e["files"]) / 1e6
        print(f"  {name}: {len(e['files'])} files, {total:.0f} MB")


if __name__ == "__main__":
    main()
