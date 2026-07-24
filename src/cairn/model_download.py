"""Verified model provisioning.

Downloads ONNX model files from version-pinned GitHub Release assets (see
models_manifest.py), verifying every file's sha256. The checksum is the
guarantee that a download IS the intended model — the exact integrity check
whose absence let the wrong model run silently. HuggingFace is a secondary
mirror; the sha256 gates it too, so a drifted mirror fails loudly rather than
installing a wrong model.

Used by `cairn setup`, by auto-download-on-first-run (embedding.py), and by the
reranker. Progress and notices go to stderr so they never corrupt an MCP stdio
channel.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from cairn.models_manifest import ASSET_BASE, MODELS
from cairn.paths import cache_dir


def models_root() -> Path:
    return cache_dir() / "models"


def model_dir(model_name: str) -> Path:
    return models_root() / MODELS[model_name]["dir"]


def is_model_present(model_name: str) -> bool:
    """Cheap check: all manifest files exist with the expected size."""
    spec = MODELS.get(model_name)
    if not spec:
        return False
    d = model_dir(model_name)
    for f in spec["files"]:
        p = d / f["name"]
        if not p.exists() or p.stat().st_size != f["size"]:
            return False
    return True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(url: str, dest: Path, expected_sha: str, *, quiet: bool) -> None:
    """Stream url → dest.tmp, computing sha256 as we go; rename on match."""
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": "cairn-model-download"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (https only)
        total = int(resp.headers.get("Content-Length", 0))
        got = 0
        last_pct = -1
        with open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                got += len(chunk)
                if not quiet and total > 4_000_000:
                    pct = int(got * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"\r    {dest.name}: {pct}%", end="", file=sys.stderr, flush=True)
                        last_pct = pct
    if not quiet and total > 4_000_000:
        print("", file=sys.stderr)
    actual = h.hexdigest()
    if actual != expected_sha:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"Checksum mismatch for {dest.name}: expected {expected_sha[:12]}…, "
            f"got {actual[:12]}… (refusing to install — this is NOT the intended model)"
        )
    tmp.replace(dest)


class ModelDownloadError(RuntimeError):
    pass


def download_model(model_name: str, *, quiet: bool = False, force: bool = False) -> Path:
    """Download + verify all files for a model. Returns the model dir.

    Tries the pinned GitHub Release asset first, then the HuggingFace mirror.
    Every file is sha256-verified regardless of source.
    """
    spec = MODELS.get(model_name)
    if not spec:
        raise ModelDownloadError(f"Unknown model '{model_name}' (not in manifest)")

    d = model_dir(model_name)
    d.mkdir(parents=True, exist_ok=True)

    if not quiet:
        total_mb = sum(f["size"] for f in spec["files"]) / 1e6
        print(f"  Downloading {model_name} ({total_mb:.0f} MB, verified)…", file=sys.stderr)

    for f in spec["files"]:
        dest = d / f["name"]
        if dest.exists() and not force and dest.stat().st_size == f["size"] and _sha256(dest) == f["sha256"]:
            continue  # already present and verified
        sources = [ASSET_BASE + f["asset"]]
        if f.get("hf_url"):
            sources.append(f["hf_url"])
        last_err: Optional[Exception] = None
        for url in sources:
            try:
                _download_one(url, dest, f["sha256"], quiet=quiet)
                last_err = None
                break
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e  # try next source
            except ModelDownloadError:
                raise  # checksum mismatch is fatal — don't silently try a mirror and accept it
        if last_err is not None:
            raise ModelDownloadError(
                f"Failed to download {f['name']} for {model_name} from any source: {last_err}"
            )

    _write_sidecar(model_name, d)
    if not quiet:
        print(f"  {model_name} ready at {d}", file=sys.stderr)
    return d


def _write_sidecar(model_name: str, d: Path) -> None:
    """Write cairn.json from code (never downloaded) for models that need one.

    The sidecar controls pooling; a wrong sidecar silently corrupts embeddings,
    so it is generated from the package's own config, not fetched.
    """
    import json

    key = MODELS[model_name].get("sidecar")
    if key != "gte":
        return
    from cairn.embedding import GTE_SIDECAR

    cfg = {k: v for k, v in GTE_SIDECAR.items() if k != "source"}
    (d / "cairn.json").write_text(json.dumps(cfg, indent=1))


def ensure_model(model_name: str, *, quiet: bool = False) -> Optional[Path]:
    """Return the model dir, auto-downloading if absent (opt out with
    CAIRN_NO_MODEL_DOWNLOAD=1). Returns None if absent and download is disabled
    or fails — the caller degrades loudly (hash fallback + drift warning)."""
    if is_model_present(model_name):
        return model_dir(model_name)
    if os.environ.get("CAIRN_NO_MODEL_DOWNLOAD") == "1":
        return None
    try:
        return download_model(model_name, quiet=quiet)
    except Exception as e:  # network/offline/checksum — never hang or crash the caller
        print(f"  Model auto-download failed for {model_name}: {e}", file=sys.stderr)
        return None
