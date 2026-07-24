"""Verified model download: the sha256 check is the guarantee that a download
IS the intended model. Corrupt/wrong bytes must fail loudly, never install."""

import hashlib
import io

import pytest

import cairn.model_download as MD


class _FakeResp:
    """Minimal urlopen() context-manager stand-in."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self._buf.read(n)


def _serve(monkeypatch, data: bytes):
    monkeypatch.setattr(MD.urllib.request, "urlopen", lambda req, timeout=60: _FakeResp(data))


def test_download_one_verifies_good_checksum(monkeypatch, tmp_path):
    data = b"the intended model bytes"
    _serve(monkeypatch, data)
    dest = tmp_path / "model.onnx"
    MD._download_one("https://x/model.onnx", dest, hashlib.sha256(data).hexdigest(), quiet=True)
    assert dest.read_bytes() == data
    assert not (tmp_path / "model.onnx.tmp").exists()  # tmp cleaned up


def test_download_one_rejects_bad_checksum(monkeypatch, tmp_path):
    _serve(monkeypatch, b"WRONG MODEL bytes")
    dest = tmp_path / "model.onnx"
    with pytest.raises(MD.ModelDownloadError, match="Checksum mismatch"):
        MD._download_one("https://x/model.onnx", dest, hashlib.sha256(b"expected").hexdigest(), quiet=True)
    assert not dest.exists()  # never installed
    assert not (tmp_path / "model.onnx.tmp").exists()  # bad tmp removed


def test_download_model_full_flow_verifies(monkeypatch, tmp_path):
    """A whole model downloads + verifies from fake sources."""
    files = {"model.onnx": b"MODEL", "config.json": b"{}"}
    manifest = {
        "toy-model": {
            "dir": "toy-onnx",
            "sidecar": None,
            "files": [
                {"name": n, "asset": f"toy-onnx__{n}", "sha256": hashlib.sha256(b).hexdigest(),
                 "size": len(b), "hf_url": f"https://hf/{n}"}
                for n, b in files.items()
            ],
        }
    }
    monkeypatch.setattr(MD, "MODELS", manifest)
    monkeypatch.setattr(MD, "models_root", lambda: tmp_path)

    def fake_urlopen(req, timeout=60):
        # asset name is the last path segment; map back to the file bytes
        name = req.full_url.rsplit("__", 1)[-1] if "__" in req.full_url else req.full_url.rsplit("/", 1)[-1]
        return _FakeResp(files[name])

    monkeypatch.setattr(MD.urllib.request, "urlopen", fake_urlopen)

    d = MD.download_model("toy-model", quiet=True)
    assert (d / "model.onnx").read_bytes() == b"MODEL"
    assert MD.is_model_present("toy-model")


def test_ensure_model_returns_present_without_download(monkeypatch):
    monkeypatch.setattr(MD, "is_model_present", lambda name: True)
    monkeypatch.setattr(MD, "model_dir", lambda name: "/already/here")
    # If it tried to download, this would explode:
    monkeypatch.setattr(MD, "download_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download")))
    assert MD.ensure_model("gte-modernbert-base") == "/already/here"


def test_ensure_model_respects_no_download_env(monkeypatch):
    monkeypatch.setattr(MD, "is_model_present", lambda name: False)
    monkeypatch.setenv("CAIRN_NO_MODEL_DOWNLOAD", "1")
    assert MD.ensure_model("gte-modernbert-base") is None


def test_manifest_shape_is_sane():
    from cairn.models_manifest import ASSET_BASE, MODELS

    assert ASSET_BASE.startswith("https://github.com/") and ASSET_BASE.endswith("/")
    assert "gte-modernbert-base" in MODELS and "ms-marco-MiniLM-L-6-v2" in MODELS
    for name, spec in MODELS.items():
        assert spec["files"], name
        for f in spec["files"]:
            assert len(f["sha256"]) == 64, (name, f["name"])
            assert f["size"] > 0
            assert "/" not in f["asset"]  # flat asset name for a GH release
