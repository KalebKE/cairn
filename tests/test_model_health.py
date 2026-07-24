"""Model-drift prevention: `model_health_warnings()` must loudly flag when Cairn
is running a non-intended model, silently accept intentional overrides, and the
session-start hook must actually print the warnings.

Regression guard for the incident where Cairn ran ms-marco for months while
everyone believed it ran bge-reranker-v2-m3, undetected.
"""

import io
from contextlib import redirect_stdout

import pytest

import cairn.embedding as E
import cairn.reranker as R
from cairn.model_health import model_health_warnings


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # Neutralize any ambient env overrides so tests control resolution.
    monkeypatch.delenv("CAIRN_ONNX_MODEL_DIR", raising=False)
    monkeypatch.delenv("CAIRN_RERANKER_MODEL", raising=False)
    monkeypatch.delenv("CAIRN_CROSS_ENCODER", raising=False)
    yield


# --------------------------------------------------------------------------
# Embedder drift
# --------------------------------------------------------------------------


def test_no_warning_when_embedder_is_intended(monkeypatch):
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/fake/gte")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": E.INTENDED_EMBEDDING_MODEL})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    # reranker side neutral (on disk, intended)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: (R.INTENDED_RERANKER_MODEL, "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    assert model_health_warnings() == []


def test_embedder_fallback_warns_with_fix(monkeypatch):
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/fake/minilm")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": "all-MiniLM-L6-v2"})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: (R.INTENDED_RERANKER_MODEL, "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    w = model_health_warnings()
    assert any("all-MiniLM-L6-v2" in x and E.INTENDED_EMBEDDING_MODEL in x for x in w)
    assert any("cairn setup --download-model" in x for x in w)


def test_legacy_bge_warns_with_migrate_hint(monkeypatch):
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/fake/bge")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": "bge-small-en-v1.5"})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: (R.INTENDED_RERANKER_MODEL, "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    w = model_health_warnings()
    assert any("migrate-embeddings" in x for x in w), w


def test_explicit_embedder_override_silences_warning(monkeypatch):
    monkeypatch.setenv("CAIRN_ONNX_MODEL_DIR", "/some/custom/model")
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/some/custom/model")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": "custom-thing"})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: (R.INTENDED_RERANKER_MODEL, "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    # user chose it → no embedder drift warning
    assert not any("Embedding model" in x for x in model_health_warnings())


def test_degraded_embeddings_warn_and_dominate(monkeypatch):
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/x")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": "all-MiniLM-L6-v2"})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: True)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: (R.INTENDED_RERANKER_MODEL, "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    w = model_health_warnings()
    assert any("DEGRADED" in x for x in w)
    # degraded dominates — don't also emit the which-model line
    assert not any("not the intended" in x for x in w)


# --------------------------------------------------------------------------
# Reranker drift
# --------------------------------------------------------------------------


def test_reranker_non_intended_without_override_warns(monkeypatch):
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/g")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": E.INTENDED_EMBEDDING_MODEL})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: ("bge-reranker-v2-m3", "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    w = model_health_warnings()
    assert any("bge-reranker-v2-m3" in x and R.INTENDED_RERANKER_MODEL in x for x in w)


def test_reranker_override_silences_warning(monkeypatch):
    monkeypatch.setenv("CAIRN_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/g")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": E.INTENDED_EMBEDDING_MODEL})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: ("bge-reranker-v2-m3", "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: "/d")
    assert not any("not the intended" in x for x in model_health_warnings())


def test_cross_encoder_disabled_suppresses_reranker_warnings(monkeypatch):
    monkeypatch.setenv("CAIRN_CROSS_ENCODER", "0")
    monkeypatch.setattr(E, "_get_onnx_model_dir", lambda: "/g")
    monkeypatch.setattr(E, "get_embedding_model_info", lambda: {"model_name": E.INTENDED_EMBEDDING_MODEL})
    monkeypatch.setattr(E, "is_embedding_degraded", lambda: False)
    monkeypatch.setattr(R, "_resolve_reranker_model", lambda: ("bge-reranker-v2-m3", "/d"))
    monkeypatch.setattr(R, "_get_model_dir", lambda: None)  # absent
    assert not any("Reranker" in x for x in model_health_warnings())


# --------------------------------------------------------------------------
# Reranker resolver: silent-fallback path is GONE by construction
# --------------------------------------------------------------------------


def test_resolver_returns_intended_by_default(monkeypatch):
    monkeypatch.delenv("CAIRN_RERANKER_MODEL", raising=False)
    assert R._resolve_reranker_model()[0] == R.INTENDED_RERANKER_MODEL


def test_resolver_ignores_bge_on_disk(monkeypatch, tmp_path):
    # Even if a bge model.onnx exists on disk, without the env override the
    # resolver must NOT silently pick it — that was the original bug.
    monkeypatch.delenv("CAIRN_RERANKER_MODEL", raising=False)
    (tmp_path / "model.onnx").write_bytes(b"x")
    monkeypatch.setitem(
        R._AVAILABLE_MODELS["bge-reranker-v2-m3"]["precisions"]["int8"],
        "dir",
        str(tmp_path),
    )
    assert R._resolve_reranker_model()[0] == R.INTENDED_RERANKER_MODEL


def test_resolver_honors_explicit_bge_override(monkeypatch):
    monkeypatch.setenv("CAIRN_RERANKER_MODEL", "bge-reranker-v2-m3")
    assert R._resolve_reranker_model()[0] == "bge-reranker-v2-m3"


# --------------------------------------------------------------------------
# Session-start hook must PRINT the warnings (the surface that was missing)
# --------------------------------------------------------------------------


def test_session_start_prints_model_warnings(monkeypatch):
    """The hook fetches status() (which carries warnings) but historically
    printed only the label. Guard that model warnings now reach stdout."""
    from cairn.hooks import session_start

    # The hook resolves welcome/status/_get_store from cairn.bridge at call
    # time (deferred imports), so patch them on the bridge module.
    import cairn.bridge as bridge

    monkeypatch.setattr(
        bridge, "welcome",
        lambda *a, **k: {"observation_prefix": "", "recent_memories": [], "memory_count": 1},
        raising=False,
    )
    monkeypatch.setattr(
        bridge, "status",
        lambda: {"ok": False, "status": "warning",
                 "warnings": ["⚠️ Reranker: running 'bge-reranker-v2-m3', not the intended "
                              "'ms-marco-MiniLM-L-6-v2'."]},
        raising=False,
    )

    class _FakeStore:
        def edge_count(self):
            return 0

        def get_last_capture_time(self):
            return None

        def count(self):
            return 1

    monkeypatch.setattr(bridge, "_get_store", lambda: _FakeStore(), raising=False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            session_start.main()
        except SystemExit:
            pass
    out = buf.getvalue()
    assert "not the intended" in out, out
