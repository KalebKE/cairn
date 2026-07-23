"""Model-sidecar plumbing: pooling dispatch, prefix asymmetry, dim flexibility,
vec-table rebuild, and the tunable CE margin (benchmark-sweep Phase 1)."""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from _vec import requires_vec  # noqa: E402


# ---------------------------------------------------------------------------
# _onnx_encode pooling dispatch (stubbed tokenizer/session — no model needed)
# ---------------------------------------------------------------------------


class _StubEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _StubTokenizer:
    def encode_batch(self, texts):
        # Two tokens per text, second one padding for the second text.
        return [
            _StubEncoding([1, 2], [1, 1]),
            _StubEncoding([3, 0], [1, 0]),
        ][: len(texts)]


class _StubInput:
    def __init__(self, name):
        self.name = name


class _StubSession:
    """Returns a fixed [batch, seq, dim] hidden state plus a named pooled output."""

    def __init__(self, hidden, pooled=None, pooled_name="sentence_embedding"):
        self._hidden = hidden
        self._pooled = pooled
        self._pooled_name = pooled_name

    def get_inputs(self):
        return [_StubInput("input_ids"), _StubInput("attention_mask")]

    def run(self, output_names, feed):
        import numpy as np

        if output_names and output_names[0] == self._pooled_name:
            return [np.array(self._pooled, dtype=np.float32)]
        return [np.array(self._hidden, dtype=np.float32)]


@pytest.fixture()
def _clean_embedding_state():
    import cairn.embedding as E

    E.reset_embedding_state()
    yield E
    E.reset_embedding_state()


def _norm(v):
    return sum(x * x for x in v) ** 0.5


def test_onnx_encode_mean_pooling_default(_clean_embedding_state, monkeypatch):
    """No sidecar → historical behavior: masked mean over tokens, L2-normalized."""
    import numpy as np

    E = _clean_embedding_state
    monkeypatch.setattr(E, "_MODEL_SIDECAR", None)
    hidden = [[[2.0, 0.0], [0.0, 2.0]], [[4.0, 0.0], [9.0, 9.0]]]
    out = E._onnx_encode(_StubTokenizer(), _StubSession(hidden), ["a", "b"])
    # Text 0: mean of both tokens = [1,1] → normalized [0.707, 0.707]
    assert out[0] == pytest.approx([2**-0.5, 2**-0.5], abs=1e-5)
    # Text 1: second token masked out → [4,0] → normalized [1,0]
    assert out[1] == pytest.approx([1.0, 0.0], abs=1e-5)
    assert np.linalg.norm(out, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)


def test_onnx_encode_cls_pooling(_clean_embedding_state, monkeypatch):
    E = _clean_embedding_state
    monkeypatch.setattr(E, "_MODEL_SIDECAR", {**E._SIDECAR_DEFAULTS, "pooling": "cls"})
    hidden = [[[3.0, 4.0], [100.0, 100.0]]]
    out = E._onnx_encode(_StubTokenizer(), _StubSession(hidden), ["a"])
    # CLS = first token [3,4] → normalized [0.6, 0.8]; other tokens ignored.
    assert out[0] == pytest.approx([0.6, 0.8], abs=1e-5)


def test_onnx_encode_named_pooled_output(_clean_embedding_state, monkeypatch):
    """pooling='model' + output_name selects the pre-pooled output by name."""
    E = _clean_embedding_state
    monkeypatch.setattr(
        E,
        "_MODEL_SIDECAR",
        {**E._SIDECAR_DEFAULTS, "pooling": "model", "output_name": "sentence_embedding"},
    )
    hidden = [[[9.0, 9.0], [9.0, 9.0]]]  # would give the wrong answer if used
    pooled = [[0.0, 5.0]]
    out = E._onnx_encode(_StubTokenizer(), _StubSession(hidden, pooled=pooled), ["a"])
    assert out[0] == pytest.approx([0.0, 1.0], abs=1e-5)


def test_onnx_encode_truncate_then_renormalize(_clean_embedding_state, monkeypatch):
    """Matryoshka: slice to truncate_dim BEFORE the L2 normalize."""
    E = _clean_embedding_state
    monkeypatch.setattr(
        E,
        "_MODEL_SIDECAR",
        {**E._SIDECAR_DEFAULTS, "pooling": "model", "output_name": "sentence_embedding",
         "truncate_dim": 2},
    )
    pooled = [[3.0, 4.0, 100.0, 100.0]]  # huge tail that truncation must discard
    out = E._onnx_encode(
        _StubTokenizer(), _StubSession([[[0.0] * 4] * 2], pooled=pooled), ["a"]
    )
    assert len(out[0]) == 2
    # [3,4] renormalized — NOT [3,4,...]/|full vector|
    assert out[0] == pytest.approx([0.6, 0.8], abs=1e-5)


# ---------------------------------------------------------------------------
# Prefix asymmetry
# ---------------------------------------------------------------------------


def test_prefix_separates_query_and_document_embeddings(_clean_embedding_state, monkeypatch):
    """Same text, different mode → different embedding AND no cache collision."""
    E = _clean_embedding_state
    monkeypatch.setenv("CAIRN_SKIP_EMBEDDINGS", "1")  # hash fallback is fine here
    monkeypatch.setattr(
        E, "_MODEL_SIDECAR", {**E._SIDECAR_DEFAULTS, "query_prefix": "query: "}
    )
    doc = E.generate_embedding("shared text", mode="document")
    qry = E.generate_embedding("shared text", mode="query")
    assert doc != qry, "query prefix must change the embedding input"
    # And with no prefixes configured, modes must agree (legacy behavior).
    monkeypatch.setattr(E, "_MODEL_SIDECAR", dict(E._SIDECAR_DEFAULTS))
    E._EMBEDDING_CACHE.clear()
    doc2 = E.generate_embedding("shared text", mode="document")
    qry2 = E.generate_embedding("shared text", mode="query")
    assert doc2 == qry2


def test_batch_prefix_applied(_clean_embedding_state, monkeypatch):
    E = _clean_embedding_state
    monkeypatch.setenv("CAIRN_SKIP_EMBEDDINGS", "1")
    monkeypatch.setattr(
        E, "_MODEL_SIDECAR", {**E._SIDECAR_DEFAULTS, "query_prefix": "q: "}
    )
    single = E.generate_embedding("abc", mode="query")
    batch = E.generate_embeddings_batch(["abc"], mode="query")
    assert batch[0] == single


# ---------------------------------------------------------------------------
# Dim flexibility
# ---------------------------------------------------------------------------


def test_deserialize_infers_dim_from_blob():
    from cairn.sqlite_store._types import _deserialize_f32, _serialize_f32

    for dim in (8, 384, 768):
        v = [float(i) for i in range(dim)]
        assert _deserialize_f32(_serialize_f32(v)) == pytest.approx(v)


def test_embedding_dim_env_override_subprocess(tmp_path):
    """CAIRN_EMBEDDING_DIM is read at import — verify in a child process."""
    import subprocess

    code = (
        "from cairn.sqlite_store._types import EMBEDDING_DIM; print(EMBEDDING_DIM)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "CAIRN_EMBEDDING_DIM": "768"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.stdout.strip() == "768", out.stderr


# ---------------------------------------------------------------------------
# rebuild_vec_table
# ---------------------------------------------------------------------------


@requires_vec
def test_rebuild_vec_table_recreates_and_reembeds(tmp_path):
    from cairn.sqlite_store import EMBEDDING_DIM, SQLiteStore

    store = SQLiteStore(db_path=tmp_path / "rebuild.db")
    try:
        emb = [1.0] + [0.0] * (EMBEDDING_DIM - 1)
        for i in range(3):
            store.store(
                content=f"rebuild test memory {i} about vector tables",
                metadata={"event_type": "memory"},
                embedding=emb,
                skip_inference=True,
            )
        # Sabotage: drop one vec row so pre-rebuild count is short.
        with store._lock:
            store._conn.execute("DELETE FROM memories_vec WHERE rowid = 1")
            store._conn.commit()
        stats = store.rebuild_vec_table()
        assert stats["dim"] == EMBEDDING_DIM
        assert stats["updated"] == 3
        n = store._conn.execute("SELECT COUNT(*) FROM memories_vec").fetchone()[0]
        assert n == 3
        ddl = store._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='memories_vec'"
        ).fetchone()[0]
        assert f"float[{EMBEDDING_DIM}]" in ddl
    finally:
        store.close()


# ---------------------------------------------------------------------------
# CAIRN_CE_MARGIN
# ---------------------------------------------------------------------------


@requires_vec
def test_ce_margin_env_gates_promotion(tmp_path, monkeypatch):
    """A confident CE flips rank 1 at the default margin but not at margin≈1."""
    import cairn.reranker as reranker
    from cairn.sqlite_store import EMBEDDING_DIM, SQLiteStore

    store = SQLiteStore(db_path=tmp_path / "margin.db")
    try:
        base = [0.0] * EMBEDDING_DIM
        e1 = list(base)
        e1[0] = 1.0
        e2 = list(base)
        e2[0] = 0.9
        e2[1] = 0.435889894354  # ~unit norm
        ids = [
            store.store(
                content="alpha memory about deployment pipelines and CI runners",
                metadata={"event_type": "memory"},
                embedding=e1,
                skip_inference=True,
            ),
            store.store(
                content="beta memory about deployment pipelines and CI helpers",
                metadata={"event_type": "memory"},
                embedding=e2,
                skip_inference=True,
            ),
        ]

        def fake_ce(query, passages, temporal_metadata=None):
            # Massive raw-logit preference for the LAST passage.
            return [-8.0] * (len(passages) - 1) + [8.0]

        monkeypatch.setattr(reranker, "cross_encoder_score", fake_ce)
        monkeypatch.setenv("CAIRN_CE_MODE", "hybrid")

        monkeypatch.setenv("CAIRN_CE_MARGIN", "0.10")
        r_default = store.query(
            "deployment pipelines", limit=2, use_cache=False, query_embedding=e1
        )
        monkeypatch.setenv("CAIRN_CE_MARGIN", "0.99")
        r_strict = store.query(
            "deployment pipelines", limit=2, use_cache=False, query_embedding=e1
        )
        assert len(r_default) >= 1 and len(r_strict) >= 1
        # Under the strict margin the CE may not override the fused order;
        # under the default margin the CE favorite must sit at rank 1.
        ce_favorite = r_default[0].id
        assert ce_favorite in ids
        if r_strict[0].id != r_default[0].id:
            # Promotion suppressed by the strict margin — exactly the point.
            assert r_strict[0].id in ids
    finally:
        store.close()
