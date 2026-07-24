"""Model-drift detection: is Cairn running the models it's supposed to?

Cairn once ran the wrong reranker for months (ms-marco instead of the assumed
bge-reranker-v2-m3) because model resolution is a silent disk-probe, the
substitution was logged at INFO, and the only visible log sink is WARNING. The
fix is structural: declare an INTENDED model for the embedder and reranker,
compare the resolved model against it, and surface any divergence LOUDLY where
a human/agent actually looks — the session-start briefing, `cairn status`, and
`cairn doctor` all consume `model_health_warnings()`.

This module must stay cheap: it resolves models (disk probes, memoized, no
network) but never LOADS one. It lives at the top level (below `bridge`) so the
store layer's `check_memory_health` can call it without an import cycle.
"""

from __future__ import annotations

import os
from typing import List


def model_health_warnings() -> List[str]:
    """Return human-readable warnings when the running models are not the
    intended ones (or are degraded). Empty list means all good.

    Each warning names what's running, what was intended, and the exact fix.
    Never raises: every probe is guarded so a health check can't break a
    session start.
    """
    warnings: List[str] = []
    _check_embedder(warnings)
    _check_reranker(warnings)
    return warnings


def _check_embedder(warnings: List[str]) -> None:
    try:
        from cairn import embedding as E

        # Resolve first (mutates _EMBEDDING_MODEL_NAME to the resolved model on
        # fallback), then read identity. Cheap disk probe, no model load.
        E._get_onnx_model_dir()
        info = E.get_embedding_model_info()
        resolved = info.get("model_name")

        if E.is_embedding_degraded():
            warnings.append(
                "⚠️ Embeddings are DEGRADED to a hash fallback — semantic search "
                "is broken (stored vectors won't match query vectors). The ONNX "
                "model failed to load. Fix: 'cairn setup --download-model'."
            )
            return  # degraded dominates; don't also warn about which model

        # An explicit CAIRN_ONNX_MODEL_DIR override is an intentional choice —
        # don't nag about it.
        if os.environ.get("CAIRN_ONNX_MODEL_DIR"):
            return

        if resolved and resolved != E.INTENDED_EMBEDDING_MODEL:
            warnings.append(
                f"⚠️ Embedding model: running '{resolved}', not the intended "
                f"'{E.INTENDED_EMBEDDING_MODEL}'. Retrieval quality is reduced. "
                f"Fix: 'cairn setup --download-model'"
                + (
                    " then 'cairn migrate-embeddings'"
                    if resolved == "bge-small-en-v1.5"
                    else ""
                )
                + "."
            )
    except Exception:
        # A health probe must never break the caller.
        pass


def _check_reranker(warnings: List[str]) -> None:
    try:
        import cairn.reranker as R

        # Cross-encoder deliberately disabled → not a fault, say nothing.
        if os.environ.get("CAIRN_CROSS_ENCODER") == "0":
            return

        resolved_name, _ = R._resolve_reranker_model()

        # Explicit CAIRN_RERANKER_MODEL override is intentional — don't nag.
        if not os.environ.get("CAIRN_RERANKER_MODEL"):
            if resolved_name != R.INTENDED_RERANKER_MODEL:
                warnings.append(
                    f"⚠️ Reranker: running '{resolved_name}', not the intended "
                    f"'{R.INTENDED_RERANKER_MODEL}'. Fix: unset CAIRN_RERANKER_MODEL "
                    f"or run 'cairn setup --download-model'."
                )

        # Intended (or overridden) model chosen, but files not on disk yet.
        # Not an error — it downloads on first use — but worth flagging so a
        # failed/blocked download isn't silent.
        if R._get_model_dir() is None:
            warnings.append(
                f"ℹ️ Reranker '{resolved_name}' is not downloaded yet; it will "
                f"fetch on first use (or run 'cairn setup --download-model'). "
                f"Set CAIRN_RERANKER_AUTODOWNLOAD=0 to disable auto-download."
            )
    except Exception:
        pass
