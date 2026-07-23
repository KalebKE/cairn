#!/usr/bin/env python3
"""Fetch candidate ONNX embedding models for the benchmark sweep.

Downloads a model's ONNX export into ~/.cache/cairn/models/<name>-onnx/
(NON-default directory names, so nothing auto-flips the live install), writes
a cairn.json sidecar describing pooling / output / prefixes / dims, and
validates the result before it can be trusted:

- sanity (always): dim matches, embeddings are finite/normalized, and a
  similar sentence pair scores well above a dissimilar one. Catches
  catastrophic pooling/output/tokenizer mistakes.
- reference (if sentence-transformers is importable): encode reference texts
  with the upstream model and require cosine > 0.999 against the ONNX path.
  This is the real guard; run it from any venv that has sentence-transformers
  (it is deliberately NOT a cairn dependency — PyTorch stays out of cairn).

Usage:
    python scripts/fetch_embedder.py arctic-embed-s
    python scripts/fetch_embedder.py gte-modernbert-base
    HF_TOKEN=... python scripts/fetch_embedder.py embeddinggemma-300m
    python scripts/fetch_embedder.py embeddinggemma-300m --truncate 512
    python scripts/fetch_embedder.py --list

--truncate creates a sibling <name>-<dim>-onnx dir that symlinks the model
files and carries a truncate_dim sidecar (Matryoshka arms without duplicating
gigabytes).
"""

import argparse
import json
import os
import sys
from pathlib import Path

MODELS_ROOT = Path(os.path.expanduser("~/.cache/cairn/models"))

# Registry: everything the sidecar-driven loader needs, per model.
# pooling: "cls" | "mean" | "model" (model = export ships a pooled output).
# output_name: select the ONNX output by name (None = legacy index heuristic).
REGISTRY = {
    "arctic-embed-s": {
        "repo_id": "Snowflake/snowflake-arctic-embed-s",
        "files": [
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
        ],
        "st_reference": "Snowflake/snowflake-arctic-embed-s",
        "sidecar": {
            "model_name": "arctic-embed-s",
            "model_version": "v1",
            "dim": 384,
            "pooling": "cls",
            "output_name": None,
            "query_prefix": "Represent this sentence for searching relevant passages: ",
            "doc_prefix": "",
            "max_length": 512,
            "pad_id": 0,
            "pad_token": "[PAD]",
        },
    },
    "gte-modernbert-base": {
        "repo_id": "Alibaba-NLP/gte-modernbert-base",
        "files": [
            ("onnx/model.onnx", "model.onnx"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
        ],
        "st_reference": "Alibaba-NLP/gte-modernbert-base",
        "sidecar": {
            "model_name": "gte-modernbert-base",
            "model_version": "v1",
            "dim": 768,
            "pooling": "cls",
            "output_name": None,
            "query_prefix": "",
            "doc_prefix": "",
            "max_length": 512,
            # ModernBERT pad token — verified against tokenizer_config at
            # fetch time (see _resolve_pad below).
            "pad_id": None,
            "pad_token": None,
        },
    },
    "embeddinggemma-300m": {
        "repo_id": "onnx-community/embeddinggemma-300m-ONNX",
        "files": [
            ("onnx/model.onnx", "model.onnx"),
            ("onnx/model.onnx_data", "model.onnx_data"),
            ("tokenizer.json", "tokenizer.json"),
            ("config.json", "config.json"),
            ("tokenizer_config.json", "tokenizer_config.json"),
        ],
        "st_reference": "google/embeddinggemma-300m",
        "sidecar": {
            "model_name": "embeddinggemma-300m",
            "model_version": "v1",
            "dim": 768,
            "pooling": "model",
            "output_name": "sentence_embedding",
            # EmbeddingGemma's documented retrieval prompts.
            "query_prefix": "task: search result | query: ",
            "doc_prefix": "title: none | text: ",
            "max_length": 512,
            "pad_id": None,
            "pad_token": None,
        },
    },
}

SANITY_TEXTS = [
    "The cat sat quietly on the warm windowsill.",
    "A kitten was resting near the sunny window.",
    "Quarterly revenue exceeded analyst expectations this fiscal year.",
]


def _resolve_pad(dest: Path, sidecar: dict) -> None:
    """Fill pad_id/pad_token from tokenizer_config.json when not pinned."""
    if sidecar.get("pad_id") is not None and sidecar.get("pad_token"):
        return
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(dest / "tokenizer.json"))
        cfg = json.loads((dest / "tokenizer_config.json").read_text())
        pad_token = cfg.get("pad_token")
        if isinstance(pad_token, dict):
            pad_token = pad_token.get("content")
        pad_token = pad_token or "[PAD]"
        pad_id = tok.token_to_id(pad_token)
        if pad_id is None:
            raise ValueError(f"pad token {pad_token!r} not in vocab")
        sidecar["pad_token"] = pad_token
        sidecar["pad_id"] = pad_id
        print(f"  pad token resolved: {pad_token!r} (id={pad_id})")
    except Exception as e:
        print(f"ERROR: could not resolve pad token: {e}")
        sys.exit(1)


def _encode_with_cairn(model_dir: Path, texts, mode="document"):
    """Encode via cairn's own loader in a pristine state, honoring the sidecar."""
    os.environ["CAIRN_ONNX_MODEL_DIR"] = str(model_dir)
    import cairn.embedding as E

    E.reset_embedding_state()
    vecs = [E.generate_embedding(t, mode=mode) for t in texts]
    if E.is_embedding_degraded() or E.get_active_backend() != "onnx":
        print(f"ERROR: model did not load via ONNX (backend={E.get_active_backend()})")
        sys.exit(1)
    return vecs


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def validate(model_dir: Path, sidecar: dict) -> None:
    """Sanity always; reference check when sentence-transformers is available."""
    expected_dim = sidecar.get("truncate_dim") or sidecar["dim"]
    vecs = _encode_with_cairn(model_dir, SANITY_TEXTS)

    for v in vecs:
        if len(v) != expected_dim:
            print(f"ERROR: dim {len(v)} != expected {expected_dim}")
            sys.exit(1)
        norm = sum(x * x for x in v) ** 0.5
        if not (0.99 < norm < 1.01):
            print(f"ERROR: embedding not normalized (|v|={norm:.4f})")
            sys.exit(1)

    sim_close = _cos(vecs[0], vecs[1])
    sim_far = _cos(vecs[0], vecs[2])
    print(f"  sanity: sim(similar)={sim_close:.3f} sim(dissimilar)={sim_far:.3f}")
    if sim_close - sim_far < 0.15:
        print("ERROR: similar pair does not separate from dissimilar pair — "
              "pooling/output config is likely wrong for this export.")
        sys.exit(1)

    ref_repo = None
    for name, spec in REGISTRY.items():
        if spec["sidecar"]["model_name"] == sidecar["model_name"]:
            ref_repo = spec["st_reference"]
    try:
        from sentence_transformers import SentenceTransformer  # optional, heavy
    except ImportError:
        print("  reference check SKIPPED (sentence-transformers not installed) — "
              "run this script once from a venv that has it before trusting results.")
        return

    st = SentenceTransformer(ref_repo, trust_remote_code=True)
    prompts = {"query": sidecar.get("query_prefix", ""), "document": sidecar.get("doc_prefix", "")}
    for mode in ("document", "query"):
        ours = _encode_with_cairn(model_dir, SANITY_TEXTS, mode=mode)
        theirs = st.encode(
            [prompts[mode] + t for t in SANITY_TEXTS], normalize_embeddings=True
        )
        for i, (a, b) in enumerate(zip(ours, theirs)):
            b = list(map(float, b))[: len(a)]
            nb = sum(x * x for x in b) ** 0.5
            b = [x / nb for x in b]  # renormalize after any truncation
            c = _cos(a, b)
            if c < 0.999:
                print(f"ERROR: reference mismatch ({mode}, text {i}): cosine={c:.5f}")
                sys.exit(1)
    print("  reference check PASSED (cosine > 0.999 vs sentence-transformers)")


def fetch(name: str, truncate: int | None) -> None:
    spec = REGISTRY[name]
    dest = MODELS_ROOT / f"{name}-onnx"

    if truncate:
        base = dest
        if not (base / "model.onnx").exists():
            print(f"ERROR: fetch {name} without --truncate first ({base} missing)")
            sys.exit(1)
        variant = MODELS_ROOT / f"{name}-{truncate}-onnx"
        variant.mkdir(parents=True, exist_ok=True)
        for f in base.iterdir():
            if f.name == "cairn.json":
                continue
            link = variant / f.name
            if not link.exists():
                link.symlink_to(f)
        sidecar = dict(json.loads((base / "cairn.json").read_text()))
        sidecar["truncate_dim"] = truncate
        sidecar["model_name"] = f"{name}-{truncate}"
        (variant / "cairn.json").write_text(json.dumps(sidecar, indent=1))
        print(f"Variant dir: {variant}")
        validate(variant, sidecar)
        print("OK")
        return

    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for remote, local in spec["files"]:
        target = dest / local
        if target.exists():
            print(f"  {local}: already present")
            continue
        print(f"  downloading {spec['repo_id']}/{remote} ...")
        try:
            got = hf_hub_download(repo_id=spec["repo_id"], filename=remote, token=token)
        except Exception as e:
            if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
                print(f"ERROR: {spec['repo_id']} looks gated. Accept the license on "
                      f"huggingface.co and set HF_TOKEN, then retry.")
            else:
                print(f"ERROR downloading {remote}: {e}")
            sys.exit(1)
        # hf_hub_download returns a cache path; hard-link or copy into place.
        import shutil

        shutil.copyfile(got, target)

    sidecar = dict(spec["sidecar"])
    _resolve_pad(dest, sidecar)
    (dest / "cairn.json").write_text(json.dumps(sidecar, indent=1))
    print(f"Model dir: {dest}")
    validate(dest, sidecar)
    print("OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", nargs="?", choices=sorted(REGISTRY))
    ap.add_argument("--truncate", type=int, default=None,
                    help="create a Matryoshka-truncated variant dir (e.g. 512)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.model:
        for n, s in REGISTRY.items():
            print(f"{n}: {s['repo_id']} dim={s['sidecar']['dim']} pooling={s['sidecar']['pooling']}")
        sys.exit(0)
    fetch(args.model, args.truncate)
