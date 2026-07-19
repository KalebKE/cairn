#!/usr/bin/env python3
"""Convert bge-reranker-v2-m3 to ONNX for use as Cairn's cross-encoder reranker.

bge-reranker-v2-m3 is a general-purpose reranker (not web-search-specific like MS-MARCO),
which performs better on conversational memory queries.

Requires: pip install optimum[exporters] torch transformers
(These are NOT runtime deps — only needed once for conversion.)

Usage:
    python3 scripts/convert_bge_reranker.py
    # Then set: export CAIRN_RERANKER_MODEL=bge-reranker-v2-m3
    # Or restart Cairn — it auto-detects the model if present.
"""
import os
import subprocess
import sys
from pathlib import Path

MODEL_ID = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = Path(os.path.expanduser("~/.cache/cairn/models/bge-reranker-v2-m3-onnx"))


def convert():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if (OUTPUT_DIR / "model.onnx").exists() and (OUTPUT_DIR / "tokenizer.json").exists():
        print(f"Model already exists at {OUTPUT_DIR}")
        return True

    print(f"Converting {MODEL_ID} to ONNX...")
    print("This requires: pip install optimum[exporters] torch transformers")
    print()

    # Use optimum CLI for export
    result = subprocess.run(
        [
            sys.executable, "-m", "optimum.exporters.onnx",
            "--model", MODEL_ID,
            "--task", "text-classification",
            str(OUTPUT_DIR),
        ],
        capture_output=False,
        timeout=600,
    )

    if result.returncode != 0:
        print("\nOptimum export failed. Trying manual conversion...")
        return manual_convert()

    if (OUTPUT_DIR / "model.onnx").exists():
        print(f"\nModel converted to {OUTPUT_DIR}")
        print("Set CAIRN_RERANKER_MODEL=bge-reranker-v2-m3 or restart Cairn.")
        return True

    print("Export completed but model.onnx not found")
    return False


def manual_convert():
    """Fallback: manual PyTorch -> ONNX conversion."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        print("Manual conversion requires: pip install torch transformers")
        return False

    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    model.eval()

    # Save tokenizer in fast format
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Create dummy inputs
    dummy = tokenizer(
        "query",
        "passage to rerank",
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )

    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in dummy:
        input_names.append("token_type_ids")

    dynamic_axes = {name: {0: "batch", 1: "seq"} for name in input_names}
    dynamic_axes["logits"] = {0: "batch"}

    onnx_path = OUTPUT_DIR / "model.onnx"
    print(f"Exporting to {onnx_path}...")

    torch.onnx.export(
        model,
        tuple(dummy[name] for name in input_names),
        str(onnx_path),
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )

    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"\nModel converted: {onnx_path} ({size_mb:.1f} MB)")
        print("Set CAIRN_RERANKER_MODEL=bge-reranker-v2-m3 or restart Cairn.")
        return True

    print("Export failed — model.onnx not created")
    return False


if __name__ == "__main__":
    success = convert()
    sys.exit(0 if success else 1)
