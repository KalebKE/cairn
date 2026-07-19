#!/usr/bin/env python3
"""Download the ONNX embedding model for Cairn."""
import os
import sys
import urllib.request
from pathlib import Path

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILES = [
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.txt",
]

HF_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main/onnx"

def download_model(output_dir: str):
    """Download ONNX model files to the specified directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (output_path / "model.onnx").exists():
        print(f"Model already exists at {output_path}")
        return

    print(f"Downloading model to {output_path}...")

    # Try optimum export first (produces optimized ONNX)
    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "optimum.exporters.onnx",
                "--model", MODEL_REPO,
                str(output_path),
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode == 0 and (output_path / "model.onnx").exists():
            print("Model exported via optimum")
            return
    except Exception:
        pass

    # Fallback: download individual files from HuggingFace
    for filename in ONNX_FILES:
        target = output_path / filename
        if target.exists():
            continue

        # Try onnx/ subdirectory first (where HF stores ONNX exports)
        url = f"{HF_BASE}/{filename}"
        try:
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(url, str(target))
        except Exception:
            # Try root of repo
            url_root = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{filename}"
            try:
                urllib.request.urlretrieve(url_root, str(target))
            except Exception as e:
                print(f"  Failed to download {filename}: {e}")

    if (output_path / "model.onnx").exists():
        print("Model download complete!")
    else:
        print("WARNING: model.onnx not found after download. You may need to export manually:")
        print("  pip install optimum[exporters]")
        print(f"  optimum-cli export onnx --model {MODEL_REPO} {output_path}")

if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.cache/cairn/models/all-MiniLM-L6-v2-onnx")
    download_model(output)
