"""Version-pinned, checksum-verified model manifest.

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

MODELS = {'gte-modernbert-base': {'dir': 'gte-modernbert-base-onnx',
                         'sidecar': 'gte',
                         'files': [{'name': 'model.onnx',
                                    'asset': 'gte-modernbert-base-onnx__model.onnx',
                                    'sha256': '947f31df7effaeec4edb57c50e4ed7e0f2034d9336063f92615b92e3e0d24d78',
                                    'size': 596392315,
                                    'hf_url': 'https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/onnx/model.onnx'},
                                   {'name': 'config.json',
                                    'asset': 'gte-modernbert-base-onnx__config.json',
                                    'sha256': '8ba54dc3d35d7194f5178a4194b649f146753e02dabd22bdca5c5cbac15069ed',
                                    'size': 1184,
                                    'hf_url': 'https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/config.json'},
                                   {'name': 'tokenizer_config.json',
                                    'asset': 'gte-modernbert-base-onnx__tokenizer_config.json',
                                    'sha256': '9654072f7c873161814043cf08cb5ed72f71d0b935abcd4e267935cb34352c21',
                                    'size': 20867,
                                    'hf_url': 'https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/tokenizer_config.json'},
                                   {'name': 'tokenizer.json',
                                    'asset': 'gte-modernbert-base-onnx__tokenizer.json',
                                    'sha256': '6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30',
                                    'size': 3583228,
                                    'hf_url': 'https://huggingface.co/Alibaba-NLP/gte-modernbert-base/resolve/main/tokenizer.json'}]},
 'ms-marco-MiniLM-L-6-v2': {'dir': 'ms-marco-MiniLM-L-6-v2-onnx',
                            'sidecar': None,
                            'files': [{'name': 'model.onnx',
                                       'asset': 'ms-marco-MiniLM-L-6-v2-onnx__model.onnx',
                                       'sha256': '5d3e70fd0c9ff14b9b5169a51e957b7a9c74897afd0a35ce4bd318150c1d4d4a',
                                       'size': 91011230,
                                       'hf_url': 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx'},
                                      {'name': 'config.json',
                                       'asset': 'ms-marco-MiniLM-L-6-v2-onnx__config.json',
                                       'sha256': '380e02c93f431831be65d99a4e7e5f67c133985bf2e77d9d4eba46847190bacc',
                                       'size': 794,
                                       'hf_url': 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/config.json'},
                                      {'name': 'tokenizer.json',
                                       'asset': 'ms-marco-MiniLM-L-6-v2-onnx__tokenizer.json',
                                       'sha256': 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66',
                                       'size': 711396,
                                       'hf_url': 'https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json'}]}}
