"""Backward-compatibility shim — renamed to embedding.py in v0.11.0."""

from cairn.embedding import *  # noqa: F401,F403
from cairn.embedding import (  # noqa: F401
    generate_embedding,
    generate_embeddings_batch,
    generate_embedding_async,
    generate_embeddings_batch_async,
    preload_embedding_model,
    preload_embedding_model_async,
    get_embedding_model_info,
    get_embedding_info,
    get_active_backend,
    has_onnx_runtime,
    has_sentence_transformers,
    reset_embedding_state,
    is_embedding_degraded,
)
