from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class TextEmbedderError(RuntimeError):
    pass


def load_model(model_path: str):
    path = Path(str(model_path or "").strip())
    if not path.exists():
        raise TextEmbedderError(f"embedding_model_missing:{path}")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"sentence_transformers_unavailable:{exc}") from exc
    try:
        return SentenceTransformer(str(path))
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"embedding_model_load_failed:{exc}") from exc


def encode_texts(model, texts: Iterable[str]) -> np.ndarray:
    seq = [str(x or "").strip() for x in texts]
    if not seq:
        return np.zeros((0, 0), dtype=np.float32)
    try:
        emb = model.encode(seq, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"embedding_encode_failed:{exc}") from exc
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise TextEmbedderError(f"embedding_invalid_shape:{arr.shape}")
    return arr
