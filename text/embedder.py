from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Iterable

import numpy as np


class TextEmbedderError(RuntimeError):
    pass


def _load_sentence_transformers(path: Path):
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


def _load_onnx(path: Path):
    try:
        import onnxruntime as ort  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"onnxruntime_unavailable:{exc}") from exc
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"tokenizers_unavailable:{exc}") from exc

    onnx_dir = path / "onnx" if (path / "onnx").is_dir() else path
    model_file = None
    for cand in ("model.onnx", "model_O4.onnx", "model_qint8_avx512_vnni.onnx"):
        p = onnx_dir / cand
        if p.exists():
            model_file = p
            break
    if model_file is None:
        raise TextEmbedderError(f"onnx_model_missing:{onnx_dir}")
    tok_file = onnx_dir / "tokenizer.json"
    if not tok_file.exists():
        raise TextEmbedderError(f"onnx_tokenizer_missing:{tok_file}")

    tok = Tokenizer.from_file(str(tok_file))
    max_len = 512
    cfg_file = onnx_dir / "config.json"
    if cfg_file.exists():
        try:
            cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
            max_len = int(cfg.get("max_position_embeddings") or 512)
        except Exception:
            max_len = 512
    if max_len <= 0:
        max_len = 512
    try:
        tok.enable_truncation(max_length=max_len)
    except Exception:
        # Keep explicit clipping in encode_texts as a second guard.
        pass
    sess = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    output_names = [o.name for o in sess.get_outputs()]
    if not input_names:
        raise TextEmbedderError("onnx_inputs_empty")
    if not output_names:
        raise TextEmbedderError("onnx_outputs_empty")
    return {
        "backend": "onnx",
        "tokenizer": tok,
        "session": sess,
        "input_names": input_names,
        "output_names": output_names,
        "max_length": int(max_len),
    }


def load_model(model_path: str):
    path = Path(str(model_path or "").strip())
    if not path.exists():
        raise TextEmbedderError(f"embedding_model_missing:{path}")
    backend = str(os.getenv("TEXT_EMBED_BACKEND", "onnx")).strip().lower()
    if backend in {"onnx", "ort"}:
        return _load_onnx(path)
    if backend in {"sentence_transformers", "st"}:
        return _load_sentence_transformers(path)
    raise TextEmbedderError(f"unsupported_text_embed_backend:{backend}")


def encode_texts(model, texts: Iterable[str]) -> np.ndarray:
    seq = [str(x or "").strip() for x in texts]
    if not seq:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(model, dict) and str(model.get("backend")) == "onnx":
        tok = model.get("tokenizer")
        sess = model.get("session")
        input_names = list(model.get("input_names") or [])
        output_names = list(model.get("output_names") or [])
        model_max_len = int(model.get("max_length") or 512)
        if model_max_len <= 0:
            model_max_len = 512
        if tok is None or sess is None:
            raise TextEmbedderError("onnx_model_invalid")
        enc = tok.encode_batch(seq)
        max_len = min(model_max_len, max((len(e.ids) for e in enc), default=1))
        bsz = len(enc)
        input_ids = np.zeros((bsz, max_len), dtype=np.int64)
        attention_mask = np.zeros((bsz, max_len), dtype=np.int64)
        token_type_ids = np.zeros((bsz, max_len), dtype=np.int64)
        for i, e in enumerate(enc):
            n = min(max_len, len(e.ids))
            if n <= 0:
                continue
            input_ids[i, :n] = np.asarray(e.ids[:n], dtype=np.int64)
            attention_mask[i, :n] = np.asarray(e.attention_mask[:n], dtype=np.int64)
            if hasattr(e, "type_ids") and e.type_ids is not None:
                token_type_ids[i, :n] = np.asarray(e.type_ids[:n], dtype=np.int64)
        feed = {}
        for n in input_names:
            low = str(n).lower()
            if "input" in low and "id" in low:
                feed[n] = input_ids
            elif "attention" in low and "mask" in low:
                feed[n] = attention_mask
            elif "token_type" in low:
                feed[n] = token_type_ids
            else:
                # Conservative default for unknown required input tensors.
                feed[n] = input_ids
        out = sess.run(output_names, feed)
        if not out:
            raise TextEmbedderError("onnx_output_empty")
        hidden = np.asarray(out[0], dtype=np.float32)
        if hidden.ndim != 3:
            raise TextEmbedderError(f"onnx_hidden_invalid_shape:{hidden.shape}")
        mask = attention_mask.astype(np.float32)[..., None]
        summed = np.sum(hidden * mask, axis=1)
        denom = np.clip(np.sum(mask, axis=1), 1e-6, None)
        arr = summed / denom
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norm, 1e-12, None)
        if arr.ndim != 2:
            raise TextEmbedderError(f"embedding_invalid_shape:{arr.shape}")
        return np.asarray(arr, dtype=np.float32)
    try:
        emb = model.encode(seq, convert_to_numpy=True, normalize_embeddings=True)
    except Exception as exc:  # pragma: no cover
        raise TextEmbedderError(f"embedding_encode_failed:{exc}") from exc
    arr = np.array(emb, dtype=np.float32)
    if arr.ndim != 2:
        raise TextEmbedderError(f"embedding_invalid_shape:{arr.shape}")
    return arr
