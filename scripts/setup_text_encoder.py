#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_MODEL_ID = "intfloat/multilingual-e5-small"
DEFAULT_OUT_DIR = "artifacts/models/text_encoder/multilingual-e5-small"


def main() -> int:
    ap = argparse.ArgumentParser(description="Download and cache text embedding model locally")
    ap.add_argument("--model-id", default=os.getenv("TEXT_EMBED_MODEL_ID", DEFAULT_MODEL_ID))
    ap.add_argument("--out-dir", default=os.getenv("TEXT_EMBED_MODEL_PATH", DEFAULT_OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    model_id = str(args.model_id or "").strip()
    out_dir = Path(str(args.out_dir or "").strip())
    if not model_id:
        raise RuntimeError("text_embed_model_id_required")
    if not str(out_dir).strip():
        raise RuntimeError("text_embed_out_dir_required")

    if out_dir.exists() and (not bool(args.force)):
        print(
            json.dumps(
                {
                    "status": "ok",
                    "message": "model_dir_exists_skip_download",
                    "model_id": model_id,
                    "model_path": str(out_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0

    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"sentence_transformers_unavailable:{exc}") from exc

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_id)
    model.save(str(out_dir))

    print(
        json.dumps(
            {
                "status": "ok",
                "model_id": model_id,
                "model_path": str(out_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
