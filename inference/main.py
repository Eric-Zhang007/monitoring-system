from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from models.multimodal_gate import ResidualGateHead
from models.patchtst import PatchTSTBackbone


if torch is not None:
    class LiquidSequenceModel(torch.nn.Module):
        def __init__(self, *, lookback: int, feature_dim: int, d_model: int, n_layers: int, dropout: float, text_indices, quality_indices):
            super().__init__()
            self.backbone = PatchTSTBackbone(
                feature_dim=feature_dim,
                lookback=lookback,
                d_model=d_model,
                n_layers=n_layers,
                dropout=dropout,
            )
            self.text_indices = list(text_indices)
            self.quality_indices = list(quality_indices)
            self.head = ResidualGateHead(hidden_dim=d_model, text_dim=len(self.text_indices), quality_dim=len(self.quality_indices), out_dim=4)

        def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor):
            h = self.backbone(x_values, x_mask)
            last_values = x_values[:, -1, :]
            text_vec = last_values[:, self.text_indices]
            quality_vec = last_values[:, self.quality_indices]
            return self.head(h, text_vec, quality_vec)
else:  # pragma: no cover
    class LiquidSequenceModel:  # type: ignore[no-redef]
        pass


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_model(model_dir: Path):
    if torch is None:
        raise RuntimeError("torch_required_for_inference")
    manifest = validate_manifest_dir(model_dir, expected_schema_hash=SCHEMA_HASH)
    weights_path = model_dir / manifest["files"]["weights"]
    ckpt = torch.load(weights_path, map_location="cpu")

    if str(ckpt.get("schema_hash") or "") != SCHEMA_HASH:
        raise RuntimeError("schema_hash_mismatch_weights")
    lookback = int(ckpt.get("lookback") or 0)
    if lookback <= 0:
        raise RuntimeError("invalid_lookback_in_weights")
    feature_dim = int(ckpt.get("feature_dim") or 0)
    if feature_dim != FEATURE_DIM:
        raise RuntimeError(f"feature_dim_mismatch_weights:{feature_dim}:{FEATURE_DIM}")

    model = LiquidSequenceModel(
        lookback=lookback,
        feature_dim=feature_dim,
        d_model=int(ckpt.get("d_model") or 128),
        n_layers=int(ckpt.get("n_layers") or 2),
        dropout=float(ckpt.get("dropout") or 0.1),
        text_indices=list(ckpt.get("text_indices") or []),
        quality_indices=list(ckpt.get("quality_indices") or []),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, manifest, lookback


def _predict(model, values, mask) -> Dict[str, float]:
    xv = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    xm = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred, gate = model(xv, xm)
    p = pred.squeeze(0).cpu().numpy().tolist()
    g = float(gate.squeeze(0).item())
    return {
        "ret_1h": float(p[0]),
        "ret_4h": float(p[1]),
        "ret_1d": float(p[2]),
        "ret_7d": float(p[3]),
        "gate": g,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict liquid inference with sequence parity")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--model-dir", default=os.getenv("LIQUID_MODEL_DIR", "artifacts/models/liquid_main"))
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--end-ts", default="")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model, manifest, lookback = _load_model(model_dir)

    end_ts = _parse_ts(args.end_ts) if str(args.end_ts).strip() else datetime.now(timezone.utc)
    seq = fetch_sequence(
        db_url=str(args.database_url),
        symbol=str(args.symbol).upper(),
        end_ts=end_ts,
        lookback=lookback,
    )
    if str(seq["schema_hash"]) != SCHEMA_HASH:
        raise RuntimeError("schema_hash_mismatch_sequence")

    pred = _predict(model, seq["values"], seq["mask"])

    out = {
        "status": "ok",
        "model_id": manifest["model_id"],
        "schema_hash": SCHEMA_HASH,
        "symbol": seq["symbol"],
        "lookback": lookback,
        "coverage_summary": seq["coverage_summary"],
        "predictions": pred,
        "end_ts": seq["end_ts"],
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
