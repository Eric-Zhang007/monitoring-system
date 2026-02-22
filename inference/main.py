from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from models.liquid_model import build_liquid_model_from_checkpoint


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_model(model_dir: Path):
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

    model = build_liquid_model_from_checkpoint(ckpt)
    model.eval()
    return model, manifest, ckpt, lookback


def _predict(model, ckpt, values, mask) -> Dict[str, object]:
    xv = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    xm = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = model(xv, xm)
    mu = out.mu.squeeze(0).cpu().numpy()
    sigma = torch.exp(out.log_sigma).squeeze(0).cpu().numpy()
    q = out.q.squeeze(0).cpu().numpy() if out.q is not None else None
    dlogit = out.direction_logit.squeeze(0).cpu().numpy() if out.direction_logit is not None else None

    cal = ckpt.get("calibration") if isinstance(ckpt.get("calibration"), dict) else {}
    sigma_scale = float(cal.get("sigma_scale", 1.0) or 1.0)
    direction_temperature = float(cal.get("direction_temperature", 1.0) or 1.0)

    sigma_cal = np.clip(sigma * sigma_scale, 1e-6, None)
    if dlogit is not None:
        probs = 1.0 / (1.0 + np.exp(-np.clip(dlogit / max(1e-6, direction_temperature), -40.0, 40.0)))
    else:
        probs = 1.0 / (1.0 + np.exp(-np.clip(mu / np.clip(sigma_cal, 1e-6, None), -40.0, 40.0)))

    horizons = [str(h) for h in list(ckpt.get("horizons") or ["1h", "4h", "1d", "7d"])]
    payload: Dict[str, object] = {
        "mu": {h: float(mu[i]) for i, h in enumerate(horizons)},
        "sigma": {h: float(sigma_cal[i]) for i, h in enumerate(horizons)},
        "confidence": {h: float(np.clip(probs[i], 0.01, 0.99)) for i, h in enumerate(horizons)},
    }
    if q is not None:
        payload["quantiles"] = {
            h: {"p10": float(q[i, 0]), "p50": float(q[i, 1]), "p90": float(q[i, 2])}
            for i, h in enumerate(horizons)
        }
    if dlogit is not None:
        payload["direction_logit"] = {h: float(dlogit[i]) for i, h in enumerate(horizons)}
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="Strict liquid inference with sequence parity")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--model-dir", default=os.getenv("LIQUID_MODEL_DIR", "artifacts/models/liquid_main"))
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--end-ts", default="")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    model, manifest, ckpt, lookback = _load_model(model_dir)

    end_ts = _parse_ts(args.end_ts) if str(args.end_ts).strip() else datetime.now(timezone.utc)
    seq = fetch_sequence(
        db_url=str(args.database_url),
        symbol=str(args.symbol).upper(),
        end_ts=end_ts,
        lookback=lookback,
    )
    if str(seq["schema_hash"]) != SCHEMA_HASH:
        raise RuntimeError("schema_hash_mismatch_sequence")

    pred = _predict(model, ckpt, seq["values"], seq["mask"])

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
