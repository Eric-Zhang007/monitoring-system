#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops_artifacts.validate import validate_manifest_dir
from models.liquid_model import build_liquid_model_from_checkpoint
from training.datasets.liquid_panel_cache_dataset import LiquidPanelCacheDataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Run inference smoke using packed artifact and cache sample")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-json", default="artifacts/smoke/inference_smoke.json")
    args = ap.parse_args()

    manifest = validate_manifest_dir(Path(str(args.artifact_dir)))
    ckpt = torch.load(Path(str(args.artifact_dir)) / str(manifest["files"]["weights"]), map_location="cpu")
    model = build_liquid_model_from_checkpoint(ckpt)
    model.eval()

    ds = LiquidPanelCacheDataset(cache_dir=args.cache_dir, lookback=int(ckpt.get("lookback") or 96), horizons=("1h", "4h", "1d", "7d"))
    sample = ds[0]
    with torch.no_grad():
        out = model(
            sample["x_values"].unsqueeze(0),
            sample["x_mask"].unsqueeze(0),
            symbol_id=sample["symbol_id"].unsqueeze(0),
            regime_features=sample["regime_features"].unsqueeze(0),
            regime_mask=sample["regime_mask"].unsqueeze(0),
        )
    horizons = [str(h) for h in (ckpt.get("horizons") or ["1h", "4h", "1d", "7d"])]
    mu = out.mu.squeeze(0).cpu().numpy()
    sigma = torch.exp(out.log_sigma).squeeze(0).cpu().numpy()
    q = out.q.squeeze(0).cpu().numpy() if out.q is not None else None
    dprob = torch.sigmoid(out.direction_logit).squeeze(0).cpu().numpy() if out.direction_logit is not None else None
    ew = out.expert_weights.squeeze(0).cpu().numpy() if out.expert_weights is not None else None
    rp = out.regime_probs.squeeze(0).cpu().numpy() if out.regime_probs is not None else None

    payload = {
        "status": "ok",
        "symbol": sample["symbol"],
        "end_ts": int(sample["end_ts"]),
        "model_id": manifest.get("model_id"),
        "predictions": {
            "mu": {h: float(mu[i]) for i, h in enumerate(horizons)},
            "sigma": {h: float(sigma[i]) for i, h in enumerate(horizons)},
            "quantiles": {
                h: {"p10": float(q[i, 0]), "p50": float(q[i, 1]), "p90": float(q[i, -1])}
                for i, h in enumerate(horizons)
            }
            if q is not None
            else {},
            "direction_prob": {h: float(dprob[i]) for i, h in enumerate(horizons)} if dprob is not None else {},
            "expert_weights": {h: {"trend": float(ew[0]), "mean_reversion": float(ew[1]), "liquidation_risk": float(ew[2]), "neutral": float(ew[3])} for h in horizons}
            if ew is not None and ew.shape[0] >= 4
            else {},
            "regime_probs": {h: {"trend": float(rp[0]), "crowding": float(rp[1]), "liquidation": float(rp[2])} for h in horizons}
            if rp is not None and rp.shape[0] >= 3
            else {},
        },
    }
    out_path = Path(str(args.out_json))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "out_json": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
