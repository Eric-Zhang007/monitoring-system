from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops_artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from inference.feature_reader import fetch_sequence
from models.liquid_model import build_liquid_model_from_checkpoint
from training.cache.panel_cache import build_multi_tf_vector_from_payload, compute_regime_features, normalize_context_timeframes


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


def _load_multi_tf_context(
    *,
    db_url: str,
    symbol: str,
    as_of: datetime,
    primary_timeframe: str,
    timeframes: list[str],
) -> Dict[str, object]:
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT as_of_ts, context_json, coverage_json
                FROM market_context_multi_tf
                WHERE UPPER(symbol) = %s
                  AND primary_timeframe = %s
                  AND as_of_ts <= %s
                ORDER BY as_of_ts DESC
                LIMIT 1
                """,
                (str(symbol).upper(), str(primary_timeframe).lower(), as_of),
            )
            row = cur.fetchone()
    if not row:
        raise RuntimeError(f"multi_tf_context_row_missing:{symbol}:{primary_timeframe}")
    return {
        "source": "market_context_multi_tf",
        "context": dict(row.get("context_json") or {}),
        "coverage": dict(row.get("coverage_json") or {}),
    }


def _predict(model, ckpt, values, mask, *, symbol: str, db_url: str, end_ts: datetime) -> Dict[str, object]:
    xv = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
    xm = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    symbol_to_id = ckpt.get("symbol_to_id") if isinstance(ckpt.get("symbol_to_id"), dict) else {}
    if symbol_to_id:
        if symbol not in symbol_to_id:
            raise RuntimeError(f"symbol_not_in_universe_snapshot:{symbol}")
        sid = int(symbol_to_id[symbol])
    else:
        sid = 0
    regime_np, regime_mask_np = compute_regime_features(np.asarray(values, dtype=np.float32), np.asarray(mask, dtype=np.uint8))
    regime_vec = np.asarray(regime_np[-1], dtype=np.float32)
    regime_msk = np.asarray(regime_mask_np[-1], dtype=np.uint8)
    expected_regime_dim = int(ckpt.get("regime_dim") or regime_vec.shape[0])
    if expected_regime_dim > regime_vec.shape[0]:
        context_timeframes = normalize_context_timeframes(
            ckpt.get("context_timeframes") or os.getenv("ANALYST_CONTEXT_TIMEFRAMES", "5m,15m,1h,4h,1d")
        )
        primary_tf = str(os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m")).strip().lower() or "5m"
        payload = _load_multi_tf_context(
            db_url=db_url,
            symbol=symbol,
            as_of=end_ts,
            primary_timeframe=primary_tf,
            timeframes=context_timeframes,
        )
        mtf_vec, mtf_msk = build_multi_tf_vector_from_payload(payload, timeframes=context_timeframes, require_complete=True)
        regime_vec = np.concatenate([regime_vec, mtf_vec], axis=0).astype(np.float32)
        regime_msk = np.concatenate([regime_msk, mtf_msk], axis=0).astype(np.uint8)
    if regime_vec.shape[0] != expected_regime_dim:
        raise RuntimeError(f"regime_dim_mismatch_inference:{regime_vec.shape[0]}:{expected_regime_dim}")
    regime_last = torch.tensor(regime_vec, dtype=torch.float32).unsqueeze(0)
    regime_mask_last = torch.tensor(regime_msk.astype(np.float32), dtype=torch.float32).unsqueeze(0)
    sid_t = torch.tensor([sid], dtype=torch.long)
    with torch.no_grad():
        out = model(
            xv,
            xm,
            symbol_id=sid_t,
            regime_features=regime_last,
            regime_mask=regime_mask_last,
        )
    mu = out.mu.squeeze(0).cpu().numpy()
    sigma = torch.exp(out.log_sigma).squeeze(0).cpu().numpy()
    q = out.q.squeeze(0).cpu().numpy() if out.q is not None else None
    dlogit = out.direction_logit.squeeze(0).cpu().numpy() if out.direction_logit is not None else None
    expert_w = out.expert_weights.squeeze(0).cpu().numpy() if out.expert_weights is not None else None
    regime_p = out.regime_probs.squeeze(0).cpu().numpy() if out.regime_probs is not None else None

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
    if expert_w is not None:
        names = ["trend", "mean_reversion", "liquidation_risk", "neutral"]
        vec = expert_w if expert_w.ndim == 1 else np.mean(expert_w, axis=0)
        s = float(np.sum(vec))
        if s > 0:
            vec = vec / s
        payload["expert_weights"] = {names[i]: float(vec[i]) for i in range(min(len(names), vec.shape[0]))}
    if regime_p is not None:
        names = ["trend", "crowding", "liquidation"]
        vec = regime_p if regime_p.ndim == 1 else np.mean(regime_p, axis=0)
        s = float(np.sum(vec))
        if s > 0:
            vec = vec / s
        payload["regime_probs"] = {names[i]: float(vec[i]) for i in range(min(len(names), vec.shape[0]))}
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

    pred = _predict(
        model,
        ckpt,
        seq["values"],
        seq["mask"],
        symbol=str(seq["symbol"]),
        db_url=str(args.database_url),
        end_ts=end_ts,
    )

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
