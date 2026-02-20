#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from inference.liquid_feature_contract import LIQUID_FEATURE_SCHEMA_VERSION, LIQUID_FEATURE_KEYS


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    d = X.shape[1]
    eye = np.eye(d, dtype=np.float64)
    w = np.linalg.solve(X.T @ X + l2 * eye, X.T @ y)
    return w.astype(np.float64)


def _infer_text_feature_indices(keys: List[str]) -> List[int]:
    out: List[int] = []
    for idx, key in enumerate(keys):
        k = str(key or "").strip().lower()
        if not k:
            continue
        if k.startswith("social_") or k.startswith("event_") or k.startswith("source_") or k.startswith("sentiment_"):
            out.append(idx)
            continue
        if k in {"comment_skew", "cross_source_consensus", "novelty_confidence_blend"}:
            out.append(idx)
            continue
        m = re.fullmatch(r"latent_(\d+)", k)
        if m:
            latent_id = int(m.group(1))
            if latent_id >= 64:
                out.append(idx)
    return sorted(set(out))


def _apply_feature_dropout(
    X: np.ndarray,
    indices: List[int],
    prob: float,
    seed: int,
) -> np.ndarray:
    if X.size == 0 or not indices:
        return X
    p = max(0.0, min(0.95, float(prob)))
    if p <= 1e-9:
        return X
    out = X.copy()
    rng = np.random.default_rng(int(seed))
    mask = rng.random((out.shape[0], len(indices))) < p
    selected = out[:, indices]
    selected[mask] = 0.0
    out[:, indices] = selected
    return out


def _ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    if X.ndim != 2:
        return np.zeros((0,), dtype=np.float64)
    if X.shape[1] == 0:
        return np.zeros((X.shape[0],), dtype=np.float64)
    if w.shape[0] != X.shape[1]:
        return np.zeros((X.shape[0],), dtype=np.float64)
    return (X @ w).astype(np.float64)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x.astype(np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _load_matrix(
    db_url: str,
    start_dt: datetime,
    end_dt: datetime,
    symbols: List[str],
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, as_of_ts, features
                FROM feature_matrix_main
                WHERE as_of_ts >= %s AND as_of_ts <= %s
                  AND symbol = ANY(%s)
                ORDER BY symbol ASC, as_of_ts ASC
                """,
                (start_dt, end_dt, symbols),
            )
            rows = [dict(r) for r in cur.fetchall()]

    by_symbol: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}
    for r in rows:
        symbol = str(r.get("symbol") or "").upper()
        ts = r.get("as_of_ts")
        payload = r.get("features") if isinstance(r.get("features"), dict) else {}
        if not symbol or ts is None:
            continue
        by_symbol.setdefault(symbol, []).append((ts, payload))

    X: List[List[float]] = []
    y: List[float] = []
    metas: List[str] = []
    step = max(1, int(horizon_steps))
    for symbol, seq in by_symbol.items():
        if len(seq) <= step:
            continue
        for i in range(0, len(seq) - step):
            ts, payload = seq[i]
            future_payload = seq[i + step][1]
            feat = [float(payload.get(k, 0.0) or 0.0) for k in LIQUID_FEATURE_KEYS]
            # Use future realized return proxy to avoid same-row target leakage.
            target = float(future_payload.get("ret_1", 0.0) or 0.0)
            X.append(feat)
            y.append(target)
            metas.append(f"{symbol}@{ts}")

    if not X:
        return np.zeros((0, len(LIQUID_FEATURE_KEYS)), dtype=np.float64), np.zeros((0,), dtype=np.float64), []
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), metas


def main() -> int:
    parser = argparse.ArgumentParser(description="Train multimodal baseline")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--l2", type=float, default=float(os.getenv("MULTIMODAL_L2", "0.05")))
    parser.add_argument("--horizon-steps", type=int, default=int(os.getenv("MULTIMODAL_TARGET_HORIZON_STEPS", "1")))
    parser.add_argument("--text-dropout-prob", type=float, default=float(os.getenv("MULTIMODAL_TEXT_DROPOUT_PROB", "0.1")))
    parser.add_argument("--fusion-mode", choices=["single_ridge", "residual_gate"], default=os.getenv("MULTIMODAL_FUSION_MODE", "single_ridge"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("TRAIN_SEED", "42")))
    parser.add_argument("--out", default="artifacts/models/multimodal_candidate.json")
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    X, y, _ = _load_matrix(args.database_url, start_dt, end_dt, symbols, horizon_steps=max(1, int(args.horizon_steps)))
    if X.shape[0] < 500:
        raise SystemExit("insufficient_rows_feature_matrix_main")

    cut = int(X.shape[0] * 0.8)
    cut = max(64, min(cut, X.shape[0] - 64))
    X_train, y_train = X[:cut], y[:cut]
    X_val, y_val = X[cut:], y[cut:]

    x_mean = X_train.mean(axis=0)
    x_std = np.clip(X_train.std(axis=0), 1e-6, None)
    Xn_train = (X_train - x_mean) / x_std
    Xn_val = (X_val - x_mean) / x_std
    text_indices = _infer_text_feature_indices(LIQUID_FEATURE_KEYS)
    non_text_indices = [i for i in range(len(LIQUID_FEATURE_KEYS)) if i not in set(text_indices)]
    Xn_train_for_fit = _apply_feature_dropout(
        Xn_train,
        indices=text_indices,
        prob=float(args.text_dropout_prob),
        seed=int(args.seed),
    )
    fusion_mode = str(args.fusion_mode).strip().lower()
    pred_val: np.ndarray
    out_extra: Dict[str, object] = {}
    if fusion_mode == "residual_gate" and text_indices:
        x_train_base = Xn_train_for_fit[:, non_text_indices] if non_text_indices else Xn_train_for_fit
        x_val_base = Xn_val[:, non_text_indices] if non_text_indices else Xn_val
        w_base = _ridge_fit(x_train_base, y_train, l2=float(args.l2))
        base_train = _ridge_predict(x_train_base, w_base)
        base_val = _ridge_predict(x_val_base, w_base)

        x_train_text = Xn_train_for_fit[:, text_indices]
        x_val_text = Xn_val[:, text_indices]
        residual_train = y_train - base_train
        w_text = _ridge_fit(x_train_text, residual_train, l2=float(args.l2))
        delta_train = _ridge_predict(x_train_text, w_text)
        delta_val = _ridge_predict(x_val_text, w_text)

        text_activity_train = np.mean(np.abs(x_train_text), axis=1) if x_train_text.size else np.zeros((x_train_text.shape[0],), dtype=np.float64)
        gate_mu = float(np.mean(text_activity_train)) if text_activity_train.size else 0.0
        gate_sigma = float(np.std(text_activity_train)) if text_activity_train.size else 0.0
        gate_scale = max(gate_sigma, 1e-6)
        gate_train = _sigmoid((text_activity_train - gate_mu) / gate_scale)
        denom = float(np.sum((gate_train * delta_train) ** 2))
        gate_multiplier = 1.0
        if denom > 1e-12:
            gate_multiplier = float(np.sum((y_train - base_train) * (gate_train * delta_train)) / denom)
        gate_multiplier = float(max(0.0, min(2.0, gate_multiplier)))

        text_activity_val = np.mean(np.abs(x_val_text), axis=1) if x_val_text.size else np.zeros((x_val_text.shape[0],), dtype=np.float64)
        gate_val = _sigmoid((text_activity_val - gate_mu) / gate_scale)
        pred_val = base_val + gate_multiplier * gate_val * delta_val
        out_extra = {
            "fusion_mode": "residual_gate",
            "base_feature_indices": [int(i) for i in non_text_indices],
            "base_weights": w_base.tolist(),
            "text_weights": w_text.tolist(),
            "gate_mu": gate_mu,
            "gate_sigma": gate_sigma,
            "gate_multiplier": gate_multiplier,
            "gate_val_mean": float(np.mean(gate_val)) if gate_val.size else 0.0,
            "gate_val_p90": float(np.percentile(gate_val, 90)) if gate_val.size else 0.0,
            "delta_val_mean_abs": float(np.mean(np.abs(delta_val))) if delta_val.size else 0.0,
        }
    else:
        w = _ridge_fit(Xn_train_for_fit, y_train, l2=float(args.l2))
        pred_val = Xn_val @ w
        out_extra = {
            "fusion_mode": "single_ridge",
            "weights": w.tolist(),
        }

    mse = float(np.mean((pred_val - y_val) ** 2))
    mae = float(np.mean(np.abs(pred_val - y_val)))

    out_obj: Dict[str, object] = {
        "model_name": "multimodal_ridge",
        "model_version": "main",
        "track": "liquid",
        "type": "tabular_linear",
        "feature_version": "feature-store-main",
        "feature_payload_schema_version": str(LIQUID_FEATURE_SCHEMA_VERSION),
        "feature_dim": int(X.shape[1]),
        "data_version": "2018_now_full_window",
        "symbols": symbols,
        "train_start": start_dt.isoformat().replace("+00:00", "Z"),
        "train_end": end_dt.isoformat().replace("+00:00", "Z"),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "val_mse": mse,
        "val_mae": mae,
        "fusion_mode": fusion_mode,
        "text_dropout_prob": float(max(0.0, min(0.95, float(args.text_dropout_prob)))),
        "text_feature_count": int(len(text_indices)),
        "text_feature_indices": [int(i) for i in text_indices],
        "dropout_seed": int(args.seed),
        "target_definition": f"future_ret_1_step_{max(1, int(args.horizon_steps))}",
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    out_obj.update(out_extra)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "out": str(out_path), "val_mse": mse, "val_mae": mae}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
