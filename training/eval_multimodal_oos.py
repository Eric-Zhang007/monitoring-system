#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from inference.liquid_feature_contract import LIQUID_FEATURE_KEYS
from validation import (
    evaluate_regression_oos,
    purged_kfold_slices,
    resolve_validation_protocol,
    summarize_fold_metrics,
    validation_protocol_to_dict,
    walk_forward_slices,
)


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _load_matrix(
    db_url: str,
    start_dt: datetime,
    end_dt: datetime,
    symbols: List[str],
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
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

    by_symbol: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        symbol = str(r.get("symbol") or "").upper()
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(r)

    X, y = [], []
    step = max(1, int(horizon_steps))
    for _, seq in by_symbol.items():
        if len(seq) <= step:
            continue
        for i in range(0, len(seq) - step):
            payload = seq[i].get("features") if isinstance(seq[i].get("features"), dict) else {}
            future_payload = seq[i + step].get("features") if isinstance(seq[i + step].get("features"), dict) else {}
            X.append([float(payload.get(k, 0.0) or 0.0) for k in LIQUID_FEATURE_KEYS])
            y.append(float(future_payload.get("ret_1", 0.0) or 0.0))
    if not X:
        return np.zeros((0, len(LIQUID_FEATURE_KEYS))), np.zeros((0,))
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.float64)


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float = 5e-2) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(d), X.T @ y)


def _summarize_basic(folds: List[Dict[str, float]]) -> Dict[str, float]:
    if not folds:
        return {"folds": 0.0, "mse": 0.0, "mae": 0.0, "hit_rate": 0.0}
    return {
        "folds": float(len(folds)),
        "mse": float(np.mean([m["mse"] for m in folds])),
        "mae": float(np.mean([m["mae"] for m in folds])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in folds])),
    }


def _infer_text_feature_indices(keys: Sequence[str]) -> List[int]:
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


def _infer_macro_feature_indices(keys: Sequence[str]) -> List[int]:
    macro_exact = {
        "funding_rate",
        "onchain_norm",
        "funding_missing_flag",
        "onchain_missing_flag",
    }
    out = []
    for idx, key in enumerate(keys):
        k = str(key or "").strip().lower()
        if not k:
            continue
        if k in macro_exact or k.startswith("macro_") or k.startswith("deriv_"):
            out.append(idx)
    return sorted(set(out))


def _event_strength_index_map(keys: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for idx, key in enumerate(keys):
        out[str(key)] = idx
    return out


def _event_strength_scores(X: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    idx_map = _event_strength_index_map(keys)
    parts: List[np.ndarray] = []
    for k in ("social_buzz", "event_density", "event_decay", "event_velocity_1h"):
        idx = idx_map.get(k)
        if idx is None or idx < 0 or idx >= X.shape[1]:
            continue
        parts.append(np.abs(X[:, idx]))
    if not parts:
        return np.zeros((X.shape[0],), dtype=np.float64)
    stacked = np.stack(parts, axis=1)
    return np.mean(stacked, axis=1).astype(np.float64)


def _materialize_ablation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    ablation: str,
    keys: Sequence[str],
    event_strength_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    name = str(ablation or "full").strip().lower()
    if name in {"", "full", "none"}:
        return X.copy(), y.copy(), {"ablation": "full", "rows": int(X.shape[0]), "masked_features": 0}

    if name == "no_text":
        idx = _infer_text_feature_indices(keys)
        XX = X.copy()
        if idx:
            XX[:, idx] = 0.0
        return XX, y.copy(), {"ablation": "no_text", "rows": int(XX.shape[0]), "masked_features": int(len(idx))}

    if name == "no_macro":
        idx = _infer_macro_feature_indices(keys)
        XX = X.copy()
        if idx:
            XX[:, idx] = 0.0
        return XX, y.copy(), {"ablation": "no_macro", "rows": int(XX.shape[0]), "masked_features": int(len(idx))}

    if name in {"event_window", "event_only"}:
        strength = _event_strength_scores(X, keys)
        th = max(0.0, float(event_strength_threshold))
        mask = strength >= th
        XX = X[mask]
        yy = y[mask]
        return (
            XX,
            yy,
            {
                "ablation": "event_window",
                "rows": int(XX.shape[0]),
                "masked_features": 0,
                "event_strength_threshold": th,
                "selected_ratio": float(np.mean(mask)) if mask.size else 0.0,
            },
        )

    return X.copy(), y.copy(), {"ablation": name, "rows": int(X.shape[0]), "masked_features": 0, "unknown_ablation": True}


def _evaluate_single_dataset(
    X: np.ndarray,
    y: np.ndarray,
    *,
    protocol,
    l2: float,
) -> Dict[str, object]:
    if X.shape[0] < max(64, int(protocol.min_train_points) + int(protocol.min_test_points)):
        return {
            "status": "blocked",
            "reason": "insufficient_rows_after_ablation",
            "rows": int(X.shape[0]),
        }

    wf_folds = walk_forward_slices(
        n_samples=X.shape[0],
        train_window=protocol.wf_train_window,
        test_window=protocol.wf_test_window,
        purge_window=protocol.wf_purge_window,
        step_window=protocol.wf_step_window,
    )
    wf_basic_per_fold: List[Dict[str, float]] = []
    wf_oos_per_fold: List[Dict[str, float]] = []
    wf_conf_proxy: List[float] = []
    wf_hit_obs: List[float] = []
    for tr, te in wf_folds:
        if tr.size < protocol.min_train_points or te.size < protocol.min_test_points:
            continue
        x_mean = X[tr].mean(axis=0)
        x_std = np.clip(X[tr].std(axis=0), 1e-6, None)
        Xtr = (X[tr] - x_mean) / x_std
        Xte = (X[te] - x_mean) / x_std
        w = _ridge_fit(Xtr, y[tr], l2=l2)
        pred = Xte @ w
        mse = float(np.mean((pred - y[te]) ** 2))
        mae = float(np.mean(np.abs(pred - y[te])))
        hit = float(np.mean(((pred >= 0).astype(np.int32) == (y[te] >= 0).astype(np.int32)).astype(np.float64)))
        wf_basic_per_fold.append({"mse": mse, "mae": mae, "hit_rate": hit})
        pred_std = float(np.std(pred)) if pred.size else 0.0
        z = np.abs(pred / max(1e-6, pred_std))
        conf_proxy = 1.0 / (1.0 + np.exp(-(0.85 * z - 0.1)))
        hit_obs = ((pred >= 0).astype(np.int32) == (y[te] >= 0).astype(np.int32)).astype(np.float64)
        wf_conf_proxy.extend(conf_proxy.tolist())
        wf_hit_obs.extend(hit_obs.tolist())
        wf_oos_per_fold.append(
            evaluate_regression_oos(
                y_true=y[te],
                y_pred=pred,
                fee_bps=5.0,
                slippage_bps=3.0,
            )
        )

    pkf_folds = purged_kfold_slices(
        n_samples=X.shape[0],
        n_splits=protocol.pkf_splits,
        purge_window=protocol.pkf_purge_window,
    )
    pkf_basic_per_fold: List[Dict[str, float]] = []
    pkf_oos_per_fold: List[Dict[str, float]] = []
    for tr, te in pkf_folds:
        if tr.size < protocol.min_train_points or te.size < protocol.min_test_points:
            continue
        x_mean = X[tr].mean(axis=0)
        x_std = np.clip(X[tr].std(axis=0), 1e-6, None)
        Xtr = (X[tr] - x_mean) / x_std
        Xte = (X[te] - x_mean) / x_std
        w = _ridge_fit(Xtr, y[tr], l2=l2)
        pred = Xte @ w
        mse = float(np.mean((pred - y[te]) ** 2))
        mae = float(np.mean(np.abs(pred - y[te])))
        hit = float(np.mean(((pred >= 0).astype(np.int32) == (y[te] >= 0).astype(np.int32)).astype(np.float64)))
        pkf_basic_per_fold.append({"mse": mse, "mae": mae, "hit_rate": hit})
        pkf_oos_per_fold.append(
            evaluate_regression_oos(
                y_true=y[te],
                y_pred=pred,
                fee_bps=5.0,
                slippage_bps=3.0,
            )
        )

    wf_basic = _summarize_basic(wf_basic_per_fold)
    wf_oos = summarize_fold_metrics(wf_oos_per_fold)
    pkf_basic = _summarize_basic(pkf_basic_per_fold)
    pkf_oos = summarize_fold_metrics(pkf_oos_per_fold)
    if int(wf_basic["folds"]) <= 0:
        return {"status": "blocked", "reason": "no_valid_wf_folds", "rows": int(X.shape[0])}
    calibration_bins: List[Dict[str, float]] = []
    if wf_conf_proxy and wf_hit_obs and len(wf_conf_proxy) == len(wf_hit_obs):
        arr_c = np.asarray(wf_conf_proxy, dtype=np.float64)
        arr_h = np.asarray(wf_hit_obs, dtype=np.float64)
        order = np.argsort(arr_c)
        for idx in np.array_split(order, 5):
            if idx.size <= 0:
                continue
            calibration_bins.append(
                {
                    "count": int(idx.size),
                    "confidence_mean": float(np.mean(arr_c[idx])),
                    "hit_rate": float(np.mean(arr_h[idx])),
                }
            )

    return {
        "status": "ok",
        "rows": int(X.shape[0]),
        "folds": int(wf_basic["folds"]),
        "mse": float(wf_basic["mse"]),
        "mae": float(wf_basic["mae"]),
        "hit_rate": float(wf_basic["hit_rate"]),
        "walk_forward_ready": bool(int(wf_basic["folds"]) >= int(protocol.wf_min_folds)),
        "walk_forward": wf_oos,
        "walk_forward_basic": wf_basic,
        "purged_kfold": pkf_oos,
        "purged_kfold_basic": pkf_basic,
        "confidence_calibration_bins": calibration_bins,
        "wf_metrics_per_fold": wf_basic_per_fold,
        "pkf_metrics_per_fold": pkf_basic_per_fold,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate multimodal with purged walk-forward")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--horizon-steps", type=int, default=int(os.getenv("MULTIMODAL_TARGET_HORIZON_STEPS", "1")))
    parser.add_argument("--l2", type=float, default=float(os.getenv("MULTIMODAL_L2", "0.05")))
    parser.add_argument("--ablations", default=os.getenv("MULTIMODAL_ABLATIONS", "full,no_text,no_macro,event_window"))
    parser.add_argument(
        "--event-strength-threshold",
        type=float,
        default=float(os.getenv("MULTIMODAL_EVENT_WINDOW_THRESHOLD", "0.05")),
    )
    parser.add_argument("--out", default="artifacts/models/multimodal_eval.json")
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    X, y = _load_matrix(args.database_url, start_dt, end_dt, symbols, horizon_steps=max(1, int(args.horizon_steps)))
    if X.shape[0] < 1200:
        raise SystemExit("insufficient_rows_for_oos_eval")

    protocol = resolve_validation_protocol(prefix="LIQUID")
    ablations = []
    for x in str(args.ablations).split(","):
        cur = str(x).strip().lower()
        if cur and cur not in ablations:
            ablations.append(cur)
    if not ablations:
        ablations = ["full"]
    if "full" not in ablations:
        ablations = ["full", *ablations]

    ablation_results: List[Dict[str, object]] = []
    for ab in ablations:
        Xa, ya, ab_meta = _materialize_ablation(
            X,
            y,
            ablation=ab,
            keys=LIQUID_FEATURE_KEYS,
            event_strength_threshold=float(args.event_strength_threshold),
        )
        cur = _evaluate_single_dataset(
            Xa,
            ya,
            protocol=protocol,
            l2=float(args.l2),
        )
        ablation_results.append({**ab_meta, **cur})

    primary = None
    for item in ablation_results:
        if str(item.get("ablation")) == "full":
            primary = item
            break
    if primary is None and ablation_results:
        primary = ablation_results[0]
    if primary is None or str(primary.get("status")) != "ok":
        raise SystemExit("no_valid_ablation_result")

    if int(primary.get("folds", 0)) < int(protocol.wf_min_folds):
        raise SystemExit("wf_min_folds_not_reached")

    out = {
        "status": "ok",
        "primary_ablation": str(primary.get("ablation") or "full"),
        "folds": int(primary["folds"]),
        "mse": float(primary["mse"]),
        "mae": float(primary["mae"]),
        "hit_rate": float(primary["hit_rate"]),
        "walk_forward_ready": bool(primary.get("walk_forward_ready")),
        "validation_protocol": validation_protocol_to_dict(protocol),
        "walk_forward": primary["walk_forward"],
        "walk_forward_basic": primary["walk_forward_basic"],
        "purged_kfold": primary["purged_kfold"],
        "purged_kfold_basic": primary["purged_kfold_basic"],
        "ablation_results": ablation_results,
        "confidence_calibration_bins": primary.get("confidence_calibration_bins") or [],
        "ablation_config": {
            "requested": ablations,
            "event_strength_threshold": float(max(0.0, float(args.event_strength_threshold))),
            "l2": float(args.l2),
        },
        "target_definition": f"future_ret_1_step_{max(1, int(args.horizon_steps))}",
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wf_metrics_per_fold": primary["wf_metrics_per_fold"],
        "pkf_metrics_per_fold": primary["pkf_metrics_per_fold"],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "out": str(out_path), "folds": int(out["folds"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
