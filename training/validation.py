from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class ValidationProtocol:
    wf_train_window: int = 512
    wf_test_window: int = 96
    wf_purge_window: int = 12
    wf_step_window: int = 96
    wf_min_folds: int = 3
    pkf_splits: int = 5
    pkf_purge_window: int = 12
    min_train_points: int = 64
    min_test_points: int = 16


def _env_int(candidates: List[str], default: int) -> int:
    for name in candidates:
        raw = str(os.getenv(name, "")).strip()
        if not raw:
            continue
        try:
            return int(raw)
        except Exception:
            continue
    return int(default)


def resolve_validation_protocol(prefix: str = "LIQUID") -> ValidationProtocol:
    key_prefix = str(prefix or "LIQUID").strip().upper()
    wf_train_window = _env_int([f"{key_prefix}_WF_TRAIN_WINDOW", "VALIDATION_WF_TRAIN_WINDOW"], 512)
    wf_test_window = _env_int([f"{key_prefix}_WF_TEST_WINDOW", "VALIDATION_WF_TEST_WINDOW"], 96)
    wf_purge_window = _env_int([f"{key_prefix}_WF_PURGE_WINDOW", "VALIDATION_WF_PURGE_WINDOW"], 12)
    wf_step_window = _env_int([f"{key_prefix}_WF_STEP_WINDOW", "VALIDATION_WF_STEP_WINDOW"], wf_test_window)
    wf_min_folds = _env_int([f"{key_prefix}_WF_MIN_FOLDS", "VALIDATION_WF_MIN_FOLDS"], 3)
    pkf_splits = _env_int([f"{key_prefix}_PURGED_KFOLD_SPLITS", "VALIDATION_PURGED_KFOLD_SPLITS"], 5)
    pkf_purge_window = _env_int([f"{key_prefix}_PURGED_KFOLD_PURGE", "VALIDATION_PURGED_KFOLD_PURGE"], 12)
    min_train_points = _env_int([f"{key_prefix}_MIN_TRAIN_POINTS", "VALIDATION_MIN_TRAIN_POINTS"], 64)
    min_test_points = _env_int([f"{key_prefix}_MIN_TEST_POINTS", "VALIDATION_MIN_TEST_POINTS"], 16)
    return ValidationProtocol(
        wf_train_window=max(1, int(wf_train_window)),
        wf_test_window=max(1, int(wf_test_window)),
        wf_purge_window=max(0, int(wf_purge_window)),
        wf_step_window=max(1, int(wf_step_window)),
        wf_min_folds=max(1, int(wf_min_folds)),
        pkf_splits=max(2, int(pkf_splits)),
        pkf_purge_window=max(0, int(pkf_purge_window)),
        min_train_points=max(1, int(min_train_points)),
        min_test_points=max(1, int(min_test_points)),
    )


def validation_protocol_to_dict(protocol: ValidationProtocol) -> Dict[str, int]:
    return {
        "wf_train_window": int(protocol.wf_train_window),
        "wf_test_window": int(protocol.wf_test_window),
        "wf_purge_window": int(protocol.wf_purge_window),
        "wf_step_window": int(protocol.wf_step_window),
        "wf_min_folds": int(protocol.wf_min_folds),
        "pkf_splits": int(protocol.pkf_splits),
        "pkf_purge_window": int(protocol.pkf_purge_window),
        "min_train_points": int(protocol.min_train_points),
        "min_test_points": int(protocol.min_test_points),
    }


def walk_forward_slices(
    n_samples: int,
    train_window: int,
    test_window: int,
    purge_window: int,
    step_window: int | None = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0 or train_window <= 0 or test_window <= 0:
        return []
    step = step_window or test_window
    if step <= 0:
        step = test_window
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_window
        test_start = train_end + max(0, purge_window)
        test_end = test_start + test_window
        if test_end > n_samples:
            break
        train_idx = np.arange(train_start, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        splits.append((train_idx, test_idx))
        start += step
    return splits


def purged_kfold_slices(
    n_samples: int,
    n_splits: int = 5,
    purge_window: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_samples <= 0 or n_splits < 2:
        return []
    fold_sizes = [n_samples // n_splits] * n_splits
    for i in range(n_samples % n_splits):
        fold_sizes[i] += 1
    offsets = [0]
    for s in fold_sizes:
        offsets.append(offsets[-1] + s)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    all_idx = np.arange(n_samples, dtype=np.int64)
    for k in range(n_splits):
        test_start = offsets[k]
        test_end = offsets[k + 1]
        test_idx = all_idx[test_start:test_end]
        left_end = max(0, test_start - purge_window)
        right_start = min(n_samples, test_end + purge_window)
        train_idx = np.concatenate([all_idx[:left_end], all_idx[right_start:]], axis=0)
        if train_idx.size == 0 or test_idx.size == 0:
            continue
        out.append((train_idx, test_idx))
    return out


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(abs(np.min(dd)))


def evaluate_regression_oos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fee_bps: float = 5.0,
    slippage_bps: float = 3.0,
) -> Dict[str, float]:
    if y_true.size == 0 or y_pred.size == 0:
        return {
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    std_yt = float(np.std(yt))
    std_yp = float(np.std(yp))
    if std_yt < 1e-12 or std_yp < 1e-12:
        ic = 0.0
    else:
        ic = float(np.corrcoef(yt, yp)[0, 1])
    signal = np.sign(yp)
    signal_prev = np.roll(signal, 1)
    signal_prev[0] = 0.0
    turnover = np.abs(signal - signal_prev)
    turnover_mean = float(np.mean(turnover))
    cost = (fee_bps + slippage_bps) / 10000.0
    strategy_ret = signal * yt - turnover * cost
    pnl_after_cost = float(np.mean(strategy_ret))
    hit_rate = float(np.mean(np.sign(yt) == signal))
    max_dd = _max_drawdown(strategy_ret)
    return {
        "ic": ic,
        "hit_rate": hit_rate,
        "turnover": turnover_mean,
        "pnl_after_cost": pnl_after_cost,
        "max_drawdown": max_dd,
    }


def summarize_fold_metrics(folds: List[Dict[str, float]]) -> Dict[str, float]:
    if not folds:
        return {
            "folds": 0.0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }
    return {
        "folds": float(len(folds)),
        "ic": float(np.mean([f["ic"] for f in folds])),
        "hit_rate": float(np.mean([f["hit_rate"] for f in folds])),
        "turnover": float(np.mean([f["turnover"] for f in folds])),
        "pnl_after_cost": float(np.mean([f["pnl_after_cost"] for f in folds])),
        "max_drawdown": float(np.max([f["max_drawdown"] for f in folds])),
    }
