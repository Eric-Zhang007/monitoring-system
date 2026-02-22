from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from cost.cost_profile import compute_cost_map, load_cost_profile


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = _rankdata(x)
    ry = _rankdata(y)
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    curve = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return float(abs(np.min(dd)))


def _avg_holding(signal: np.ndarray) -> float:
    hold_lengths: List[int] = []
    cur = 0
    for v in signal:
        if abs(v) > 1e-9:
            cur += 1
        elif cur > 0:
            hold_lengths.append(cur)
            cur = 0
    if cur > 0:
        hold_lengths.append(cur)
    if not hold_lengths:
        return 0.0
    return float(np.mean(np.array(hold_lengths, dtype=np.float64)))


def _ece(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if probs.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        out += abs(conf - acc) * (float(np.sum(mask)) / float(probs.size))
    return float(out)


def evaluate_liquid_metrics(
    *,
    horizons: Sequence[str],
    y_raw: np.ndarray,
    y_net: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    direction_logit: np.ndarray | None,
    q: np.ndarray | None,
    feature_context: Mapping[str, float] | None = None,
    cost_profile_name: str = "standard",
) -> Dict[str, Dict[str, float]]:
    feature_context = dict(feature_context or {})
    out: Dict[str, Dict[str, float]] = {}
    eps = 1e-6

    for h_idx, h in enumerate(horizons):
        hr = str(h)
        yt_raw = y_raw[:, h_idx].astype(np.float64)
        yt_net = y_net[:, h_idx].astype(np.float64)
        pred_mu = mu[:, h_idx].astype(np.float64)
        pred_sigma = np.clip(sigma[:, h_idx].astype(np.float64), eps, None)

        mse = float(np.mean((pred_mu - yt_net) ** 2))
        mae = float(np.mean(np.abs(pred_mu - yt_net)))
        ic = _spearman(pred_mu, yt_net)
        hit = float(np.mean(np.sign(pred_mu) == np.sign(yt_net)))

        pos = np.clip(pred_mu / pred_sigma, -1.0, 1.0)
        prev = np.roll(pos, 1)
        prev[0] = 0.0
        turnover = np.abs(pos - prev)

        dynamic_cost_bps = compute_cost_map(
            horizons=[hr],
            profile=load_cost_profile(cost_profile_name),
            market_state={"realized_vol": float(np.mean(np.abs(yt_raw))), "notional_usd": float(feature_context.get("notional_usd", 1.0) or 1.0)},
            liquidity_features=feature_context,
            turnover_estimate=float(np.mean(turnover)),
        )[hr]
        pnl = pos * yt_raw - turnover * (dynamic_cost_bps / 1e4)
        sharpe = float(np.mean(pnl) / max(1e-9, np.std(pnl)))
        mdd = _max_drawdown(pnl)
        avg_hold = _avg_holding(pos)

        brier = 0.0
        ece = 0.0
        if direction_logit is not None:
            dlogit = direction_logit[:, h_idx].astype(np.float64)
            probs = 1.0 / (1.0 + np.exp(-np.clip(dlogit, -40.0, 40.0)))
            labels = (yt_net >= 0.0).astype(np.float64)
            brier = float(np.mean((probs - labels) ** 2))
            ece = _ece(probs, labels)

        q_cover = 0.0
        if q is not None and q.shape[2] >= 3:
            q10 = q[:, h_idx, 0].astype(np.float64)
            q90 = q[:, h_idx, -1].astype(np.float64)
            q_cover = float(np.mean((yt_net >= q10) & (yt_net <= q90)))

        out[hr] = {
            "mse": mse,
            "mae": mae,
            "spearman_ic": ic,
            "hit_rate": hit,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "turnover": float(np.mean(turnover)),
            "avg_holding": avg_hold,
            "brier": brier,
            "ece": ece,
            "q10_q90_coverage": q_cover,
        }

    return out
