from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def ks_statistic(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    x = np.sort(np.array(a, dtype=np.float64))
    y = np.sort(np.array(b, dtype=np.float64))
    all_vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, all_vals, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, all_vals, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    if not reference or not current:
        return 0.0
    ref = np.array(reference, dtype=np.float64)
    cur = np.array(current, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def flatten_numeric_payloads(payloads: List[Dict[str, float]]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for p in payloads:
        if not isinstance(p, dict):
            continue
        for k, v in p.items():
            if isinstance(v, (int, float)):
                out.setdefault(str(k), []).append(float(v))
    return out


def feature_drift_score(reference_payloads: List[Dict[str, float]], current_payloads: List[Dict[str, float]]) -> Tuple[float, float]:
    ref_map = flatten_numeric_payloads(reference_payloads)
    cur_map = flatten_numeric_payloads(current_payloads)
    keys = set(ref_map.keys()) | set(cur_map.keys())
    if not keys:
        return 0.0, 0.0
    psi_vals = []
    mean_shift = []
    for k in keys:
        rv = ref_map.get(k, [])
        cv = cur_map.get(k, [])
        if not rv or not cv:
            continue
        psi_vals.append(psi(rv, cv))
        r_mean = float(np.mean(rv))
        c_mean = float(np.mean(cv))
        mean_shift.append(abs(c_mean - r_mean))
    if not psi_vals:
        return 0.0, 0.0
    return float(np.mean(psi_vals)), float(np.mean(mean_shift) if mean_shift else 0.0)
