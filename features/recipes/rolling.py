from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


def safe_array(values: Iterable[float]) -> np.ndarray:
    return np.array([float(v) for v in values], dtype=np.float64)


def rolling_stats(values: Iterable[float], windows: List[int]) -> Dict[str, float]:
    arr = safe_array(values)
    out: Dict[str, float] = {}
    for w in windows:
        seg = arr[-w:] if arr.size >= w else arr
        if seg.size <= 0:
            out[f"mean_{w}"] = 0.0
            out[f"std_{w}"] = 0.0
            out[f"min_{w}"] = 0.0
            out[f"max_{w}"] = 0.0
            out[f"last_{w}"] = 0.0
            out[f"zscore_{w}"] = 0.0
            out[f"ewm_mean_{w}"] = 0.0
            out[f"ewm_std_{w}"] = 0.0
            out[f"autocorr1_{w}"] = 0.0
            out[f"quantile_10_{w}"] = 0.0
            out[f"quantile_90_{w}"] = 0.0
            continue
        mean = float(np.mean(seg))
        std = float(np.std(seg))
        out[f"mean_{w}"] = mean
        out[f"std_{w}"] = std
        out[f"min_{w}"] = float(np.min(seg))
        out[f"max_{w}"] = float(np.max(seg))
        out[f"last_{w}"] = float(seg[-1])
        out[f"zscore_{w}"] = float((seg[-1] - mean) / max(std, 1e-9))
        alpha = 2.0 / (max(1, w) + 1.0)
        ewm = np.zeros_like(seg)
        ewm[0] = seg[0]
        for i in range(1, seg.size):
            ewm[i] = alpha * seg[i] + (1 - alpha) * ewm[i - 1]
        out[f"ewm_mean_{w}"] = float(ewm[-1])
        out[f"ewm_std_{w}"] = float(np.std(ewm))
        if seg.size > 2 and float(np.std(seg[:-1])) > 1e-12 and float(np.std(seg[1:])) > 1e-12:
            out[f"autocorr1_{w}"] = float(np.corrcoef(seg[:-1], seg[1:])[0, 1])
        else:
            out[f"autocorr1_{w}"] = 0.0
        out[f"quantile_10_{w}"] = float(np.quantile(seg, 0.1))
        out[f"quantile_90_{w}"] = float(np.quantile(seg, 0.9))
    return out
