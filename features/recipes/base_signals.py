from __future__ import annotations

from typing import Dict, List

import numpy as np


def compute_price_signals(prices: List[float], volumes: List[float], idx: int) -> Dict[str, float]:
    p = float(prices[idx])
    if p <= 0:
        return {}

    def _ret(k: int) -> float:
        j = max(0, idx - k)
        base = float(prices[j]) if j < len(prices) else p
        return float((p - base) / max(base, 1e-12))

    def _vol(k: int) -> float:
        j = max(0, idx - k)
        seg = np.array(prices[j: idx + 1], dtype=np.float64)
        if seg.size < 3:
            return 0.0
        return float(np.std(np.diff(np.log(np.clip(seg, 1e-12, None)))))

    ret_1 = _ret(1)
    ret_3 = _ret(3)
    ret_12 = _ret(12)
    ret_48 = _ret(48)
    ret_96 = _ret(96)
    ret_288 = _ret(288)

    vol_3 = _vol(3)
    vol_12 = _vol(12)
    vol_48 = _vol(48)
    vol_96 = _vol(96)
    vol_288 = _vol(288)

    v = float(volumes[idx]) if idx < len(volumes) else 0.0
    vh = np.array(volumes[max(0, idx - 48): idx + 1], dtype=np.float64)
    vmean = float(np.mean(vh)) if vh.size else 0.0
    vstd = float(np.std(vh)) if vh.size else 0.0
    volume_z = float((v - vmean) / max(vstd, 1e-9))

    alpha12 = 2.0 / 13.0
    alpha48 = 2.0 / 49.0
    ewm12 = p
    ewm48 = p
    for vv in prices[max(0, idx - 96): idx + 1]:
        vv = float(vv)
        ewm12 = alpha12 * vv + (1 - alpha12) * ewm12
        ewm48 = alpha48 * vv + (1 - alpha48) * ewm48

    return {
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_12": ret_12,
        "ret_48": ret_48,
        "ret_96": ret_96,
        "ret_288": ret_288,
        "vol_3": vol_3,
        "vol_12": vol_12,
        "vol_48": vol_48,
        "vol_96": vol_96,
        "vol_288": vol_288,
        "log_volume": float(np.log1p(max(0.0, v))),
        "volume_z": volume_z,
        "volume_impact": float(abs(ret_1) / max(np.sqrt(max(1.0, v)), 1e-9)),
        "price_ewm_12": float(ewm12),
        "price_ewm_48": float(ewm48),
    }
