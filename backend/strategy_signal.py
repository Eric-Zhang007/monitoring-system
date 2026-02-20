from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def strategy_bucket(
    *,
    track: str,
    score: float,
    confidence: float,
    target: str,
    latest_feature_payload_fn: Callable[[str], Dict[str, Any]],
) -> str:
    if track == "vc":
        return "event"
    try:
        f = latest_feature_payload_fn(str(target or "").upper())
    except Exception:
        f = {}
    ret_1 = float(f.get("ret_1", 0.0) or 0.0)
    ret_12 = float(f.get("ret_12", 0.0) or 0.0)
    vol_z = float(f.get("vol_z", 0.0) or 0.0)
    event_decay = float(f.get("event_decay", 0.0) or 0.0)
    trend_strength = float(np.tanh(ret_12 * 8.0 + ret_1 * 3.0 + float(score) * 2.0))
    mr_signal = abs(ret_1) + abs(ret_12)
    is_low_noise = abs(vol_z) < 1.2
    if abs(trend_strength) > 0.35 and confidence > 0.55 and is_low_noise:
        return "trend"
    if mr_signal < 0.004 and confidence > 0.6 and is_low_noise:
        return "mean_reversion"
    if event_decay > 0.0:
        return "event"
    return "event"
