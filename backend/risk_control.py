from __future__ import annotations

from typing import Tuple


def drawdown_thresholds(dd_limit: float, warn_default: float, near_default: float) -> Tuple[float, float]:
    warn = float(warn_default)
    near = float(near_default)
    if dd_limit > 1e-9:
        warn = min(warn, max(0.0, dd_limit * 0.9))
        near = min(near, max(0.0, dd_limit * 0.98))
    if near <= warn:
        near = min(max(warn + 0.005, near), dd_limit if dd_limit > 0 else warn + 0.005)
    return max(0.0, warn), max(0.0, near)
