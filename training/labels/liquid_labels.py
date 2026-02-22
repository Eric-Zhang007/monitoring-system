from __future__ import annotations

from typing import Dict, Mapping, Sequence


def compute_label_targets(
    *,
    prices: Sequence[float],
    index: int,
    horizon_steps: Mapping[str, int],
    cost_bps: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    p0 = float(prices[index])
    if p0 <= 0:
        raise ValueError("invalid_price")
    costs = dict(cost_bps or {"1h": 4.0, "4h": 6.0, "1d": 10.0, "7d": 14.0})
    out: Dict[str, float] = {}
    for h in ("1h", "4h", "1d", "7d"):
        step = int(horizon_steps[h])
        p1 = float(prices[index + step])
        raw = (p1 - p0) / max(1e-12, p0)
        cost = float(costs.get(h, 0.0)) / 10000.0
        net = raw - cost
        out[f"ret_{h}_raw"] = float(raw)
        out[f"ret_{h}_net"] = float(net)
        out[f"direction_{h}"] = 1.0 if net >= 0 else 0.0
        out[f"risk_proxy_{h}"] = float(abs(raw))
    return out
