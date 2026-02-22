from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from cost.cost_profile import compute_cost_bps, load_cost_profile


def compute_label_targets(
    *,
    prices: Sequence[float],
    index: int,
    horizon_steps: Mapping[str, int],
    cost_bps: Mapping[str, float] | None = None,
    market_state: Mapping[str, Any] | None = None,
    liquidity_features: Mapping[str, Any] | None = None,
    turnover_estimate: float | None = None,
    cost_profile_name: str = "standard",
) -> Dict[str, float]:
    p0 = float(prices[index])
    if p0 <= 0:
        raise ValueError("invalid_price")
    profile = load_cost_profile(cost_profile_name)
    costs = dict(cost_bps or {})
    out: Dict[str, float] = {}
    for h in ("1h", "4h", "1d", "7d"):
        if h not in costs:
            costs[h] = compute_cost_bps(
                horizon=h,
                profile=profile,
                market_state=market_state,
                liquidity_features=liquidity_features,
                turnover_estimate=turnover_estimate,
            )
        step = int(horizon_steps[h])
        p1 = float(prices[index + step])
        raw = (p1 - p0) / max(1e-12, p0)
        cost = float(costs.get(h, 0.0)) / 10000.0
        net = raw - cost
        out[f"ret_{h}_raw"] = float(raw)
        out[f"cost_{h}_bps"] = float(costs.get(h, 0.0))
        out[f"ret_{h}_net"] = float(net)
        out[f"direction_{h}"] = 1.0 if net >= 0 else 0.0
        out[f"risk_proxy_{h}"] = float(abs(raw))
    return out
