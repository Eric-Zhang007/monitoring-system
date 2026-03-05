from __future__ import annotations

from cost.cost_profile import compute_cost_bps, compute_cost_breakdown_bps, compute_cost_map, load_cost_profile
from training.labels.liquid_labels import compute_label_targets


def test_cost_profile_consistent_across_labels_eval_decision_paths():
    profile = load_cost_profile("standard")
    market = {"realized_vol": 0.03, "funding_rate": 0.0002, "notional_usd": 50000.0}
    liq = {"liquidity_score": 0.7, "orderbook_depth_total": 2_000_000.0, "spread_bps": 5.0}
    c1 = compute_cost_bps(
        horizon="1h",
        profile=profile,
        market_state=market,
        liquidity_features=liq,
        turnover_estimate=0.4,
        account_state={"notional_usd": 50000.0},
    )
    label = compute_label_targets(
        prices=[100.0 + i * 0.2 for i in range(3000)],
        index=100,
        horizon_steps={"1h": 12, "4h": 48, "1d": 288, "7d": 2016},
        market_state=market,
        liquidity_features=liq,
        turnover_estimate=0.4,
        cost_profile_name="standard",
    )
    c_label = float(label["cost_1h_bps"])
    c_map = float(compute_cost_map(horizons=("1h",), profile=profile, market_state=market, liquidity_features=liq, turnover_estimate=0.4)["1h"])
    c_breakdown = float(
        compute_cost_breakdown_bps(
            horizon="1h",
            profile=profile,
            market_state=market,
            liquidity_features=liq,
            turnover_estimate=0.4,
            account_state={"notional_usd": 50000.0},
        )["total_bps"]
    )
    assert c1 > 0
    assert abs(c1 - c_label) < 1e-6
    assert abs(c1 - c_map) < 1e-6
    assert abs(c1 - c_breakdown) < 1e-6
