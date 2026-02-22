from __future__ import annotations

import numpy as np

from cost.cost_profile import compute_cost_bps, compute_cost_map, load_cost_profile
from training.labels.liquid_labels import compute_label_targets
from training.validation import evaluate_regression_oos


def test_cost_profile_parity_labels_action_eval():
    profile = load_cost_profile("standard")
    market = {"realized_vol": 0.02}
    liq = {"liquidity_score": 0.8}

    c1 = compute_cost_bps(horizon="1h", profile=profile, market_state=market, liquidity_features=liq, turnover_estimate=0.5)
    labels = compute_label_targets(
        prices=[100.0 + i for i in range(2500)],
        index=100,
        horizon_steps={"1h": 12, "4h": 48, "1d": 288, "7d": 2016},
        market_state=market,
        liquidity_features=liq,
        turnover_estimate=0.5,
        cost_profile_name="standard",
    )
    c_label = float(labels["cost_1h_bps"])

    costs_action = compute_cost_map(
        horizons=("1h", "4h", "1d", "7d"),
        profile=profile,
        market_state=market,
        liquidity_features=liq,
        turnover_estimate=0.5,
    )
    c_action = float(costs_action["1h"])

    metrics = evaluate_regression_oos(
        y_true=np.array([0.01, -0.01, 0.005, -0.005], dtype=np.float64),
        y_pred=np.array([0.02, -0.01, 0.002, -0.002], dtype=np.float64),
        horizon="1h",
        cost_profile_name="standard",
        liquidity_features={"liquidity_score": 0.8},
    )
    c_eval = float(metrics["cost_bps_used"])

    assert c1 > 0
    assert abs(c1 - c_label) < 1e-6
    assert abs(c1 - c_action) < 1e-6
    assert c_eval > 0
