from __future__ import annotations

from cost.cost_profile import compute_cost_bps, estimate_turnover, load_cost_profile
from training.labels.liquid_labels import compute_label_targets


def test_estimate_turnover_dynamic_regime_and_horizon():
    market = {"realized_vol": 0.03, "funding_rate": 0.0003}
    liq_good = {"liquidity_score": 0.9, "spread_bps": 4.0, "orderbook_depth_total": 3_000_000.0}
    liq_bad = {"liquidity_score": 0.2, "spread_bps": 20.0, "orderbook_depth_total": 120_000.0}

    t_1h_good = estimate_turnover(horizon="1h", market_state=market, liquidity_features=liq_good, account_state={"turnover_estimate": 0.35})
    t_7d_good = estimate_turnover(horizon="7d", market_state=market, liquidity_features=liq_good, account_state={"turnover_estimate": 0.35})
    t_1h_bad = estimate_turnover(horizon="1h", market_state=market, liquidity_features=liq_bad, account_state={"turnover_estimate": 0.35})

    assert t_1h_good > t_7d_good
    assert t_1h_bad > t_1h_good


def test_cost_bps_none_turnover_equals_explicit_estimated_turnover():
    profile = load_cost_profile("standard")
    market = {"realized_vol": 0.025, "funding_rate": 0.0002, "notional_usd": 50_000.0}
    liq = {"liquidity_score": 0.65, "spread_bps": 7.0, "orderbook_depth_total": 1_500_000.0}
    acct = {"turnover_estimate": 0.4, "notional_usd": 50_000.0}

    est = estimate_turnover(horizon="1h", market_state=market, liquidity_features=liq, account_state=acct)
    c_none = compute_cost_bps(
        horizon="1h",
        profile=profile,
        market_state=market,
        liquidity_features=liq,
        account_state=acct,
        turnover_estimate=None,
    )
    c_explicit = compute_cost_bps(
        horizon="1h",
        profile=profile,
        market_state=market,
        liquidity_features=liq,
        account_state=acct,
        turnover_estimate=est,
    )
    assert abs(float(c_none) - float(c_explicit)) < 1e-9


def test_label_targets_dynamic_turnover_parity_with_cost_module():
    market = {"realized_vol": 0.02, "funding_rate": 0.0001, "notional_usd": 80_000.0}
    liq = {"liquidity_score": 0.7, "orderbook_depth_total": 2_500_000.0, "spread_bps": 5.0}
    acct = {"turnover_estimate": 0.3, "notional_usd": 80_000.0}
    prices = [100.0 + 0.1 * i for i in range(2500)]
    idx = 100
    horizon_steps = {"1h": 12, "4h": 48, "1d": 288, "7d": 2016}

    labels = compute_label_targets(
        prices=prices,
        index=idx,
        horizon_steps=horizon_steps,
        market_state=market,
        liquidity_features=liq,
        account_state=acct,
        turnover_estimate=None,
        cost_profile_name="standard",
    )
    c1 = compute_cost_bps(
        horizon="1h",
        profile=load_cost_profile("standard"),
        market_state=market,
        liquidity_features=liq,
        account_state=acct,
        turnover_estimate=None,
    )
    assert abs(float(labels["cost_1h_bps"]) - float(c1)) < 1e-9
