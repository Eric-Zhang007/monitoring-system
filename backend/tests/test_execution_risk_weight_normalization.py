from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_order_to_risk_weight_prefers_weight_delta_over_price_and_qty():
    order_low_px = {
        "target": "BTC",
        "side": "buy",
        "quantity": 0.01,
        "est_price": 1000.0,
        "metadata": {"weight_delta": 0.03},
    }
    order_high_px = {
        "target": "BTC",
        "side": "buy",
        "quantity": 0.01,
        "est_price": 100000.0,
        "metadata": {"weight_delta": 0.03},
    }
    w1 = router_mod._order_to_risk_weight(order_low_px, risk_equity_usd=10000.0)
    w2 = router_mod._order_to_risk_weight(order_high_px, risk_equity_usd=10000.0)
    assert w1 == 0.03
    assert w2 == 0.03


def test_order_to_risk_weight_uses_notional_over_equity_when_no_weight_delta():
    order = {
        "target": "ETH",
        "side": "sell",
        "quantity": 2.0,
        "est_price": 2500.0,
        "metadata": {},
    }
    weight = router_mod._order_to_risk_weight(order, risk_equity_usd=100000.0)
    assert weight is not None
    assert abs(float(weight) - (-5000.0 / 100000.0)) < 1e-12
    assert abs(float(weight) - float(order["quantity"])) > 1e-9


def test_infer_execution_risk_positions_does_not_treat_qty_as_weight():
    orders = [
        {
            "target": "BTC",
            "track": "liquid",
            "side": "buy",
            "quantity": 3.0,
            "est_price": 100.0,
            "metadata": {},
        }
    ]
    out = router_mod._infer_execution_risk_positions(orders, risk_equity_usd=10000.0)
    assert len(out) == 1
    assert abs(out[0].weight - 0.03) < 1e-12
    assert abs(out[0].weight - 3.0) > 1e-9
