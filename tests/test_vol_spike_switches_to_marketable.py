from __future__ import annotations

from account_state.models import ExecutionStats
from execution_style_selector import ExecutionStyleSelector
from risk_state.models import RiskRegime, RiskState


def test_vol_spike_high_signal_switches_to_marketable_limit():
    selector = ExecutionStyleSelector()
    risk = RiskState(
        regime=RiskRegime.GREEN,
        hard_limits_ok=True,
        soft_penalty_factors={"pos_scale": 1.0, "band_scale": 1.0, "cost_scale": 1.0, "exec_style_bias": "neutral"},
        reason_codes=[],
    )
    order_intent = {"score": 1.2, "qty": 0.5, "horizon_seconds": 3600}
    style = selector.select_style(
        order_intent=order_intent,
        risk_state=risk,
        market_snapshot={"realized_vol": 0.2, "vol_spike": True},
        exec_stats=ExecutionStats(slippage_bps_p90=5.0),
    )
    assert style is not None
    assert str(style["style"]) == "marketable_limit"
    assert int(style["n_slices"]) == 1

