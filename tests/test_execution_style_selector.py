from __future__ import annotations

from account_state.models import ExecutionStats
from execution_style_selector import ExecutionStyleSelector
from risk_state.models import RiskRegime, RiskState


def test_slippage_high_biases_to_passive_twap():
    selector = ExecutionStyleSelector()
    risk = RiskState(
        regime=RiskRegime.YELLOW,
        hard_limits_ok=True,
        soft_penalty_factors={"pos_scale": 0.7, "band_scale": 1.0, "cost_scale": 1.3, "exec_style_bias": "passive"},
        reason_codes=["slippage_p90_high"],
    )
    style = selector.select_style(
        order_intent={"score": 0.4, "qty": 0.5, "horizon_seconds": 24 * 3600},
        risk_state=risk,
        market_snapshot={"realized_vol": 0.03, "adv_qty": 1000.0},
        exec_stats=ExecutionStats(slippage_bps_p90=22.0),
    )
    assert style is not None
    assert str(style["style"]) == "passive_twap"
    assert int(style["n_slices"]) >= 2

