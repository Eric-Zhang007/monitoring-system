from __future__ import annotations

from account_state.models import AccountHealth, AccountState, BalanceState, ExecutionStats
from risk_manager import RiskManager


def _healthy_account() -> AccountState:
    return AccountState(
        balances=BalanceState(cash=10000.0, equity=10000.0, free_margin=6000.0, used_margin=4000.0, margin_ratio=2.5),
        execution_stats=ExecutionStats(slippage_bps_p50=3.0, slippage_bps_p90=6.0, reject_rate_5m=0.01),
        health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
    )


def test_high_slippage_triggers_soft_penalty_not_freeze():
    account = _healthy_account()
    account.execution_stats.slippage_bps_p90 = 25.0
    risk = RiskManager().evaluate(account=account, symbol="BTC", order_intent={}, market_snapshot={})
    assert str(risk.regime.value) == "YELLOW"
    assert bool(risk.hard_limits_ok) is True
    assert float(risk.soft_penalty_factors["pos_scale"]) < 1.0
    assert float(risk.soft_penalty_factors["cost_scale"]) > 1.0
    assert str(risk.soft_penalty_factors["exec_style_bias"]) == "passive"


def test_stale_account_forces_red():
    account = _healthy_account()
    account.health.is_fresh = False
    risk = RiskManager().evaluate(account=account, symbol="BTC", order_intent={}, market_snapshot={})
    assert str(risk.regime.value) == "RED"
    assert bool(risk.hard_limits_ok) is False
    assert "account_state_stale" in list(risk.reason_codes)

