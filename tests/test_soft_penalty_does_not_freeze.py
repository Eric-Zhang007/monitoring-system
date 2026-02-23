from __future__ import annotations

from account_state.models import AccountHealth, AccountState, BalanceState, ExecutionStats
from risk_manager import RiskManager


def test_soft_penalty_slippage_high_not_freeze():
    account = AccountState(
        balances=BalanceState(cash=5000.0, equity=5000.0, free_margin=4000.0, used_margin=1000.0, margin_ratio=2.0),
        execution_stats=ExecutionStats(slippage_bps_p90=28.0, reject_rate_5m=0.05),
        health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
    )
    risk = RiskManager().evaluate(account=account, symbol="ETH", order_intent={}, market_snapshot={"realized_vol": 0.03})
    assert str(risk.regime.value) == "YELLOW"
    assert bool(risk.hard_limits_ok) is True
    assert float(risk.soft_penalty_factors["pos_scale"]) < 1.0

