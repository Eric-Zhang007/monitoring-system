from __future__ import annotations

from account_state.models import AccountState, BalanceState, ExecutionStats, PositionState


def test_account_state_contract_fields_and_defaults():
    s = AccountState()
    dumped = s.model_dump()
    for key in ("ts", "balances", "positions", "open_orders", "execution_stats", "health"):
        assert key in dumped
    assert s.health.is_fresh is False
    assert s.health.recon_ok is False
    assert s.balances.account_currency == "USD"


def test_account_state_serialization_roundtrip():
    s = AccountState(
        venue="bitget",
        adapter="bitget_live",
        balances=BalanceState(cash=100.0, equity=120.0, free_margin=80.0, used_margin=40.0, margin_ratio=2.0),
        positions={"BTCUSDT": PositionState(symbol="BTCUSDT", qty=0.1, avg_cost=60000.0, unrealized_pnl=12.0)},
        execution_stats=ExecutionStats(slippage_bps_p50=4.0, slippage_bps_p90=12.0, reject_rate_5m=0.05),
    )
    payload = s.model_dump(mode="json")
    restored = AccountState.model_validate(payload)
    assert restored.venue == "bitget"
    assert "BTCUSDT" in restored.positions
    assert float(restored.execution_stats.slippage_bps_p90) == 12.0
