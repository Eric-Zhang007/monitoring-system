from __future__ import annotations

import asyncio

import v2_router as router_mod
from account_state.models import AccountHealth, AccountState, BalanceState, ExecutionStats
from schemas_v2 import SignalGenerateRequest


def _pred(symbol: str, horizon: str):
    _ = horizon
    return {
        "target": symbol.upper(),
        "score": 0.25,
        "confidence": 0.95,
        "outputs": {
            "expected_return": 0.25,
            "signal_confidence": 0.95,
            "vol_forecast": 0.05,
            "expected_return_horizons": {"1h": 0.25, "4h": 0.25, "1d": 0.25, "7d": 0.25},
            "signal_confidence_horizons": {"1h": 0.95, "4h": 0.95, "1d": 0.95, "7d": 0.95},
            "vol_forecast_horizons": {"1h": 0.05, "4h": 0.05, "1d": 0.05, "7d": 0.05},
            "stack": {"coverage_summary": {"missing_ratio": 0.0, "observed_ratio": 1.0}},
            "degraded": False,
            "degraded_reasons": [],
        },
        "explanation": {},
        "model_name": "liquid_main",
        "model_version": "main",
    }


class _Repo:
    def __init__(self):
        self.last_signal_meta = {}
        self.last_trace = {}

    def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
        _ = (track, strategy_id)
        return False

    def insert_signal_candidate(self, **kwargs):
        self.last_signal_meta = dict(kwargs.get("metadata") or {})
        return 1

    def insert_decision_trace(self, payload):
        self.last_trace = dict(payload or {})
        return 1


def _stale_account() -> AccountState:
    return AccountState(
        balances=BalanceState(cash=1000.0, equity=1000.0, free_margin=500.0, used_margin=500.0, margin_ratio=2.0),
        execution_stats=ExecutionStats(slippage_bps_p90=8.0, reject_rate_5m=0.01),
        health=AccountHealth(is_fresh=False, recon_ok=True, ws_ok=True, last_error="stale"),
    )


def _healthy_high_slip() -> AccountState:
    return AccountState(
        balances=BalanceState(cash=1000.0, equity=1000.0, free_margin=700.0, used_margin=300.0, margin_ratio=2.0),
        execution_stats=ExecutionStats(slippage_bps_p90=30.0, reject_rate_5m=0.05),
        health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
    )


def test_signal_pipeline_blocks_on_stale_account(monkeypatch):
    fake_repo = _Repo()
    monkeypatch.setattr(router_mod, "repo", fake_repo)
    monkeypatch.setattr(router_mod, "_build_liquid_prediction", _pred)
    monkeypatch.setattr(router_mod, "_strategy_bucket", lambda *args, **kwargs: "trend")
    monkeypatch.setattr(router_mod, "_get_account_state_for_signal", _stale_account)

    resp = asyncio.run(
        router_mod.generate_signal(
            SignalGenerateRequest(track="liquid", target="BTC", horizon="1h", min_confidence=0.1, policy="baseline-v2")
        )
    )
    assert str(resp.action) == "hold"
    assert str((fake_repo.last_signal_meta.get("risk_state") or {}).get("regime")) == "RED"


def test_signal_pipeline_high_slippage_uses_soft_penalty_and_passive(monkeypatch):
    fake_repo = _Repo()
    monkeypatch.setattr(router_mod, "repo", fake_repo)
    monkeypatch.setattr(router_mod, "_build_liquid_prediction", _pred)
    monkeypatch.setattr(router_mod, "_strategy_bucket", lambda *args, **kwargs: "trend")
    monkeypatch.setattr(router_mod, "_get_account_state_for_signal", _healthy_high_slip)

    resp = asyncio.run(
        router_mod.generate_signal(
            SignalGenerateRequest(track="liquid", target="BTC", horizon="1h", min_confidence=0.1, policy="baseline-v2")
        )
    )
    assert str((fake_repo.last_signal_meta.get("risk_state") or {}).get("regime")) == "YELLOW"
    assert str((fake_repo.last_signal_meta.get("execution_style") or {}).get("style")) == "passive_twap"
    assert str(resp.action) in {"buy", "sell", "hold"}

