from __future__ import annotations

from account_state.models import AccountHealth, AccountState, BalanceState
from position_sizer import PositionSizer
from risk_state.models import RiskRegime, RiskState


def _account() -> AccountState:
    return AccountState(
        balances=BalanceState(cash=10000.0, equity=10000.0, free_margin=8000.0, used_margin=2000.0, margin_ratio=5.0),
        health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
    )


def _risk(pos_scale: float = 1.0) -> RiskState:
    return RiskState(
        regime=RiskRegime.GREEN,
        hard_limits_ok=True,
        soft_penalty_factors={"pos_scale": pos_scale, "band_scale": 1.0, "cost_scale": 1.0, "exec_style_bias": "neutral"},
        reason_codes=[],
    )


def test_sigma_increase_shrinks_target():
    sizer = PositionSizer()
    base = {"symbol": "BTC", "mu": {"1h": 0.20}, "sigma": {"1h": 0.05}}
    hi_sigma = {"symbol": "BTC", "mu": {"1h": 0.20}, "sigma": {"1h": 0.25}}
    t1, _, _ = sizer.compute_target_position(base, {"1h": 0.001}, _account(), _risk())
    t2, _, _ = sizer.compute_target_position(hi_sigma, {"1h": 0.001}, _account(), _risk())
    assert abs(t2) < abs(t1)


def test_cost_increase_shrinks_target():
    sizer = PositionSizer()
    pred = {"symbol": "BTC", "mu": {"1h": 0.08}, "sigma": {"1h": 0.20}}
    t1, _, _ = sizer.compute_target_position(pred, {"1h": 0.001}, _account(), _risk())
    t2, _, _ = sizer.compute_target_position(pred, {"1h": 0.060}, _account(), _risk())
    assert abs(t2) < abs(t1)


def test_risk_pos_scale_scales_target():
    sizer = PositionSizer()
    pred = {"symbol": "BTC", "mu": {"1h": 0.20}, "sigma": {"1h": 0.05}}
    t1, _, _ = sizer.compute_target_position(pred, {"1h": 0.001}, _account(), _risk(pos_scale=1.0))
    t2, _, _ = sizer.compute_target_position(pred, {"1h": 0.001}, _account(), _risk(pos_scale=0.5))
    assert abs(t2) <= abs(t1) * 0.55
