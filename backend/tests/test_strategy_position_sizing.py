from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_score_to_size_non_linear_and_cost_aware(monkeypatch):
    monkeypatch.setenv("SIGNAL_ENTRY_Z_MIN", "0.02")
    monkeypatch.setenv("POSITION_MAX_WEIGHT_BASE", "0.2")
    cfg = router_mod._position_sizing_settings()
    low = router_mod._score_to_size(score=0.01, confidence=0.8, est_cost_bps=8.0, vol_bucket="normal", sizing_cfg=cfg)
    mid = router_mod._score_to_size(score=0.05, confidence=0.8, est_cost_bps=8.0, vol_bucket="normal", sizing_cfg=cfg)
    high_cost = router_mod._score_to_size(score=0.05, confidence=0.8, est_cost_bps=80.0, vol_bucket="normal", sizing_cfg=cfg)
    high_vol = router_mod._score_to_size(score=0.05, confidence=0.8, est_cost_bps=8.0, vol_bucket="high", sizing_cfg=cfg)
    assert low == 0.0
    assert mid > 0.0
    assert high_cost < mid
    assert high_vol < mid


def test_score_to_size_is_deterministic_given_same_cfg(monkeypatch):
    monkeypatch.setenv("SIGNAL_ENTRY_Z_MIN", "0.5")
    monkeypatch.setenv("POSITION_MAX_WEIGHT_BASE", "0.01")
    explicit_cfg = {
        "entry_z": 0.02,
        "exit_z": 0.008,
        "max_weight_base": 0.2,
        "high_vol_mult": 0.65,
        "cost_lambda": 1.2,
    }
    a = router_mod._score_to_size(score=0.05, confidence=0.8, est_cost_bps=8.0, vol_bucket="normal", sizing_cfg=explicit_cfg)
    b = router_mod._score_to_size(score=0.05, confidence=0.8, est_cost_bps=8.0, vol_bucket="normal", sizing_cfg=explicit_cfg)
    assert abs(a - b) < 1e-12


def test_drawdown_throttle_shrinks_single_weight_limit():
    proposed = [router_mod.RebalancePosition(target="BTC", track="liquid", weight=0.2)]
    adjusted, violations, _, _ = router_mod._evaluate_risk(proposed=proposed, current=[], realized_drawdown=0.11)
    assert any(v.startswith("single_weight_exceeded:") for v in violations)
    assert abs(float(adjusted[0].weight)) < 0.2
