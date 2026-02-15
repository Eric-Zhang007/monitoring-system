from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402
from schemas_v2 import RollbackCheckRequest, RolloutAdvanceRequest  # noqa: E402


class _FakeRepo:
    def __init__(self):
        self.rollout_calls = []

    def list_recent_backtest_runs(self, track: str, limit: int = 10):
        return [
            {"metrics": {"status": "completed", "hit_rate": 0.2, "max_drawdown": 0.4, "pnl_after_cost": -0.1}},
            {"metrics": {"status": "completed", "hit_rate": 0.3, "max_drawdown": 0.35, "pnl_after_cost": -0.05}},
            {"metrics": {"status": "completed", "hit_rate": 0.6, "max_drawdown": 0.1, "pnl_after_cost": 0.02}},
        ]

    def get_active_model_state(self, track: str):
        return {
            "active_model_name": "liquid_ttm_ensemble",
            "active_model_version": "v2.1",
            "previous_model_name": "liquid_ttm_ensemble",
            "previous_model_version": "v2.0",
        }

    def upsert_active_model_state(self, **kwargs):
        return None

    def save_risk_event(self, **kwargs):
        return None

    def upsert_model_rollout_state(self, **kwargs):
        self.rollout_calls.append(kwargs)
        return None


def test_rollback_check_returns_consecutive_windows_fields(monkeypatch):
    monkeypatch.setenv("ROLLBACK_CONSECUTIVE_WINDOWS", "2")
    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    req = RollbackCheckRequest(track="liquid", model_name="liquid_ttm_ensemble", model_version="v2.1", max_recent_losses=3)
    resp = asyncio.run(router_mod.check_model_rollback(req))
    assert resp.windows_failed >= 2
    assert resp.trigger_rule.startswith("consecutive_windows>=")


def test_rollout_advance_rejects_invalid_step(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    req = RolloutAdvanceRequest(
        track="liquid",
        model_name="liquid_ttm_ensemble",
        model_version="v2.1",
        current_stage_pct=10,
        next_stage_pct=100,
        windows=1,
    )
    resp = asyncio.run(router_mod.advance_model_rollout(req))
    assert resp.promoted is False
    assert resp.reason == "invalid_rollout_step"
