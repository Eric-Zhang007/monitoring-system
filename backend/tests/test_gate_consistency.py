from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402
from schemas_v2 import BacktestRunRequest  # noqa: E402


def test_backtest_gate_passed_field_reuses_evaluate_gate_result(monkeypatch):
    class _FakeRepo:
        def __init__(self):
            self.last_metrics = None
            self.promote_passed = None

        def create_backtest_run(self, **kwargs):
            return 99

        def load_feature_history(self, **kwargs):
            return [{"feature_payload": {"ret_1": 0.0}, "as_of_ts": "2025-01-01T00:00:00+00:00"}]

        def load_price_history(self, *args, **kwargs):
            return [{"price": 100.0, "timestamp": "2025-01-01T00:00:00+00:00"}]

        def finish_backtest_run(self, run_id: int, metrics):
            self.last_metrics = dict(metrics)

        def promote_model(self, **kwargs):
            self.promote_passed = bool(kwargs.get("passed"))

        def get_active_model_state(self, track: str):
            return None

        def upsert_active_model_state(self, **kwargs):
            return None

    fake_repo = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake_repo)
    monkeypatch.setattr(
        router_mod,
        "_run_model_replay_backtest",
        lambda **kwargs: {
            "status": "completed",
            "reason": "ok",
            "samples": 100,
            "ic": 0.1,
            "hit_rate": 0.6,
            "turnover": 0.2,
            "pnl_after_cost": 0.03,
            "max_drawdown": 0.05,
            "lineage_coverage": 1.0,
            "alignment_audit": {"alignment_mode_applied": "strict_asof", "alignment_version": "strict_asof_v1"},
            "leakage_checks": {"passed": True, "leakage_violations": 0, "alignment_mode": "strict_asof"},
            "cost_breakdown": {"fee": 0.0, "slippage": 0.0, "impact": 0.0},
            "model_inference_coverage": 0.0,
            "fallback_used": False,
        },
    )
    monkeypatch.setattr(router_mod, "_evaluate_gate", lambda **kwargs: (True, "passed", {"ic": 0.1, "pnl_after_cost": 0.02, "max_drawdown": 0.05}, 3))

    resp = asyncio.run(
        router_mod.run_backtest(
            BacktestRunRequest(
                track="liquid",
                targets=["BTC"],
                score_source="heuristic",
                require_model_artifact=False,
                lookback_days=90,
                train_days=35,
                test_days=7,
            )
        )
    )
    assert resp.metrics["gate_passed"] is True
    assert fake_repo.promote_passed is True
    assert fake_repo.last_metrics is not None
    assert fake_repo.last_metrics["gate_passed"] is True
