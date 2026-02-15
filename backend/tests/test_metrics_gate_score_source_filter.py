from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


class _FakeRepo:
    def __init__(self):
        self.last_data_regimes = None

    def list_recent_backtest_runs(self, track: str, limit: int = 500, include_sources=None, exclude_sources=None, data_regimes=None):
        self.last_data_regimes = data_regimes
        now = datetime.now(timezone.utc)
        return [
            {
                "created_at": now - timedelta(days=1),
                "metrics": {"status": "completed", "pnl_after_cost": 0.01},
                "config": {"score_source": "heuristic"},
                "superseded_by_run_id": None,
            },
            {
                "created_at": now - timedelta(days=2),
                "metrics": {"status": "completed", "pnl_after_cost": 0.01},
                "config": {"score_source": "model"},
                "superseded_by_run_id": None,
            },
        ]

    def get_backtest_target_pnl_window(self, track: str, window_hours: int, include_sources=None, exclude_sources=None, score_source=None, data_regimes=None):
        if score_source == "model":
            return {"BTC": {"sum": 0.1, "count": 1.0}, "ETH": {"sum": 0.08, "count": 1.0}}
        return {"BTC": {"sum": 0.1, "count": 10.0}, "ETH": {"sum": 0.1, "count": 10.0}}

    def get_execution_target_realized_window(self, track: str, window_hours: int):
        return {
            "BTC": {"sum_weighted": 100.0, "sum_notional": 1000.0, "orders": 60.0},
            "ETH": {"sum_weighted": 80.0, "sum_notional": 1000.0, "orders": 60.0},
        }


def test_parity_counts_only_requested_score_source(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(router_mod, "repo", fake)
    out = router_mod._parity_check(track="liquid", max_deviation=0.1, min_completed_runs=2, score_source="model")
    assert out["status"] == "insufficient_observation"
    assert out["reason"] == "insufficient_completed_backtests"
    assert out["score_source"] == "model"
    assert out["data_regimes"] == ["prod_live"]
    assert fake.last_data_regimes == ["prod_live"]
