from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


class _FakeRepo:
    def __init__(self, passed: bool):
        now = datetime.now(timezone.utc)
        pnl = 0.02
        self.runs = [
            {"created_at": now - timedelta(days=2), "metrics": {"status": "completed", "pnl_after_cost": pnl}, "config": {"score_source": "model"}, "superseded_by_run_id": None},
            {"created_at": now - timedelta(days=5), "metrics": {"status": "completed", "pnl_after_cost": pnl}, "config": {"score_source": "model"}, "superseded_by_run_id": None},
            {"created_at": now - timedelta(days=10), "metrics": {"status": "completed", "pnl_after_cost": pnl}, "config": {"score_source": "model"}, "superseded_by_run_id": None},
            {"created_at": now - timedelta(days=20), "metrics": {"status": "completed", "pnl_after_cost": pnl}, "config": {"score_source": "model"}, "superseded_by_run_id": None},
            {"created_at": now - timedelta(days=25), "metrics": {"status": "completed", "pnl_after_cost": pnl}, "config": {"score_source": "model"}, "superseded_by_run_id": None},
        ]
        self.passed = passed
        self.last_include = None
        self.last_exclude = None

    def list_recent_backtest_runs(
        self,
        track: str,
        limit: int = 500,
        include_sources=None,
        exclude_sources=None,
        data_regimes=None,
    ):
        self.last_include = include_sources
        self.last_exclude = exclude_sources
        return self.runs[:limit]

    def get_backtest_target_pnl_window(
        self,
        track: str,
        window_hours: int,
        include_sources=None,
        exclude_sources=None,
        score_source=None,
        data_regimes=None,
    ):
        if self.passed:
            return {"BTC": {"sum": 0.1, "count": 1.0}, "ETH": {"sum": 0.08, "count": 1.0}}
        return {"BTC": {"sum": 0.12, "count": 1.0}, "ETH": {"sum": 0.1, "count": 1.0}}

    def get_execution_target_realized_window(self, track: str, window_hours: int):
        if self.passed:
            return {
                "BTC": {"sum_weighted": 100.0, "sum_notional": 1000.0, "orders": 60.0},
                "ETH": {"sum_weighted": 80.0, "sum_notional": 1000.0, "orders": 60.0},
            }
        return {
            "BTC": {"sum_weighted": -20.0, "sum_notional": 1000.0, "orders": 60.0},
            "ETH": {"sum_weighted": -30.0, "sum_notional": 1000.0, "orders": 60.0},
        }


def test_parity_check_pass(monkeypatch):
    fake = _FakeRepo(passed=True)
    monkeypatch.setattr(router_mod, "repo", fake)
    out = router_mod._parity_check(track="liquid", max_deviation=0.10, min_completed_runs=5)
    assert out["status"] == "passed"
    assert out["passed"] is True
    assert out["comparison_basis"] == "matched_filled_orders"
    assert fake.last_include == ["prod"]
    assert fake.last_exclude == ["smoke", "async_test", "maintenance"]


def test_parity_check_fail(monkeypatch):
    monkeypatch.setattr(router_mod, "repo", _FakeRepo(passed=False))
    out = router_mod._parity_check(track="liquid", max_deviation=0.10, min_completed_runs=5)
    assert out["status"] == "failed"
    assert out["passed"] is False
