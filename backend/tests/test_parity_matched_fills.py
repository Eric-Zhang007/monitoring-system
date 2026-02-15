from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


class _RepoInsufficientTargets:
    def list_recent_backtest_runs(
        self,
        track: str,
        limit: int = 500,
        include_sources=None,
        exclude_sources=None,
        data_regimes=None,
    ):
        now = datetime.now(timezone.utc)
        return [
            {"created_at": now - timedelta(days=1), "metrics": {"status": "completed"}, "config": {"score_source": "model"}, "superseded_by_run_id": None}
            for _ in range(6)
        ]

    def get_backtest_target_pnl_window(self, track: str, window_hours: int, include_sources=None, exclude_sources=None, score_source=None, data_regimes=None):
        return {"BTC": {"sum": 0.1, "count": 2.0}}

    def get_execution_target_realized_window(self, track: str, window_hours: int):
        return {"BTC": {"sum_weighted": 100.0, "sum_notional": 1000.0, "orders": 80.0}}


class _RepoInsufficientOrders:
    def list_recent_backtest_runs(
        self,
        track: str,
        limit: int = 500,
        include_sources=None,
        exclude_sources=None,
        data_regimes=None,
    ):
        now = datetime.now(timezone.utc)
        return [
            {"created_at": now - timedelta(days=1), "metrics": {"status": "completed"}, "config": {"score_source": "model"}, "superseded_by_run_id": None}
            for _ in range(6)
        ]

    def get_backtest_target_pnl_window(self, track: str, window_hours: int, include_sources=None, exclude_sources=None, score_source=None, data_regimes=None):
        return {
            "BTC": {"sum": 0.1, "count": 2.0},
            "ETH": {"sum": 0.08, "count": 2.0},
        }

    def get_execution_target_realized_window(self, track: str, window_hours: int):
        return {
            "BTC": {"sum_weighted": 100.0, "sum_notional": 1000.0, "orders": 10.0},
            "ETH": {"sum_weighted": 80.0, "sum_notional": 1000.0, "orders": 10.0},
        }


def test_parity_insufficient_matched_targets(monkeypatch):
    monkeypatch.setattr(router_mod, "repo", _RepoInsufficientTargets())
    out = router_mod._parity_check(track="liquid", max_deviation=0.1, min_completed_runs=5)
    assert out["status"] == "insufficient_observation"
    assert out["reason"] == "insufficient_matched_targets"


def test_parity_insufficient_paper_orders(monkeypatch):
    monkeypatch.setattr(router_mod, "repo", _RepoInsufficientOrders())
    out = router_mod._parity_check(track="liquid", max_deviation=0.1, min_completed_runs=5)
    assert out["status"] == "insufficient_observation"
    assert out["reason"] == "insufficient_paper_orders"
