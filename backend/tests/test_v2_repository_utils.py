from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from v2_repository import V2Repository  # noqa: E402


def test_feature_payload_diff_numeric():
    a = {"ret_1": 0.01, "ret_3": -0.02, "bucket": 2}
    b = {"ret_1": 0.011, "ret_3": -0.018, "bucket": 2}
    max_abs, mean_abs = V2Repository._feature_payload_diff(a, b)
    assert max_abs > 0
    assert mean_abs > 0
    assert max_abs >= mean_abs


def test_feature_payload_diff_mixed_types():
    a = {"a": 1.0, "tag": "x"}
    b = {"a": 1.0, "tag": "y"}
    max_abs, mean_abs = V2Repository._feature_payload_diff(a, b)
    assert max_abs == 1.0
    assert mean_abs == 0.5


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed_sql = ""
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed_sql = str(sql)
        self.executed_params = params

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self._cursor


def test_get_execution_slippage_samples_uses_filled_statuses_only():
    rows = [{"slippage": 0.0012}, {"slippage": None}, {"slippage": -0.0004}]
    cur = _FakeCursor(rows)
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]

    out = repo.get_execution_slippage_samples(track="liquid", lookback_hours=24)
    assert out == [0.0012, -0.0004]
    assert "partially_filled" in cur.executed_sql
    assert "submitted" not in cur.executed_sql


def test_check_feature_lineage_consistency_groups_by_target_when_target_absent():
    now = datetime.now(timezone.utc)
    rows = [
        {"id": 10, "target": "BTC", "track": "liquid", "lineage_id": "ln1", "feature_payload": {"x": 1.0}, "created_at": now},
        {"id": 9, "target": "BTC", "track": "liquid", "lineage_id": "ln1", "feature_payload": {"x": 1.02}, "created_at": now},
        {"id": 8, "target": "ETH", "track": "liquid", "lineage_id": "ln1", "feature_payload": {"x": 5.0}, "created_at": now},
    ]
    cur = _FakeCursor(rows)
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]

    res = repo.check_feature_lineage_consistency(track="liquid", lineage_id="ln1", target=None, tolerance=0.001)
    assert res["reason"] == "payload_mismatch"
    assert res["compared_snapshots"] == 2
    assert res["max_abs_diff"] > 0.001
    assert "x" in res.get("mismatch_keys", [])


def test_get_execution_daily_loss_ratio_uses_edge_pnl_over_notional():
    repo = object.__new__(V2Repository)
    repo.get_execution_edge_pnls = lambda track, lookback_hours, limit=5000, strategy_id=None: [  # type: ignore[assignment]
        {"edge_pnl": -10.0, "notional": 1000.0},
        {"edge_pnl": -5.0, "notional": 500.0},
    ]
    ratio = repo.get_execution_daily_loss_ratio(track="liquid", lookback_hours=24)
    assert abs(ratio - 0.01) < 1e-9


def test_get_execution_consecutive_losses_stops_on_non_loss():
    repo = object.__new__(V2Repository)
    repo.get_execution_edge_pnls = lambda track, lookback_hours, limit=200, strategy_id=None: [  # type: ignore[assignment]
        {"edge_pnl": -1.0, "notional": 100.0},
        {"edge_pnl": -0.2, "notional": 80.0},
        {"edge_pnl": 0.0, "notional": 70.0},
        {"edge_pnl": -3.0, "notional": 120.0},
    ]
    streak = repo.get_execution_consecutive_losses(track="liquid", lookback_hours=24, limit=200)
    assert streak == 2
