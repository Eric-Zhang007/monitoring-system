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

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows[0]


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


def test_load_feature_history_dynamic_limit_scales_with_window(monkeypatch):
    rows = [{"target": "BTC", "track": "liquid", "as_of_ts": datetime.now(timezone.utc), "feature_payload": {}}]
    cur = _FakeCursor(rows)
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]

    monkeypatch.setenv("FEATURE_SNAPSHOT_ROWS_PER_DAY", "288")
    monkeypatch.setenv("FEATURE_HISTORY_MAX_LIMIT", "300000")
    out = repo.load_feature_history(target="BTC", track="liquid", lookback_days=90, data_version=None, limit=None)
    assert out == rows
    assert cur.executed_params is not None
    # last SQL param is LIMIT
    assert int(cur.executed_params[-1]) >= 26000


def test_get_ops_control_state_queries_table():
    cur = _FakeCursor(
        [
            {
                "control_key": "live_control_state",
                "payload": {"live_enabled": True},
                "source": "api",
                "updated_by": "manual",
                "updated_at": datetime.now(timezone.utc),
            }
        ]
    )
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]
    row = repo.get_ops_control_state("live_control_state")
    assert row is not None
    assert row["control_key"] == "live_control_state"
    assert "ops_control_state" in cur.executed_sql


def test_upsert_ops_control_state_writes_payload_json():
    cur = _FakeCursor(
        [
            {
                "control_key": "manual_candidate",
                "payload": {"track": "liquid"},
                "source": "api",
                "updated_by": "manual",
                "updated_at": datetime.now(timezone.utc),
            }
        ]
    )
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]
    row = repo.upsert_ops_control_state("manual_candidate", {"track": "liquid"}, updated_by="manual")
    assert row["control_key"] == "manual_candidate"
    assert cur.executed_params is not None
    assert cur.executed_params[0] == "manual_candidate"
    assert "INSERT INTO ops_control_state" in cur.executed_sql


def test_normalize_timeframes_filters_invalid_and_duplicates():
    out = V2Repository._normalize_timeframes("5m,1h,foo,1h,4h, 15m ,1x")
    assert out == ["5m", "1h", "4h", "15m"]


class _ContextCursor:
    def __init__(self):
        self.executed_sql = ""
        self.executed_params = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.executed_sql = str(sql)
        self.executed_params = params
        text = self.executed_sql.lower()
        if "to_regclass" in text:
            self._rows = [{"reg": None}]
            return
        if "from market_bars" in text:
            tf = str((params or [None, ""])[1]).lower()
            if tf == "5m":
                now = datetime.now(timezone.utc)
                self._rows = [
                    {"ts": now, "close": 101.0, "volume": 10.0},
                    {"ts": now, "close": 100.0, "volume": 9.0},
                ]
            else:
                self._rows = []
            return
        self._rows = []

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        if not self._rows:
            return None
        return self._rows[0]


def test_load_multi_timeframe_context_ondemand_marks_missing():
    cur = _ContextCursor()
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]

    out = repo.load_multi_timeframe_context(
        symbol="BTC",
        as_of=datetime.now(timezone.utc),
        timeframes=["5m", "1h"],
        primary_timeframe="5m",
    )
    assert out["source"] == "market_bars_ondemand"
    assert "5m" in out["context"]
    assert "1h" in out["context"]
    assert int(out["context"]["5m"]["missing"]) == 0
    assert int(out["context"]["1h"]["missing"]) == 1


class _UpsertCursor:
    def __init__(self):
        self.sqls = []
        self.params = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self.sqls.append(str(sql))
        self.params.append(params)

    def fetchone(self):
        return {
            "track": "liquid",
            "as_of": datetime.now(timezone.utc),
            "universe_version": "abc",
            "source": "db",
            "symbols_json": {"symbols": ["BTC", "ETH"]},
        }


def test_upsert_universe_snapshot_ensures_unique_index_before_conflict():
    cur = _UpsertCursor()
    conn = _FakeConn(cur)
    repo = object.__new__(V2Repository)
    repo._connect = lambda: conn  # type: ignore[assignment]

    row = repo.upsert_asset_universe_snapshot(
        track="liquid",
        as_of=datetime.now(timezone.utc),
        symbols=["BTC", "ETH"],
        universe_version="abc",
        source="db",
    )
    joined = "\n".join(cur.sqls)
    assert "CREATE UNIQUE INDEX IF NOT EXISTS idx_asset_universe_snapshots_track_asof_version" in joined
    assert "ON CONFLICT (track, as_of, universe_version)" in joined
    assert row["track"] == "liquid"
