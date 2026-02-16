from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def _ts(hours: int) -> datetime:
    return datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=hours)


def test_strict_asof_alignment_does_not_use_future_feature():
    prices = [
        {"price": 100.0, "volume": 1000.0, "timestamp": _ts(0)},
        {"price": 101.0, "volume": 1100.0, "timestamp": _ts(1)},
        {"price": 99.0, "volume": 1200.0, "timestamp": _ts(2)},
        {"price": 102.0, "volume": 1300.0, "timestamp": _ts(3)},
        {"price": 101.0, "volume": 1400.0, "timestamp": _ts(4)},
        {"price": 103.0, "volume": 1500.0, "timestamp": _ts(5)},
        {"price": 102.0, "volume": 1600.0, "timestamp": _ts(6)},
        {"price": 104.0, "volume": 1700.0, "timestamp": _ts(7)},
        {"price": 103.0, "volume": 1800.0, "timestamp": _ts(8)},
        {"price": 105.0, "volume": 1900.0, "timestamp": _ts(9)},
    ]
    features = [
        {
            "as_of_ts": _ts(0),
            "feature_payload": {"ret_12": 0.001, "vol_12": 0.01, "vol_48": 0.01},
            "lineage_id": "ln0",
        },
        {
            "as_of_ts": _ts(1) + timedelta(minutes=30),  # future for decision at t=1h
            "feature_payload": {"ret_12": -0.001, "vol_12": 0.01, "vol_48": 0.01},
            "lineage_id": "ln1",
        },
    ]
    aligned = router_mod._align_feature_price_rows(
        feature_rows=features,
        price_rows=prices,
        alignment_mode="strict_asof",
        max_feature_staleness_hours=24 * 14,
        alignment_version="strict_asof_v1",
    )
    assert aligned["feature_indices"][:3] == [0, 0, 1]
    out = router_mod._run_model_replay_backtest(
        feature_rows=features,
        price_rows=prices,
        fee_bps=5.0,
        slippage_bps=3.0,
        raw_series_override=np.array([1.0, -1.0], dtype=np.float64),
        alignment_mode="strict_asof",
        alignment_version="strict_asof_v1",
    )
    assert out["status"] == "completed"
    assert out["leakage_checks"]["passed"] is True
    assert out["leakage_checks"]["leakage_violations"] == 0
    assert out["alignment_audit"]["alignment_mode_applied"] == "strict_asof"


def test_strict_asof_without_timestamps_falls_back_and_fails_leakage_gate():
    features = [{"feature_payload": {"ret_12": 0.001, "vol_12": 0.01, "vol_48": 0.01}, "lineage_id": "ln0"} for _ in range(30)]
    prices = [{"price": 100.0 + i * 0.1, "volume": 1000.0 + i} for i in range(31)]
    out = router_mod._run_model_replay_backtest(
        feature_rows=features,
        price_rows=prices,
        fee_bps=5.0,
        slippage_bps=3.0,
        alignment_mode="strict_asof",
        alignment_version="strict_asof_v1",
    )
    assert out["status"] == "completed"
    assert "fallback" in str(out["alignment_audit"]["alignment_mode_applied"])
    assert out["leakage_checks"]["passed"] is False


def test_strict_asof_fail_fast_blocks_legacy_fallback():
    features = [{"feature_payload": {"ret_12": 0.001, "vol_12": 0.01, "vol_48": 0.01}, "lineage_id": "ln0"} for _ in range(30)]
    prices = [{"price": 100.0 + i * 0.1, "volume": 1000.0 + i} for i in range(31)]
    out = router_mod._run_model_replay_backtest(
        feature_rows=features,
        price_rows=prices,
        fee_bps=5.0,
        slippage_bps=3.0,
        alignment_mode="strict_asof",
        alignment_version="strict_asof_v1",
        strict_asof_fail_fast=True,
    )
    assert out["status"] == "failed"
    assert str(out["reason"]).startswith("strict_asof_legacy_fallback_blocked:")
    assert out["alignment_audit"]["fail_fast"] is True
    assert out["leakage_checks"]["passed"] is False
