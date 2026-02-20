from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_strict_asof_alignment_is_hard_fail_without_legacy_fallback():
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
    assert str(out["reason"]).startswith("strict_asof_alignment_fail_fast:")
    assert "fallback" not in str(out["reason"]).lower()
