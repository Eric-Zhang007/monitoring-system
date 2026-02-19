from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def _build_feature_rows(n: int):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = []
    for i in range(n):
        out.append(
            {
                "lineage_id": f"ln-{i}",
                "as_of_ts": base + timedelta(hours=i),
                "feature_payload": {
                    "ret_1": 0.001,
                    "ret_3": 0.002,
                    "ret_12": 0.003,
                    "ret_48": 0.001,
                    "vol_3": 0.01,
                    "vol_12": 0.01,
                    "vol_48": 0.01,
                    "vol_96": 0.01,
                    "log_volume": 7.0,
                    "vol_z": 0.1,
                    "volume_impact": 0.001,
                    "orderbook_imbalance": 0.0,
                    "funding_rate": 0.0,
                    "onchain_norm": 0.0,
                    "event_decay": 0.5,
                },
            }
        )
    return out


def _build_price_rows(n: int):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    p = 100.0
    rows = []
    for i in range(n):
        p = p * (1.0 + (0.0008 if i % 2 == 0 else -0.0006))
        rows.append({"price": p, "volume": 1000.0 + i, "timestamp": base + timedelta(hours=i)})
    return rows


def test_model_score_source_fails_without_artifact_under_strict_model_only(monkeypatch):
    monkeypatch.setattr(router_mod, "_load_tabular_model_weights", lambda _target: None)
    out = router_mod._run_model_inference_backtest(
        target="BTC",
        feature_rows=_build_feature_rows(120),
        price_rows=_build_price_rows(121),
        fee_bps=5.0,
        slippage_bps=3.0,
    )
    assert out["status"] == "failed"
    assert out["reason"] == "model_inference_incomplete"
    assert out["score_source"] == "model"
    assert out["model_inference_coverage"] == 0.0
    assert out["fallback_used"] is False
