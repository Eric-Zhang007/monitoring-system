from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def _feature_rows(n: int):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        rows.append(
            {
                "lineage_id": f"ln-{i}",
                "as_of_ts": base + timedelta(hours=i),
                "feature_payload": {
                    "ret_1": 0.001 if i % 2 == 0 else -0.001,
                    "ret_12": 0.002 if i % 3 == 0 else -0.001,
                    "vol_12": 0.01,
                    "vol_48": 0.012,
                    "event_decay": 0.2,
                },
            }
        )
    return rows


def _price_rows(n: int):
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    px = 100.0
    rows = []
    for i in range(n):
        px = px * (1.0 + (0.0009 if i % 2 == 0 else -0.0007))
        rows.append({"price": px, "volume": 1000.0 + i, "timestamp": base + timedelta(hours=i)})
    return rows


def test_backtest_model_inference_uses_liquid_service_stack(monkeypatch):
    calls = []

    class _FakeService:
        def predict_from_feature_payload(self, **kwargs):
            calls.append(kwargs)
            payload = kwargs["payload"]
            score = float(payload.get("ret_1") or 0.0) + float(payload.get("ret_12") or 0.0)
            return {
                "expected_return": score,
                "signal_confidence": 0.8,
                "vol_forecast": 0.12,
                "stack": {"nn": score * 0.7, "tabular": score * 0.3, "alpha": 0.7},
                "model_name": kwargs.get("model_name") or "liquid_ttm_ensemble",
                "model_version": kwargs.get("model_version") or "v2.1",
            }

    monkeypatch.setattr(router_mod, "_get_liquid_model_service", lambda: _FakeService())
    out = router_mod._run_model_inference_backtest(
        target="BTC",
        feature_rows=_feature_rows(120),
        price_rows=_price_rows(121),
        fee_bps=5.0,
        slippage_bps=3.0,
        model_name="liquid_ttm_ensemble",
        model_version="v2.1",
    )
    assert out["status"] == "completed"
    assert out["score_source"] == "model"
    assert out["fallback_used"] is False
    assert out["model_inference_coverage"] == 1.0
    assert calls
    assert all(str(c.get("model_name")) == "liquid_ttm_ensemble" for c in calls)
