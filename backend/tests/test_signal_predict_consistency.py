from __future__ import annotations

import asyncio
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402
from schemas_v2 import LiquidPredictRequest, SignalGenerateRequest  # noqa: E402


def _stub_liquid_prediction(symbol: str, horizon: str):
    return {
        "target": symbol.upper(),
        "score": 0.023,
        "confidence": 0.82,
        "outputs": {
            "expected_return": 0.023,
            "signal_confidence": 0.82,
            "vol_forecast": 0.11,
            "stack": {"nn": 0.02, "tabular": 0.026, "alpha": 0.7},
            "degraded": False,
            "degraded_reasons": [],
            "current_price": 123.0,
        },
        "explanation": {"model_version": "liquid_ttm_ensemble:v2.1", "feature_version": "feature-store-v2.1"},
        "model_name": "liquid_ttm_ensemble",
        "model_version": "v2.1",
    }


def test_predict_and_signal_share_same_liquid_scores(monkeypatch):
    class _FakeRepo:
        def insert_prediction(self, **kwargs):
            return 101

        def insert_signal_candidate(self, **kwargs):
            return 202

        def is_kill_switch_triggered(self, track: str, strategy_id: str) -> bool:
            return False

    monkeypatch.setattr(router_mod, "repo", _FakeRepo())
    monkeypatch.setattr(router_mod, "_build_liquid_prediction", _stub_liquid_prediction)
    monkeypatch.setattr(router_mod, "_strategy_bucket", lambda *args, **kwargs: "trend")

    pred_resp = asyncio.run(router_mod.predict_liquid(LiquidPredictRequest(symbol="BTC", horizon="1d")))
    sig_resp = asyncio.run(
        router_mod.generate_signal(
            SignalGenerateRequest(
                track="liquid",
                target="BTC",
                horizon="1d",
                min_confidence=0.5,
                policy="baseline-v2",
            )
        )
    )
    assert abs(float(pred_resp["expected_return"]) - float(sig_resp.score)) < 1e-12
    assert abs(float(pred_resp["signal_confidence"]) - float(sig_resp.confidence)) < 1e-12
