from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_strategy_bucket_not_biased_by_absolute_price_scale(monkeypatch):
    feature_payload = {
        "ret_1": 0.002,
        "ret_12": 0.01,
        "vol_z": 0.2,
        "event_decay": 0.0,
    }
    monkeypatch.setattr(router_mod, "_latest_liquid_feature_payload", lambda *_args, **_kwargs: feature_payload)
    b1 = router_mod._strategy_bucket("liquid", score=0.03, confidence=0.9, target="LOWPX")
    b2 = router_mod._strategy_bucket("liquid", score=0.03, confidence=0.9, target="HIGHPX")
    assert b1 == b2


def test_strategy_bucket_uses_standardized_return_vol_features(monkeypatch):
    payload_trend = {"ret_1": 0.01, "ret_12": 0.06, "vol_z": 0.1, "event_decay": 0.0}
    monkeypatch.setattr(router_mod, "_latest_liquid_feature_payload", lambda *_args, **_kwargs: payload_trend)
    assert router_mod._strategy_bucket("liquid", score=0.08, confidence=0.9, target="BTC") == "trend"

    payload_event = {"ret_1": 0.0, "ret_12": 0.0, "vol_z": 2.0, "event_decay": 0.3}
    monkeypatch.setattr(router_mod, "_latest_liquid_feature_payload", lambda *_args, **_kwargs: payload_event)
    assert router_mod._strategy_bucket("liquid", score=0.0, confidence=0.4, target="ETH") == "event"
