from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from liquid_model_service import LiquidModelService  # noqa: E402


class _FakeRepo:
    def get_active_model_state(self, _track: str):
        return {"active_model_name": "liquid_ttm_ensemble", "active_model_version": "v2.1"}

    def load_feature_history(self, **_kwargs):
        return []

    def latest_price_snapshot(self, _target: str):
        return {"price": 100.0}

    def recent_event_context(self, _target: str, limit: int = 8):
        return []


class _FakeRouterNoNN:
    def _load_liquid_tabular_model(self, _sym: str):
        return {"feature_dim": 2}

    def _load_torch_model(self, _path: str, _model_type: str, _feature_dim: int):
        return None

    def predict_liquid(self, _sym: str, _vec, model_name: str = "liquid_ttm_ensemble"):
        return {
            "expected_return": 0.1,
            "signal_confidence": 0.8,
            "vol_forecast": 0.12,
            "stack": {"nn": 0.0, "tabular": 0.1, "alpha": 0.0},
            "model_name": model_name,
        }


def _build_service(monkeypatch, require_nn: bool) -> LiquidModelService:
    monkeypatch.setenv("LIQUID_REQUIRE_NN_ARTIFACT", "1" if require_nn else "0")
    svc = LiquidModelService(
        repo=_FakeRepo(),
        feature_keys=["ret_1", "vol_12"],
        feature_version="feature-store-v2.1",
        data_version="v1",
    )
    svc._model_router = _FakeRouterNoNN()
    return svc


def test_has_required_artifacts_allows_tabular_fallback_when_nn_optional(monkeypatch):
    svc = _build_service(monkeypatch, require_nn=False)
    assert svc.has_required_artifacts(target="BTC", model_name="liquid_ttm_ensemble") is True


def test_predict_from_feature_payload_marks_degraded_when_nn_missing(monkeypatch):
    svc = _build_service(monkeypatch, require_nn=False)
    out = svc.predict_from_feature_payload(
        target="ETH",
        payload={"ret_1": 0.01, "vol_12": 0.2},
        model_name="liquid_ttm_ensemble",
        model_version="v2.1",
    )
    assert out["degraded"] is True
    assert "neural_artifact_missing_tabular_fallback" in out["degraded_reasons"]
    stack = out.get("stack") if isinstance(out.get("stack"), dict) else {}
    assert stack.get("neural_artifact_present") is False


def test_require_nn_artifact_raises_when_missing(monkeypatch):
    svc = _build_service(monkeypatch, require_nn=True)
    with pytest.raises(RuntimeError, match="model_artifact_missing:neural:SOL"):
        svc.predict_from_feature_payload(
            target="SOL",
            payload={"ret_1": 0.02, "vol_12": 0.3},
            model_name="liquid_ttm_ensemble",
            model_version="v2.1",
        )
