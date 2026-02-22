from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

import liquid_model_service as svc_mod  # noqa: E402
from liquid_model_service import LiquidModelService  # noqa: E402


class _FakeRepo:
    db_url = "postgresql://unused"

    def latest_price_snapshot(self, _target: str):
        return {"price": 100.0}

    def recent_event_context(self, _target: str, limit: int = 8):
        _ = limit
        return []


def _build_service(monkeypatch) -> LiquidModelService:
    monkeypatch.setattr(
        LiquidModelService,
        "_load_model_or_fail",
        lambda self: (object(), {"model_id": "liquid_main"}, 12),
    )
    return LiquidModelService(
        repo=_FakeRepo(),
        feature_keys=["ret_1", "vol_12"],
        feature_version="feature-store-main",
        data_version="v1",
    )


def test_service_init_requires_runtime_or_valid_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        LiquidModelService(
            repo=_FakeRepo(),
            feature_keys=["ret_1", "vol_12"],
            feature_version="feature-store-main",
            data_version="v1",
        )


def test_has_required_artifacts_returns_false_when_validate_fails(monkeypatch):
    svc = _build_service(monkeypatch)
    monkeypatch.setattr(svc_mod, "validate_manifest_dir", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing_manifest")))
    assert svc.has_required_artifacts(target="BTC", model_name="liquid_main") is False


def test_predict_from_feature_payload_uses_sequence_path(monkeypatch):
    svc = _build_service(monkeypatch)
    expected = {"expected_return": 0.01, "stack": {"schema_hash": "x"}}
    monkeypatch.setattr(LiquidModelService, "_predict_sequence", lambda self, **_kwargs: dict(expected))
    out = svc.predict_from_feature_payload(target="ETH", payload={"ret_1": 0.1}, horizon="1d")
    assert out == expected
