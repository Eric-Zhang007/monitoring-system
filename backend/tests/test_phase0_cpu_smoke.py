from __future__ import annotations

import asyncio
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "backend"))
sys.path.append(str(ROOT / "training"))

import train_multimodal as train_mod  # noqa: E402
import v2_router as router_mod  # noqa: E402
from liquid_model_service import LiquidModelService  # noqa: E402
from schemas_v2 import SignalGenerateRequest  # noqa: E402


class _FakeSignalRepo:
    def insert_signal_candidate(self, **kwargs):
        _ = kwargs
        return 1


def test_phase0_cpu_smoke_training_inference_signal(monkeypatch, tmp_path):
    rng = np.random.default_rng(7)
    feature_keys = list(train_mod.LIQUID_FEATURE_KEYS)
    n_rows = 640
    X = rng.normal(size=(n_rows, len(feature_keys))).astype(np.float64)
    y = (0.08 * X[:, 0] - 0.04 * X[:, 1] + rng.normal(0.0, 0.01, size=n_rows)).astype(np.float64)
    metas = [f"BTC@{i}" for i in range(n_rows)]
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "liquid_btc_lgbm_baseline_v2.json"

    monkeypatch.setattr(train_mod, "_load_matrix", lambda *args, **kwargs: (X, y, metas))
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("FEATURE_VERSION", "feature-store-main")
    monkeypatch.setenv("DATA_VERSION", "2018_now_full_window")
    monkeypatch.setenv("FEATURE_PAYLOAD_SCHEMA_VERSION", str(train_mod.LIQUID_FEATURE_SCHEMA_VERSION))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_multimodal.py",
            "--database-url",
            "postgresql://unused",
            "--start",
            "2025-01-01T00:00:00Z",
            "--end",
            "2025-12-31T00:00:00Z",
            "--symbols",
            "BTC",
            "--fusion-mode",
            "single_ridge",
            "--text-dropout-prob",
            "0.0",
            "--out",
            str(model_path),
        ],
    )
    rc = train_mod.main()
    assert rc == 0
    assert model_path.exists()

    payload = {k: float(X[-1, idx]) for idx, k in enumerate(feature_keys)}
    svc = LiquidModelService(
        repo=object(),
        feature_keys=feature_keys,
        feature_version="feature-store-main",
        data_version="2018_now_full_window",
        default_model_name="liquid_baseline",
        default_model_version="v2.0",
    )
    pred = svc.predict_from_feature_payload(
        target="BTC",
        payload=payload,
        horizon="1d",
        model_name="liquid_baseline",
        model_version="v2.0",
        require_artifact=False,
    )
    assert isinstance(float(pred["expected_return"]), float)
    assert 0.0 <= float(pred["signal_confidence"]) <= 1.0
    assert float(pred["vol_forecast"]) > 0.0

    monkeypatch.setattr(router_mod, "repo", _FakeSignalRepo())
    monkeypatch.setattr(router_mod, "_kill_switch_block_reason", lambda *args, **kwargs: None)
    monkeypatch.setattr(router_mod, "_strategy_bucket", lambda *args, **kwargs: "trend")

    def _build_pred(symbol: str, horizon: str):
        out = svc.predict_from_feature_payload(
            target=symbol,
            payload=payload,
            horizon=horizon,
            model_name="liquid_baseline",
            model_version="v2.0",
            require_artifact=False,
        )
        return {
            "target": str(symbol).upper(),
            "score": float(out["expected_return"]),
            "confidence": float(out["signal_confidence"]),
            "outputs": {
                "expected_return": float(out["expected_return"]),
                "signal_confidence": float(out["signal_confidence"]),
                "vol_forecast": float(out["vol_forecast"]),
            },
            "model_name": "liquid_baseline",
            "model_version": "v2.0",
        }

    monkeypatch.setattr(router_mod, "_build_liquid_prediction", _build_pred)

    req = SignalGenerateRequest(
        track="liquid",
        target="BTC",
        horizon="1d",
        policy="baseline-v2",
        min_confidence=0.0,
        strategy_id="phase0-smoke",
        cost_profile="standard",
        risk_profile="balanced",
    )
    resp = asyncio.run(router_mod.generate_signal(req))
    assert resp.signal_id == 1
    assert resp.track == "liquid"
    assert resp.target == "BTC"
    assert resp.action in {"buy", "sell", "hold"}
