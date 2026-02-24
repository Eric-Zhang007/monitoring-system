from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from mlops_artifacts.validate import validate_manifest_dir
from features.feature_contract import FEATURE_DIM, FEATURE_KEYS, SCHEMA_HASH
from training.datasets.liquid_sequence_dataset import SequenceSample


def _build_samples(n: int = 220, lookback: int = 16):
    out = []
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ts = base + timedelta(hours=i)
        vals = np.zeros((lookback, FEATURE_DIM), dtype=np.float32)
        msk = np.zeros((lookback, FEATURE_DIM), dtype=np.uint8)
        vals[:, 0] = 0.001 * i
        vals[:, 1] = 0.002 * np.sin(i / 10.0)
        if i % 7 == 0:
            msk[:, -8:] = 1
        y = {
            "ret_1h_raw": 0.001,
            "ret_4h_raw": 0.002,
            "ret_1d_raw": 0.003,
            "ret_7d_raw": 0.004,
            "cost_1h_bps": 6.0,
            "cost_4h_bps": 8.0,
            "cost_1d_bps": 10.0,
            "cost_7d_bps": 12.0,
            "ret_1h_net": 0.001 - 6.0 / 1e4,
            "ret_4h_net": 0.002 - 8.0 / 1e4,
            "ret_1d_net": 0.003 - 10.0 / 1e4,
            "ret_7d_net": 0.004 - 12.0 / 1e4,
            "direction_1h": 1.0,
            "direction_4h": 1.0,
            "direction_1d": 1.0,
            "direction_7d": 1.0,
            "risk_proxy_1h": 0.001,
            "risk_proxy_4h": 0.002,
            "risk_proxy_1d": 0.003,
            "risk_proxy_7d": 0.004,
        }
        out.append(
            SequenceSample(
                symbol="BTC",
                end_ts=ts,
                x_values=vals,
                x_mask=msk,
                y=y,
                schema_hash=SCHEMA_HASH,
            )
        )
    return out


def test_strict_train_artifact_can_infer(monkeypatch, tmp_path: Path):
    import training.train_liquid as train_liquid

    samples = _build_samples()
    monkeypatch.setattr(train_liquid, "load_training_samples", lambda **kwargs: samples)

    out_dir = tmp_path / "liquid_main"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_liquid.py",
            "--out-dir",
            str(out_dir),
            "--model-id",
            "liquid_main",
            "--lookback",
            "16",
            "--use-universe-snapshot",
            "0",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--backbone",
            "patchtst",
            "--train-days",
            "2",
            "--val-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--purge-gap-hours",
            "1",
            "--force-purged",
        ],
    )
    assert train_liquid.main() == 0

    manifest = validate_manifest_dir(out_dir, expected_schema_hash=SCHEMA_HASH)
    assert manifest["model_id"] == "liquid_main"

    import liquid_model_service as svc_mod

    class _Repo:
        db_url = "postgresql://unused"

    seq = {
        "values": samples[-1].x_values,
        "mask": samples[-1].x_mask,
        "schema_hash": SCHEMA_HASH,
        "symbol": "BTC",
        "start_ts": "x",
        "end_ts": "y",
        "bucket_interval": "5m",
        "coverage_summary": {"missing_ratio": 0.0, "observed_ratio": 1.0},
    }
    monkeypatch.setattr(svc_mod, "fetch_sequence", lambda **kwargs: seq)
    monkeypatch.setenv("LIQUID_MODEL_DIR", str(out_dir))

    service = svc_mod.LiquidModelService(
        repo=_Repo(),
        feature_keys=list(FEATURE_KEYS),
        feature_version="main",
        data_version="main",
    )
    pred = service.predict_from_feature_payload(target="BTC", payload={}, horizon="1h")

    assert "expected_return_horizons" in pred
    assert "signal_confidence_horizons" in pred
    assert "vol_forecast_horizons" in pred
    assert pred["stack"]["schema_hash"] == SCHEMA_HASH
    assert set(pred["expected_return_horizons"].keys()) == {"1h", "4h", "1d", "7d"}
