from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))

import liquid_model_service as svc_mod  # noqa: E402
from liquid_model_service import LiquidModelService  # noqa: E402
from features.feature_contract import FEATURE_DIM, SCHEMA_HASH  # noqa: E402


class _Repo:
    db_url = "postgresql://unused"

    def latest_price_snapshot(self, _target: str):
        return {"price": 100.0}

    def recent_event_context(self, _target: str, limit: int = 8):
        _ = limit
        return []


def _fake_ckpt() -> dict:
    return {
        "schema_hash": SCHEMA_HASH,
        "lookback": 16,
        "feature_dim": FEATURE_DIM,
        "horizons": ["1h", "4h", "1d", "7d"],
        "quantiles": [0.1, 0.5, 0.9],
        "backbone_name": "patchtst",
    }


def test_ensemble_weight_count_mismatch_fails_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LIQUID_MODEL_DIR", str(tmp_path / "primary"))
    monkeypatch.setenv(
        "LIQUID_ENSEMBLE_MODEL_DIRS",
        f"{tmp_path / 'm1'},{tmp_path / 'm2'}",
    )
    monkeypatch.setenv("LIQUID_ENSEMBLE_WEIGHTS", "0.7")

    monkeypatch.setattr(
        LiquidModelService,
        "_load_model_or_fail",
        lambda self: (object(), {"model_id": "primary", "files": {"weights": "weights.pt"}}, _fake_ckpt(), 16),
    )

    def _fake_load_from_dir(self, model_dir: Path):  # noqa: ANN001
        return svc_mod._LoadedModelBundle(  # pylint: disable=protected-access
            model=object(),
            manifest={"model_id": model_dir.name, "files": {"weights": "weights.pt"}},
            ckpt=_fake_ckpt(),
            lookback=16,
            model_dir=Path(model_dir),
        )

    monkeypatch.setattr(LiquidModelService, "_load_model_from_dir_or_fail", _fake_load_from_dir)

    with pytest.raises(RuntimeError, match="ensemble_weights_count_mismatch"):
        LiquidModelService(
            repo=_Repo(),
            feature_keys=["ret_1", "vol_12"],
            feature_version="main",
            data_version="main",
        )


def test_ensemble_aggregation_weighted_mean(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.delenv("LIQUID_ENSEMBLE_MODEL_DIRS", raising=False)
    monkeypatch.delenv("LIQUID_ENSEMBLE_WEIGHTS", raising=False)
    monkeypatch.setenv("LIQUID_MODEL_DIR", str(tmp_path / "primary"))
    monkeypatch.setattr(
        LiquidModelService,
        "_load_model_or_fail",
        lambda self: (object(), {"model_id": "primary", "files": {"weights": "weights.pt"}}, _fake_ckpt(), 16),
    )
    svc = LiquidModelService(
        repo=_Repo(),
        feature_keys=["ret_1", "vol_12"],
        feature_version="main",
        data_version="main",
    )
    svc.ensemble_enabled = True
    svc.ensemble_weights = np.asarray([0.75, 0.25], dtype=np.float64)

    members_pred = [
        {
            "horizons": ["1h", "4h", "1d", "7d"],
            "mu": np.asarray([0.01, 0.02, 0.04, 0.05], dtype=np.float64),
            "sigma": np.asarray([0.10, 0.12, 0.15, 0.20], dtype=np.float64),
            "q": np.asarray(
                [
                    [-0.05, 0.01, 0.06],
                    [-0.04, 0.02, 0.07],
                    [-0.03, 0.04, 0.09],
                    [-0.02, 0.05, 0.12],
                ],
                dtype=np.float64,
            ),
            "direction_logit": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
            "confidence": np.asarray([0.52, 0.55, 0.60, 0.62], dtype=np.float64),
            "expert_vec": np.asarray([0.6, 0.2, 0.1, 0.1], dtype=np.float64),
            "regime_probs_vec": np.asarray([0.5, 0.3, 0.2], dtype=np.float64),
            "regime_features": {"realized_vol": 0.2},
            "gate": 0.4,
            "model_id": "m1",
            "model_dir": "/tmp/m1",
            "backbone": "patchtst",
            "calibration": {"sigma_scale": 1.0},
        },
        {
            "horizons": ["1h", "4h", "1d", "7d"],
            "mu": np.asarray([0.00, 0.01, 0.02, 0.03], dtype=np.float64),
            "sigma": np.asarray([0.11, 0.13, 0.16, 0.21], dtype=np.float64),
            "q": np.asarray(
                [
                    [-0.06, 0.00, 0.05],
                    [-0.05, 0.01, 0.06],
                    [-0.04, 0.02, 0.08],
                    [-0.03, 0.03, 0.10],
                ],
                dtype=np.float64,
            ),
            "direction_logit": np.asarray([0.05, 0.1, 0.2, 0.3], dtype=np.float64),
            "confidence": np.asarray([0.50, 0.52, 0.56, 0.58], dtype=np.float64),
            "expert_vec": np.asarray([0.3, 0.3, 0.2, 0.2], dtype=np.float64),
            "regime_probs_vec": np.asarray([0.4, 0.4, 0.2], dtype=np.float64),
            "regime_features": {"realized_vol": 0.2},
            "gate": 0.2,
            "model_id": "m2",
            "model_dir": "/tmp/m2",
            "backbone": "itransformer",
            "calibration": {"sigma_scale": 1.0},
        },
    ]

    out = svc._aggregate_member_predictions(  # pylint: disable=protected-access
        members_pred=members_pred,
        horizon="1d",
        coverage_summary={"missing_ratio": 0.0},
    )
    expected_mu_1d = float(0.75 * 0.04 + 0.25 * 0.02)
    assert out["expected_return_horizons"]["1d"] == pytest.approx(expected_mu_1d, rel=1e-9)
    assert out["stack"]["ensemble"]["enabled"] is True
    assert out["stack"]["ensemble"]["member_count"] == 2
    assert out["stack"]["ensemble"]["weights"] == pytest.approx([0.75, 0.25], rel=1e-12)
