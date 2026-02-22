from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "backend"))

from liquid_model_service import LiquidModelService  # noqa: E402


class _Repo:
    db_url = "postgresql://unused"


def test_phase0_tabular_only_path_is_not_allowed_in_strict_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("LIQUID_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError):
        LiquidModelService(
            repo=_Repo(),
            feature_keys=["ret_1", "vol_12"],
            feature_version="feature-store-main",
            data_version="2018_now_full_window",
            default_model_name="liquid_baseline",
            default_model_version="v2.0",
        )
