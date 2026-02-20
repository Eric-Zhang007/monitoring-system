from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "training"))
sys.path.append(str(ROOT / "inference"))

from feature_pipeline import DERIVATIVE_FEATURE_KEYS as TRAIN_DERIV_FEATURE_KEYS  # noqa: E402
from feature_pipeline import DERIVATIVE_METRIC_NAMES as TRAIN_DERIV_METRIC_NAMES  # noqa: E402
from feature_pipeline import FeaturePipeline, LIQUID_FEATURE_KEYS as TRAIN_KEYS  # noqa: E402
from liquid_feature_contract import DERIVATIVE_FEATURE_KEYS as INFER_DERIV_FEATURE_KEYS  # noqa: E402
from liquid_feature_contract import DERIVATIVE_METRIC_NAMES as INFER_DERIV_METRIC_NAMES  # noqa: E402
from liquid_feature_contract import LIQUID_FEATURE_KEYS as INFER_KEYS  # noqa: E402


def test_derivative_feature_keys_are_aligned_between_training_and_inference_contract():
    assert list(TRAIN_DERIV_METRIC_NAMES) == list(INFER_DERIV_METRIC_NAMES)
    assert list(TRAIN_DERIV_FEATURE_KEYS) == list(INFER_DERIV_FEATURE_KEYS)
    for key in TRAIN_DERIV_FEATURE_KEYS:
        assert key in TRAIN_KEYS
        assert key in INFER_KEYS
        assert f"{key}_missing_flag" in TRAIN_KEYS
        assert f"{key}_missing_flag" in INFER_KEYS


def test_latest_before_with_missing_enforces_no_lookahead():
    base = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    ts_list = [base + timedelta(minutes=10), base + timedelta(minutes=20)]
    values = [1.2, 0.8]

    v0, m0 = FeaturePipeline._latest_before_with_missing(ts_list, values, base + timedelta(minutes=5))
    assert float(v0) == 0.0
    assert float(m0) == 1.0

    v1, m1 = FeaturePipeline._latest_before_with_missing(ts_list, values, base + timedelta(minutes=15))
    assert abs(float(v1) - 1.2) < 1e-12
    assert float(m1) == 0.0

    v2, m2 = FeaturePipeline._latest_before_with_missing(ts_list, values, base + timedelta(minutes=30))
    assert abs(float(v2) - 0.8) < 1e-12
    assert float(m2) == 0.0
