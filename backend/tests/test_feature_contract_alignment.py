from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "training"))
sys.path.append(str(ROOT / "inference"))

from feature_pipeline import LIQUID_FEATURE_KEYS as TRAIN_KEYS  # noqa: E402
from feature_pipeline import ONLINE_FEATURE_KEYS as TRAIN_ONLINE_KEYS  # noqa: E402
from liquid_feature_contract import LIQUID_FEATURE_KEYS as INFER_KEYS  # noqa: E402
from liquid_feature_contract import ONLINE_LIQUID_FEATURE_KEYS as INFER_ONLINE_KEYS  # noqa: E402


def test_feature_contract_order_dimension_and_online_projection_are_consistent():
    assert list(TRAIN_KEYS) == list(INFER_KEYS)
    assert list(TRAIN_ONLINE_KEYS) == list(INFER_ONLINE_KEYS)
    assert len(TRAIN_ONLINE_KEYS) <= len(TRAIN_KEYS)
    assert len(TRAIN_ONLINE_KEYS) == len(set(TRAIN_ONLINE_KEYS))
