from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from eval_multimodal_oos import (  # noqa: E402
    _infer_macro_feature_indices,
    _infer_text_feature_indices,
    _materialize_ablation,
)


def test_infer_feature_groups_for_ablations():
    keys = [
        "ret_1",
        "funding_rate",
        "onchain_norm",
        "social_buzz",
        "event_decay",
        "source_confidence",
        "latent_020",
        "latent_096",
        "sentiment_abs",
    ]
    text_idx = _infer_text_feature_indices(keys)
    macro_idx = _infer_macro_feature_indices(keys)
    assert text_idx == [3, 4, 5, 7, 8]
    assert macro_idx == [1, 2]


def test_materialize_no_text_masks_only_text_columns():
    keys = ["ret_1", "social_buzz", "event_density", "funding_rate"]
    X = np.array(
        [
            [1.0, 0.7, 0.2, 0.1],
            [2.0, 0.9, 0.3, 0.2],
        ],
        dtype=np.float64,
    )
    y = np.array([0.01, -0.02], dtype=np.float64)
    Xa, ya, meta = _materialize_ablation(
        X,
        y,
        ablation="no_text",
        keys=keys,
        event_strength_threshold=0.05,
    )
    assert np.array_equal(ya, y)
    assert float(Xa[0, 0]) == 1.0
    assert float(Xa[0, 3]) == 0.1
    assert float(Xa[0, 1]) == 0.0
    assert float(Xa[0, 2]) == 0.0
    assert int(meta["masked_features"]) == 2


def test_materialize_event_window_filters_by_strength():
    keys = ["ret_1", "event_density", "social_buzz", "event_decay"]
    X = np.array(
        [
            [0.0, 0.00, 0.00, 0.00],
            [0.0, 0.30, 0.10, 0.20],
            [0.0, 0.01, 0.00, 0.02],
            [0.0, 0.60, 0.40, 0.80],
        ],
        dtype=np.float64,
    )
    y = np.array([0.0, 0.1, -0.1, 0.2], dtype=np.float64)
    Xa, ya, meta = _materialize_ablation(
        X,
        y,
        ablation="event_window",
        keys=keys,
        event_strength_threshold=0.15,
    )
    assert Xa.shape[0] == 2
    assert ya.shape[0] == 2
    assert abs(float(meta["selected_ratio"]) - 0.5) < 1e-12
