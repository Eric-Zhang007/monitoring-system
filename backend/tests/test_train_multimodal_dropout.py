from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from train_multimodal import _apply_feature_dropout, _infer_text_feature_indices  # noqa: E402


def test_infer_text_feature_indices_detects_social_event_and_high_latent():
    keys = [
        "ret_1",
        "social_buzz",
        "event_decay",
        "latent_020",
        "latent_064",
        "source_x_weight",
        "comment_skew",
    ]
    out = _infer_text_feature_indices(keys)
    assert out == [1, 2, 4, 5, 6]


def test_apply_feature_dropout_is_seeded_and_only_affects_selected_columns():
    X = np.arange(24, dtype=np.float64).reshape(6, 4)
    out_a = _apply_feature_dropout(X, indices=[1, 3], prob=0.5, seed=7)
    out_b = _apply_feature_dropout(X, indices=[1, 3], prob=0.5, seed=7)
    out_c = _apply_feature_dropout(X, indices=[1, 3], prob=0.5, seed=8)
    assert np.array_equal(out_a, out_b)
    assert not np.array_equal(out_a, out_c)
    assert np.array_equal(out_a[:, 0], X[:, 0])
    assert np.array_equal(out_a[:, 2], X[:, 2])
