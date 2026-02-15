from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "inference"))

from model_router import ModelRouter  # noqa: E402


def test_align_features_pad_and_trim():
    x = np.array([1.0, 2.0], dtype=np.float32)
    padded = ModelRouter._align_features(x, 4)
    assert padded.shape[0] == 4
    assert float(padded[0]) == 1.0
    assert float(padded[1]) == 2.0
    assert float(padded[2]) == 0.0
    assert float(padded[3]) == 0.0

    y = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    trimmed = ModelRouter._align_features(y, 2)
    assert trimmed.shape[0] == 2
    assert float(trimmed[0]) == 1.0
    assert float(trimmed[1]) == 2.0


def test_normalize_features_uses_stats():
    x = np.array([10.0, 5.0, -1.0], dtype=np.float32)
    norm = {"x_mean": [8.0, 1.0, -1.0], "x_std": [2.0, 2.0, 0.5]}
    out = ModelRouter._normalize_features(x, norm)
    assert np.allclose(out, np.array([1.0, 2.0, 0.0], dtype=np.float32))
