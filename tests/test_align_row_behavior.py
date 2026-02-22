from __future__ import annotations

import numpy as np
import pytest

from features.align import FeatureAlignmentError, align_row


def test_align_row_rejects_extra_keys():
    with pytest.raises(FeatureAlignmentError):
        align_row({"not_exist": 1.0})


def test_align_row_rejects_required_missing():
    # ret_1 is required in generated schema
    with pytest.raises(FeatureAlignmentError):
        align_row({"ret_3": 0.1})


def test_align_row_returns_values_and_mask():
    row = {"ret_1": 0.1, "ret_3": 0.2, "ret_12": 0.3, "ret_48": 0.4, "ret_96": 0.5, "vol_3": 0.01, "vol_12": 0.02, "vol_48": 0.03, "log_volume": 1.0}
    out = align_row(row, allow_required_missing=True)
    assert out.values.ndim == 1
    assert out.mask.ndim == 1
    assert out.values.shape == out.mask.shape
    assert out.values.dtype == np.float32
