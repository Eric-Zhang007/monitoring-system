from __future__ import annotations

from scripts.merge_feature_views import _row_dim_mismatch
from features.feature_contract import FEATURE_DIM


def test_row_dim_mismatch_detection() -> None:
    ok = {"feature_values": [0.0] * FEATURE_DIM, "feature_mask": [0] * FEATURE_DIM}
    bad_values = {"feature_values": [0.0] * (FEATURE_DIM - 1), "feature_mask": [0] * FEATURE_DIM}
    bad_mask = {"feature_values": [0.0] * FEATURE_DIM, "feature_mask": [0] * (FEATURE_DIM + 1)}

    assert _row_dim_mismatch(ok) is False
    assert _row_dim_mismatch(bad_values) is True
    assert _row_dim_mismatch(bad_mask) is True
