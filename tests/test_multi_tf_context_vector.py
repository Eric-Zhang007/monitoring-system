from __future__ import annotations

import pytest

from training.cache.panel_cache import build_multi_tf_vector_from_payload


def test_build_multi_tf_vector_from_payload_masks_missing_rows():
    payload = {
        "context": {
            "5m": {"missing": 0, "ret_1": 0.01, "volume": 12.5, "lag_sec": 3.0},
            "1h": {"missing": 1, "ret_1": 0.0, "volume": 0.0, "lag_sec": 0.0},
        },
        "coverage": {},
    }
    vec, msk = build_multi_tf_vector_from_payload(payload, timeframes=["5m", "1h"], require_complete=True)
    assert tuple(vec.shape) == (8,)
    assert tuple(msk.shape) == (8,)
    assert float(vec[0]) == 0.0
    assert float(vec[1]) == pytest.approx(0.01)
    assert int(msk[1]) == 0
    assert float(vec[4]) == 1.0
    assert int(msk[5]) == 1
    assert int(msk[6]) == 1
    assert int(msk[7]) == 1


def test_build_multi_tf_vector_from_payload_fail_fast_on_missing_block():
    payload = {"context": {"5m": {"missing": 0, "ret_1": 0.01, "volume": 10.0, "lag_sec": 1.0}}, "coverage": {}}
    with pytest.raises(RuntimeError, match="multi_tf_context_missing_block:1h"):
        build_multi_tf_vector_from_payload(payload, timeframes=["5m", "1h"], require_complete=True)
