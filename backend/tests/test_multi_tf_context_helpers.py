from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import v2_router as router_mod  # noqa: E402


def test_parse_timeframe_list_filters_invalid():
    out = router_mod._parse_timeframe_list("5m,1h,bad,1h,4h,1x")
    assert out == ["5m", "1h", "4h"]


def test_multi_tf_context_to_regime_features_maps_expected_keys():
    context_payload = {
        "context": {
            "5m": {"missing": 0, "ret_1": 0.01, "volume": 12.5, "lag_sec": 3},
            "1h": {"missing": 1, "ret_1": 0.0, "volume": 0.0, "lag_sec": 0},
        }
    }
    out = router_mod._multi_tf_context_to_regime_features(context_payload)
    assert float(out["mtf_5m_missing"]) == 0.0
    assert float(out["mtf_5m_ret_1"]) == 0.01
    assert float(out["mtf_5m_volume"]) == 12.5
    assert float(out["mtf_1h_missing"]) == 1.0
    assert "mtf_1h_lag_sec" in out
