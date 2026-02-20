from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import feature_signal as signal_mod  # noqa: E402


def test_feature_signal_score_keeps_native_polarity():
    bullish = {
        "ret_1": 0.01,
        "ret_3": 0.02,
        "ret_12": 0.03,
        "ret_48": 0.04,
        "vol_12": 0.005,
        "vol_48": 0.006,
        "vol_96": 0.007,
        "orderbook_imbalance": 0.2,
        "funding_rate": 0.001,
        "onchain_norm": 0.1,
        "event_decay": 0.2,
    }
    bearish = dict(bullish)
    bearish["ret_1"] = -bullish["ret_1"]
    bearish["ret_3"] = -bullish["ret_3"]
    bearish["ret_12"] = -bullish["ret_12"]
    bearish["ret_48"] = -bullish["ret_48"]
    bearish["orderbook_imbalance"] = -bullish["orderbook_imbalance"]
    bearish["funding_rate"] = -bullish["funding_rate"]
    bullish_score = signal_mod.feature_signal_score(bullish)
    bearish_score = signal_mod.feature_signal_score(bearish)
    assert bullish_score > bearish_score
