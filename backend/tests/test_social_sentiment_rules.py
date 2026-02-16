from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    mod_path = Path(__file__).resolve().parents[2] / "collector" / "connectors" / "social_common.py"
    spec = importlib.util.spec_from_file_location("social_common", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sentiment_negation_and_intensity():
    mod = _load_module()
    pos = mod.sentiment_score("BTC is very bullish and strong 🚀")
    negated = mod.sentiment_score("BTC is not bullish")
    assert pos > 0.2
    assert negated < 0.0


def test_sentiment_sarcasm_is_downweighted():
    mod = _load_module()
    plain = mod.sentiment_score("great breakout bullish")
    sarcastic = mod.sentiment_score("great breakout bullish /s")
    assert plain > 0
    assert abs(sarcastic) < abs(plain)
