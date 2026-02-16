from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "import_social_events_jsonl.py"
    spec = importlib.util.spec_from_file_location("import_social_events_jsonl", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_canonical_event_maps_social_fields():
    mod = _load_module()

    raw = {
        "platform": "reddit",
        "title": "BTC bullish rally from ETFs",
        "content": "Community expects BTC breakout soon.",
        "created_at": 1735689600,
        "source_url": "https://reddit.com/r/cryptocurrency/test",
        "author": "alice",
        "author_followers": "0",
        "likes": 12,
        "comment_count": 4,
        "shares": 2,
        "comments": [{"sentiment": 0.5}, {"sentiment": -0.1}],
        "symbol_mentions": ["$btc", "eth"],
    }

    ev = mod.to_canonical_event(raw)
    payload = ev["payload"]

    assert ev["event_type"] == "market"
    assert ev["market_scope"] == "crypto"
    assert payload["social_platform"] == "reddit"
    assert payload["author"] == "alice"
    assert payload["author_followers"] == 0
    assert payload["n_comments"] == 4
    assert payload["symbol_mentions"] == ["BTC", "ETH"]
    assert payload["engagement_score"] > 0
    assert -1.0 <= payload["comment_sentiment"] <= 1.0
    assert -1.0 <= payload["post_sentiment"] <= 1.0


def test_to_canonical_event_keeps_canonical_and_fills_social_defaults():
    mod = _load_module()

    canonical = {
        "event_type": "market",
        "title": "Canonical social event",
        "occurred_at": "2026-01-01T00:00:00Z",
        "source_url": "https://example.com/post/1",
        "source_name": "social_test",
        "market_scope": "crypto",
        "payload": {"summary": "hello", "social_platform": "youtube", "author": "channel"},
        "entities": [],
    }

    ev = mod.to_canonical_event(canonical)
    payload = ev["payload"]

    assert payload["social_platform"] == "youtube"
    assert payload["author"] == "channel"
    assert "post_sentiment" in payload
    assert "comment_sentiment" in payload
    assert "symbol_mentions" in payload
