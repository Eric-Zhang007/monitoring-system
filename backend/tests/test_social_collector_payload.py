from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

pytest.importorskip("feedparser")

_collector_path = Path(__file__).resolve().parents[2] / "collector"
if _collector_path.exists():
    sys.path.append(str(_collector_path))
else:
    pytest.skip("collector module not available", allow_module_level=True)

import collector as collector_mod  # noqa: E402


class _Resp:
    def raise_for_status(self):
        return None


def test_publish_event_includes_standard_social_payload_fields(monkeypatch):
    monkeypatch.setattr(collector_mod, "start_http_server", lambda *_args, **_kwargs: None)
    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(collector_mod.requests, "post", _fake_post)

    dc = collector_mod.DataCollectorV2()
    event = {
        "event_type": "market",
        "title": "social schema default test",
        "occurred_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_url": "https://example.com/test",
        "source_name": "unit-test",
        "source_timezone": "UTC",
        "source_tier": 2,
        "confidence_score": 0.8,
        "event_importance": 0.7,
        "novelty_score": 0.6,
        "entity_confidence": 0.5,
        "market_scope": "crypto",
        "payload": {"summary": "hello"},
        "entities": [],
    }

    dc.publish_event(event, connector_name="unit_connector", fetch_status="success")
    payload = captured["body"]["events"][0]["payload"]

    required = {
        "social_platform",
        "author",
        "author_followers",
        "engagement_score",
        "comment_sentiment",
        "post_sentiment",
        "n_comments",
        "n_replies",
        "is_verified",
        "influence_tier",
        "symbol_mentions",
    }
    assert required.issubset(payload.keys())
    assert payload["social_platform"] == "none"
    assert payload["author_followers"] == 0
    assert payload["symbol_mentions"] == []


def test_normalize_social_payload_coerces_types(monkeypatch):
    monkeypatch.setattr(collector_mod, "start_http_server", lambda *_args, **_kwargs: None)
    dc = collector_mod.DataCollectorV2()

    normalized = dc._normalize_social_payload(
        {
            "social_platform": "x",
            "author": "bob",
            "author_followers": "1000",
            "engagement_score": "12.5",
            "comment_sentiment": "0.25",
            "post_sentiment": "-0.2",
            "n_comments": "5",
            "n_replies": "3",
            "is_verified": 1,
            "influence_tier": "micro",
            "symbol_mentions": ["$btc", "ETH", "BTC"],
        }
    )

    assert normalized["author_followers"] == 1000
    assert normalized["engagement_score"] == 12.5
    assert normalized["n_comments"] == 5
    assert normalized["n_replies"] == 3
    assert normalized["is_verified"] is True
    assert normalized["symbol_mentions"] == ["BTC", "ETH"]
