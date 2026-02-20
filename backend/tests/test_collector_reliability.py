from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import pytest

_collector_path = Path(__file__).resolve().parents[2] / "collector"
if _collector_path.exists():
    sys.path.append(str(_collector_path))
else:
    pytest.skip("collector module not available", allow_module_level=True)

import collector as collector_mod  # noqa: E402
from connectors.base import RateLimitError  # noqa: E402


class _RateLimitConnector:
    name = "rate_limit_test"

    def fetch(self):
        raise RateLimitError("test_rate_limit")


class _Resp:
    def raise_for_status(self):
        return None


def test_fetch_connector_rows_triggers_cooldown(monkeypatch):
    monkeypatch.setattr(collector_mod, "start_http_server", lambda *_args, **_kwargs: None)
    dc = collector_mod.DataCollectorV2()
    dc.fetch_max_retries = 1
    dc.fetch_failure_threshold = 1
    dc.fetch_cooldown_sec = 60

    rows, status = dc._fetch_connector_rows(_RateLimitConnector())
    assert rows == []
    assert status == "failure"

    rows2, status2 = dc._fetch_connector_rows(_RateLimitConnector())
    assert rows2 == []
    assert status2 == "cooldown"

    state = dc._state("rate_limit_test")
    assert state["rate_limit_count"] >= 1
    assert state["cooldown_skips"] >= 1


def test_publish_event_includes_source_quality_payload(monkeypatch):
    monkeypatch.setattr(collector_mod, "start_http_server", lambda *_args, **_kwargs: None)
    captured = {}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return _Resp()

    dc = collector_mod.DataCollectorV2()
    dc._http_post = _fake_post
    event = {
        "event_type": "market",
        "title": "test event",
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

    body = captured["body"]
    payload = body["events"][0]["payload"]
    assert payload["source_fetch_status"] == "success"
    assert "source_confidence" in payload
    assert "source_tier_weight" in payload
    assert "source_health" in payload


def test_normalize_entities_marks_synthetic_links(monkeypatch):
    monkeypatch.setattr(collector_mod, "start_http_server", lambda *_args, **_kwargs: None)
    dc = collector_mod.DataCollectorV2()
    dc.liquid_symbols = {"BTC", "ETH"}
    out = dc._normalize_entities([], market_scope="crypto")
    assert len(out) == 2
    assert all(bool((e.get("metadata") or {}).get("synthetic_link")) for e in out)
