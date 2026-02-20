from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest
from fastapi import HTTPException

sys.path.append(str(Path(__file__).resolve().parents[1]))

import routers.multimodal as multimodal_mod  # noqa: E402
from schemas_v2 import IngestSocialRequest, SocialIngestItem  # noqa: E402


class _FakeRepo:
    def __init__(self):
        self.enriched_rows = []

    def ingest_events(self, events):
        return len(events), len(events), 0, [100 + i for i in range(len(events))]

    def upsert_enriched_event_feature(self, event_id: int, payload):
        self.enriched_rows.append((event_id, payload))
        return True

    def latest_feature_snapshot(self, target: str, track: str):
        if target == "MISSING":
            return None
        return {
            "target": target,
            "track": track,
            "feature_version": "feature-store-main",
            "data_version": "v1",
            "as_of": datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
            "as_of_ts": datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
            "event_time": datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
            "feature_available_at": datetime(2026, 2, 19, 10, 1, tzinfo=timezone.utc),
            "lineage_id": "lineage-1",
            "feature_payload": {"ret_1": 0.01},
            "created_at": datetime(2026, 2, 19, 10, 1, tzinfo=timezone.utc),
        }

    def social_coverage_stats(self, window_hours: int, target: str | None = None):
        return {
            "totals": {
                "total_events": 10.0,
                "posts": 7.0,
                "comments": 3.0,
                "enriched_events": 8.0,
                "avg_ingest_lag_sec": 25.0,
                "coverage_ratio": 0.8,
            },
            "by_symbol": [
                {
                    "symbol": "BTC",
                    "samples": 6.0,
                    "posts": 4.0,
                    "comments": 2.0,
                    "enriched_events": 5.0,
                    "coverage_ratio": 0.833333,
                }
            ],
        }


def test_ingest_social_writes_events_and_enrichment(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(multimodal_mod, "repo", fake)
    payload = IngestSocialRequest(
        items=[
            SocialIngestItem(
                platform="reddit",
                kind="post",
                text="BTC up",
                symbol="BTC",
                occurred_at=datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
            )
        ]
    )
    resp = asyncio.run(multimodal_mod.ingest_social(payload))
    assert resp.accepted == 1
    assert resp.inserted == 1
    assert resp.enriched_written == 1
    assert len(fake.enriched_rows) == 1
    assert fake.enriched_rows[0][0] == 100


def test_get_latest_feature_success(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(multimodal_mod, "repo", fake)
    resp = asyncio.run(multimodal_mod.get_latest_feature(target="BTC", track="liquid"))
    assert resp.target == "BTC"
    assert resp.track == "liquid"
    assert resp.feature_payload["ret_1"] == 0.01


def test_get_latest_feature_not_found(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(multimodal_mod, "repo", fake)
    with pytest.raises(HTTPException):
        asyncio.run(multimodal_mod.get_latest_feature(target="MISSING", track="liquid"))


def test_get_social_coverage(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(multimodal_mod, "repo", fake)
    resp = asyncio.run(multimodal_mod.get_social_coverage(window_hours=24, target="btc"))
    assert resp.window_hours == 24
    assert resp.target == "BTC"
    assert resp.totals["coverage_ratio"] == 0.8
    assert resp.by_symbol[0]["symbol"] == "BTC"


def test_get_social_coverage_compat_alias(monkeypatch):
    fake = _FakeRepo()
    monkeypatch.setattr(multimodal_mod, "repo", fake)
    resp = asyncio.run(multimodal_mod.get_social_coverage_compat(window=12, target="eth"))
    assert resp.window_hours == 12
    assert resp.target == "ETH"
