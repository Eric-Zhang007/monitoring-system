from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from schemas_v2 import Entity, SocialIngestItem  # noqa: E402
from services.multimodal_service import social_items_to_events  # noqa: E402


def test_social_items_to_events_builds_market_event_and_enrichment():
    item = SocialIngestItem(
        platform="reddit",
        kind="comment",
        title="",
        text="BTC sentiment turning positive",
        symbol="btc",
        occurred_at=datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
        ingested_at=datetime(2026, 2, 19, 10, 1, tzinfo=timezone.utc),
        author="alice",
        author_followers=1200,
        engagement_score=56.0,
        post_sentiment=0.6,
        comment_sentiment=0.2,
        metadata={"language": "en", "novelty_score": 0.8},
    )
    events, enrich_rows = social_items_to_events([item])
    assert len(events) == 1
    assert len(enrich_rows) == 1

    evt = events[0]
    assert evt.event_type == "market"
    assert evt.source_name == "social:reddit"
    assert evt.payload.get("social_platform") == "reddit"
    assert evt.payload.get("social_kind") == "comment"
    assert any((e.entity_type == "asset" and e.symbol == "BTC") for e in evt.entities)

    enrich = enrich_rows[0]
    assert enrich["social_platform"] == "reddit"
    assert enrich["social_kind"] == "comment"
    assert enrich["language"] == "en"
    assert enrich["ingest_lag_sec"] >= 0.0
    assert 0.0 <= float(enrich["coverage_score"]) <= 1.0


def test_social_items_to_events_keeps_existing_asset_entity_without_duplication():
    item = SocialIngestItem(
        platform="x",
        kind="post",
        title="ETH update",
        text="ETH ecosystem post",
        symbol="eth",
        occurred_at=datetime(2026, 2, 19, 10, 0, tzinfo=timezone.utc),
        entities=[
            Entity(
                entity_type="asset",
                name="ETH",
                symbol="ETH",
                metadata={"source": "manual"},
            )
        ],
    )
    events, _ = social_items_to_events([item])
    assert len(events) == 1
    asset_entities = [e for e in events[0].entities if e.entity_type == "asset" and e.symbol == "ETH"]
    assert len(asset_entities) == 1

