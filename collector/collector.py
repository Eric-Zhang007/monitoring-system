"""
V2 Data Collector
- Plugin-based connectors (GDELT / RSS / SEC)
- Canonical event schema
- Redis Streams as real-time buffer
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List

import redis

from connectors import (
    EarningsAlphaVantageConnector,
    GDELTConnector,
    MacroFREDConnector,
    OnChainCoinGeckoConnector,
    RSSConnector,
    SECSubmissionsConnector,
)
from entity_linking import extract_entities

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataCollectorV2:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url, decode_responses=True)
        self.event_stream = os.getenv("EVENT_STREAM", "event_stream")
        self.news_stream = os.getenv("NEWS_STREAM", "news_stream")

        rss_feeds = [
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://www.prnewswire.com/rss/financial-services-latest-news/financial-services-latest-news-list.rss",
        ]
        sec_ciks = ["320193", "1045810", "789019"]  # Apple/NVDA/MSFT
        alpha_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
        coingecko_ids = [
            s.strip().lower()
            for s in os.getenv("ONCHAIN_IDS", "bitcoin,ethereum,solana").split(",")
            if s.strip()
        ]

        self.connectors = [
            GDELTConnector(query="venture capital startup funding OR IPO", max_records=25),
            RSSConnector(feeds=rss_feeds),
            SECSubmissionsConnector(cik_list=sec_ciks),
            MacroFREDConnector(),
            OnChainCoinGeckoConnector(ids=coingecko_ids),
        ]
        if alpha_key:
            self.connectors.append(EarningsAlphaVantageConnector(api_key=alpha_key, horizon=os.getenv("EARNINGS_HORIZON", "3month")))

    def publish_event(self, event: Dict):
        occurred_at = event["occurred_at"]
        dedup_key = f"{event.get('source_name', '')}|{event.get('title', '').strip().lower()}|{event.get('source_url', '')}"
        dedup_cluster_id = hashlib.sha256(dedup_key.encode("utf-8")).hexdigest()[:24]
        latency_ms = None
        try:
            latency_ms = int((datetime.utcnow() - datetime.fromisoformat(occurred_at.replace("Z", "+00:00")).replace(tzinfo=None)).total_seconds() * 1000)
            if latency_ms < 0:
                latency_ms = 0
        except Exception:
            latency_ms = None

        enriched_entities = event.get("entities", []) or extract_entities(
            event.get("title", ""),
            str(event.get("payload", {}).get("summary", "")),
        )

        payload = {
            "event_type": event["event_type"],
            "title": event["title"],
            "occurred_at": occurred_at,
            "source_url": event.get("source_url") or "",
            "source_name": event.get("source_name") or "",
            "source_timezone": event.get("source_timezone", "UTC"),
            "source_tier": str(event.get("source_tier", 3)),
            "confidence_score": str(event.get("confidence_score", 0.5)),
            "event_importance": str(event.get("event_importance", event.get("confidence_score", 0.5))),
            "novelty_score": str(event.get("novelty_score", 0.5)),
            "entity_confidence": str(event.get("entity_confidence", 0.5)),
            "latency_ms": str(latency_ms or 0),
            "dedup_cluster_id": event.get("dedup_cluster_id") or dedup_cluster_id,
            "payload": json.dumps(event.get("payload", {}), ensure_ascii=False),
            "entities": json.dumps(enriched_entities, ensure_ascii=False),
            "ingested_at": datetime.utcnow().isoformat() + "Z",
        }

        self.redis.xadd(self.event_stream, payload)

        # Backward compatibility path for existing news consumers.
        self.redis.xadd(
            self.news_stream,
            {
                "title": event["title"],
                "url": event.get("source_url") or "",
                "symbol": "OTHER",
                "priority": "medium",
                "sentiment": "neutral",
                "time": datetime.utcnow().isoformat() + "Z",
                "summary": str(event.get("payload", {}))[:500],
            },
        )

    def run_once(self) -> int:
        total = 0
        for connector in self.connectors:
            try:
                raw_rows = connector.fetch()
                logger.info("connector=%s fetched=%d", connector.name, len(raw_rows))
                for raw in raw_rows:
                    event = connector.normalize(raw)
                    self.publish_event(event)
                    total += 1
            except Exception as exc:
                logger.error("connector=%s failed: %s", connector.name, exc)
        return total

    def run(self, interval: int = 300):
        logger.info("collector-v2 started interval=%ss", interval)
        while True:
            try:
                sent = self.run_once()
                logger.info("cycle published=%d", sent)
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("collector-v2 stopped")
                break
            except Exception as exc:
                logger.error("collector-v2 cycle error: %s", exc)
                time.sleep(30)


if __name__ == "__main__":
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379")
    interval = int(os.getenv("COLLECT_INTERVAL", "300"))
    DataCollectorV2(redis_url=redis_url).run(interval=interval)
