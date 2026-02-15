from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from typing import Dict, List

import feedparser

from connectors.base import BaseConnector


class RSSConnector(BaseConnector):
    name = "rss"

    def __init__(self, feeds: List[str]):
        self.feeds = feeds

    def fetch(self) -> List[Dict]:
        rows: List[Dict] = []
        for feed_url in self.feeds:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:20]:
                rows.append({"feed": feed_url, "entry": entry})
        return rows

    def normalize(self, raw: Dict) -> Dict:
        entry = raw["entry"]
        published = entry.get("published") or entry.get("updated")
        occurred_at = dt.datetime.utcnow()
        if published:
            try:
                occurred_at = parsedate_to_datetime(published)
            except (TypeError, ValueError):
                pass

        title = entry.get("title", "Untitled")
        summary = entry.get("summary", "")
        event_type = "funding" if "funding" in (title + summary).lower() else "market"

        return {
            "event_type": event_type,
            "market_scope": "crypto",
            "title": title,
            "occurred_at": occurred_at.astimezone(dt.timezone.utc).isoformat(),
            "source_url": entry.get("link"),
            "source_name": raw.get("feed"),
            "source_timezone": "UTC",
            "source_tier": 2,
            "confidence_score": 0.55,
            "event_importance": 0.52,
            "novelty_score": 0.5,
            "entity_confidence": 0.35,
            "payload": {"summary": summary[:1200]},
            "entities": [],
        }
