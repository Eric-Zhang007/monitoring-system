from __future__ import annotations

from typing import Dict, List
import xml.etree.ElementTree as ET

import requests
try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None

from connectors.base import BaseConnector
from connectors.social_common import (
    estimate_engagement_score,
    extract_symbol_mentions,
    parse_datetime_to_iso_z,
    sentiment_score,
    social_event_type,
    social_payload,
    symbols_to_entities,
)


class TelegramConnector(BaseConnector):
    name = "social_telegram"

    def __init__(self, feeds: List[str] | None = None):
        self.feeds = [f.strip() for f in (feeds or []) if f and f.strip()]

    def fetch(self) -> List[Dict]:
        if not self.feeds:
            return []
        rows: List[Dict] = []
        for feed_url in self.feeds:
            if feedparser is not None:
                feed = feedparser.parse(feed_url)
                for entry in (feed.entries or [])[:40]:
                    rows.append({"feed": feed_url, "entry": entry})
                continue
            try:
                resp = requests.get(feed_url, timeout=20)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
            except Exception:
                continue
            for item in root.findall(".//item")[:40]:
                rows.append(
                    {
                        "feed": feed_url,
                        "entry": {
                            "title": (item.findtext("title") or "").strip(),
                            "summary": (item.findtext("description") or "").strip(),
                            "description": (item.findtext("description") or "").strip(),
                            "link": (item.findtext("link") or "").strip(),
                            "author": (item.findtext("author") or "").strip(),
                            "published": (item.findtext("pubDate") or "").strip(),
                        },
                    }
                )
        return rows

    def normalize(self, raw: Dict) -> Dict:
        entry = raw.get("entry") or {}
        title = str(entry.get("title") or "").strip()
        summary = str(entry.get("summary") or entry.get("description") or "").strip()
        text = f"{title}\n{summary}".strip()
        author = str(entry.get("author") or "telegram_channel")

        symbols = extract_symbol_mentions(text)
        post_s = sentiment_score(text)
        event_type, market_scope = social_event_type(text)

        payload = social_payload(
            social_platform="telegram",
            author=author,
            author_followers=0,
            engagement_score=estimate_engagement_score(likes=0, comments=0, shares=0, views=0),
            comment_sentiment=0.0,
            post_sentiment=post_s,
            n_comments=0,
            n_replies=0,
            is_verified=False,
            influence_tier="unknown",
            symbol_mentions=symbols,
            summary=text,
            extra={
                "telegram_feed": str(raw.get("feed") or ""),
                "telegram_mode": "rss_public_placeholder",
            },
        )

        return {
            "event_type": event_type,
            "market_scope": market_scope,
            "title": title[:220] if title else "Telegram post",
            "occurred_at": parse_datetime_to_iso_z(entry.get("published") or entry.get("updated")),
            "source_url": str(entry.get("link") or ""),
            "source_name": "telegram_rss",
            "source_timezone": "UTC",
            "source_tier": 4,
            "confidence_score": 0.5,
            "event_importance": 0.45,
            "novelty_score": 0.5,
            "entity_confidence": 0.42,
            "payload": payload,
            "entities": symbols_to_entities(symbols),
        }
