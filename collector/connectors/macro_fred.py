from __future__ import annotations

import datetime as dt
from email.utils import parsedate_to_datetime
from typing import Dict, List
import xml.etree.ElementTree as ET

import requests
try:
    import feedparser  # type: ignore
except Exception:  # pragma: no cover
    feedparser = None

from connectors.base import BaseConnector


class MacroFREDConnector(BaseConnector):
    name = "macro_fred"

    def __init__(self, release_ids: List[str] | None = None):
        self.release_ids = release_ids or ["10", "53", "54"]  # GDP, CPI, unemployment related feeds

    def fetch(self) -> List[Dict]:
        rows: List[Dict] = []
        for rid in self.release_ids:
            url = f"https://fred.stlouisfed.org/releases?rid={rid}&output=1"
            if feedparser is not None:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    rows.append({"rid": rid, "entry": entry, "feed_url": url})
                continue
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
            except Exception:
                continue
            for item in root.findall(".//item")[:20]:
                rows.append(
                    {
                        "rid": rid,
                        "feed_url": url,
                        "entry": {
                            "title": (item.findtext("title") or "").strip(),
                            "summary": (item.findtext("description") or "").strip(),
                            "link": (item.findtext("link") or "").strip(),
                            "published": (item.findtext("pubDate") or "").strip(),
                        },
                    }
                )
        return rows

    def normalize(self, raw: Dict) -> Dict:
        entry = raw["entry"]
        published = entry.get("published") or entry.get("updated")
        occurred_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
        if published:
            try:
                occurred_at = parsedate_to_datetime(published).astimezone(dt.timezone.utc)
            except (TypeError, ValueError):
                pass

        title = entry.get("title", "FRED macro release")
        summary = entry.get("summary", "")
        return {
            "event_type": "market",
            "market_scope": "macro",
            "title": f"Macro release: {title}",
            "occurred_at": occurred_at.isoformat(),
            "source_url": entry.get("link"),
            "source_name": "fred",
            "source_timezone": "UTC",
            "source_tier": 1,
            "confidence_score": 0.9,
            "event_importance": 0.8,
            "novelty_score": 0.6,
            "entity_confidence": 0.4,
            "payload": {"summary": summary[:1200], "release_id": raw.get("rid")},
            "entities": [
                {
                    "entity_type": "asset",
                    "name": "US_MACRO",
                    "symbol": "US_MACRO",
                    "country": "US",
                    "sector": "macro",
                    "metadata": {"rid": raw.get("rid")},
                }
            ],
        }
