from __future__ import annotations

import datetime as dt
from typing import Dict, List

import requests

from connectors.base import BaseConnector, RateLimitError


class GDELTConnector(BaseConnector):
    name = "gdelt"

    def __init__(self, query: str = "venture capital startup funding", max_records: int = 20):
        self.query = query
        self.max_records = max_records

    def fetch(self) -> List[Dict]:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": self.query,
            "mode": "ArtList",
            "maxrecords": self.max_records,
            "format": "json",
            "sort": "datedesc",
        }
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            raise RateLimitError("gdelt_rate_limited")
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])

    def normalize(self, raw: Dict) -> Dict:
        ts = raw.get("seendate") or raw.get("socialimage")
        if isinstance(ts, str) and len(ts) >= 14 and ts[:14].isdigit():
            occurred = dt.datetime.strptime(ts[:14], "%Y%m%d%H%M%S").isoformat() + "Z"
        else:
            occurred = dt.datetime.utcnow().isoformat() + "Z"

        title = raw.get("title") or "Untitled"
        return {
            "event_type": "market",
            "market_scope": "equity",
            "title": title,
            "occurred_at": occurred,
            "source_url": raw.get("url"),
            "source_name": raw.get("domain") or "gdelt",
            "source_timezone": "UTC",
            "source_tier": 3,
            "confidence_score": 0.6,
            "event_importance": 0.58,
            "novelty_score": 0.62,
            "entity_confidence": 0.3,
            "payload": {
                "language": raw.get("language"),
                "tone": raw.get("tone"),
                "query": self.query,
            },
            "entities": [],
        }
