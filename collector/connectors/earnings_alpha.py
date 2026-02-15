from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Dict, List

import requests

from connectors.base import BaseConnector, RateLimitError


class EarningsAlphaVantageConnector(BaseConnector):
    name = "earnings_alpha_vantage"

    def __init__(self, api_key: str, horizon: str = "3month"):
        self.api_key = api_key
        self.horizon = horizon

    def fetch(self) -> List[Dict]:
        if not self.api_key:
            return []
        url = "https://www.alphavantage.co/query"
        params = {"function": "EARNINGS_CALENDAR", "horizon": self.horizon, "apikey": self.api_key}
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            raise RateLimitError("alphavantage_rate_limited")
        resp.raise_for_status()
        text = resp.text.strip()
        if not text:
            return []
        rows: List[Dict] = []
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            rows.append(row)
        return rows[:300]

    def normalize(self, raw: Dict) -> Dict:
        symbol = (raw.get("symbol") or "UNKNOWN").upper()
        report_date = raw.get("reportDate") or dt.datetime.utcnow().date().isoformat()
        occurred_at = f"{report_date}T00:00:00Z"
        eps_est = raw.get("estimate")
        return {
            "event_type": "market",
            "market_scope": "equity",
            "title": f"Earnings calendar: {symbol}",
            "occurred_at": occurred_at,
            "source_url": "https://www.alphavantage.co/documentation/",
            "source_name": "alpha_vantage",
            "source_timezone": "UTC",
            "source_tier": 2,
            "confidence_score": 0.8,
            "event_importance": 0.75,
            "novelty_score": 0.55,
            "entity_confidence": 0.8,
            "payload": {
                "fiscalDateEnding": raw.get("fiscalDateEnding"),
                "estimate": eps_est,
                "currency": raw.get("currency"),
            },
            "entities": [
                {
                    "entity_type": "asset",
                    "name": symbol,
                    "symbol": symbol,
                    "country": "US",
                    "sector": None,
                    "metadata": {"source": "alpha_vantage"},
                }
            ],
        }
