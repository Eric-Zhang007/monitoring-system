from __future__ import annotations

import datetime as dt
from typing import Dict, List

import requests

from connectors.base import BaseConnector, RateLimitError


class OnChainCoinGeckoConnector(BaseConnector):
    name = "onchain_coingecko"

    def __init__(self, ids: List[str] | None = None, vs_currency: str = "usd"):
        self.ids = ids or ["bitcoin", "ethereum", "solana"]
        self.vs_currency = vs_currency

    def fetch(self) -> List[Dict]:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": self.vs_currency,
            "ids": ",".join(self.ids),
            "order": "market_cap_desc",
            "per_page": max(1, len(self.ids)),
            "page": 1,
            "sparkline": "false",
            "price_change_percentage": "24h,7d",
        }
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 429:
            raise RateLimitError("coingecko_rate_limited")
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    def normalize(self, raw: Dict) -> Dict:
        symbol = (raw.get("symbol") or "UNK").upper()
        name = raw.get("name") or symbol
        occurred_at = dt.datetime.utcnow().isoformat() + "Z"
        return {
            "event_type": "market",
            "market_scope": "crypto",
            "title": f"On-chain/market snapshot: {name}",
            "occurred_at": occurred_at,
            "source_url": f"https://www.coingecko.com/en/coins/{raw.get('id', '')}",
            "source_name": "coingecko",
            "source_timezone": "UTC",
            "source_tier": 2,
            "confidence_score": 0.7,
            "event_importance": 0.65,
            "novelty_score": 0.5,
            "entity_confidence": 0.85,
            "payload": {
                "current_price": raw.get("current_price"),
                "market_cap": raw.get("market_cap"),
                "total_volume": raw.get("total_volume"),
                "price_change_percentage_24h": raw.get("price_change_percentage_24h"),
                "price_change_percentage_7d_in_currency": raw.get("price_change_percentage_7d_in_currency"),
            },
            "entities": [
                {
                    "entity_type": "asset",
                    "name": symbol,
                    "symbol": symbol,
                    "country": None,
                    "sector": "crypto",
                    "metadata": {"id": raw.get("id"), "source": "coingecko"},
                }
            ],
        }
