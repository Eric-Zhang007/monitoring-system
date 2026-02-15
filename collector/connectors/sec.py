from __future__ import annotations

import datetime as dt
from typing import Dict, List

import requests

from connectors.base import BaseConnector, RateLimitError


class SECSubmissionsConnector(BaseConnector):
    name = "sec_submissions"

    def __init__(self, cik_list: List[str], user_agent: str = "monitoring-system/2.0 contact@example.com"):
        self.cik_list = cik_list
        self.headers = {"User-Agent": user_agent}

    def fetch(self) -> List[Dict]:
        rows: List[Dict] = []
        for cik in self.cik_list:
            padded = cik.zfill(10)
            url = f"https://data.sec.gov/submissions/CIK{padded}.json"
            resp = requests.get(url, headers=self.headers, timeout=20)
            if resp.status_code == 429:
                raise RateLimitError("sec_rate_limited")
            if resp.status_code != 200:
                continue
            rows.append({"cik": cik, "data": resp.json()})
        return rows

    def normalize(self, raw: Dict) -> Dict:
        data = raw["data"]
        name = data.get("name", "Unknown")
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])

        form = forms[0] if forms else "UNKNOWN"
        filing_date = dates[0] if dates else dt.datetime.utcnow().date().isoformat()

        return {
            "event_type": "regulatory",
            "market_scope": "equity",
            "title": f"SEC filing {form} by {name}",
            "occurred_at": f"{filing_date}T00:00:00Z",
            "source_url": f"https://www.sec.gov/cgi-bin/browse-edgar?CIK={raw['cik']}",
            "source_name": "sec-edgar",
            "source_timezone": "UTC",
            "source_tier": 1,
            "confidence_score": 0.9,
            "event_importance": 0.9,
            "novelty_score": 0.6,
            "entity_confidence": 0.92,
            "payload": {
                "cik": raw["cik"],
                "form": form,
            },
            "entities": [
                {
                    "entity_type": "company",
                    "name": name,
                    "symbol": None,
                    "country": "US",
                    "sector": None,
                    "metadata": {"cik": raw["cik"]},
                }
            ],
        }
