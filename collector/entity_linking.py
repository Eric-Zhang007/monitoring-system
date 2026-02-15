from __future__ import annotations

import re
from typing import Dict, List

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
WORD_RE = re.compile(r"\b[A-Za-z]{3,16}\b")

ALIAS_MAP = {
    "BTC": "BTC",
    "BITCOIN": "BTC",
    "XBT": "BTC",
    "ETH": "ETH",
    "ETHEREUM": "ETH",
    "SOL": "SOL",
    "SOLANA": "SOL",
}


def extract_entities(title: str, summary: str = "") -> List[Dict]:
    text = f"{title} {summary}".strip()
    if not text:
        return []
    entities: List[Dict] = []
    seen = set()
    tokens = [w.upper() for w in WORD_RE.findall(text)]
    for t in tokens:
        normalized = ALIAS_MAP.get(t)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        entities.append(
            {
                "entity_type": "asset",
                "name": normalized,
                "symbol": normalized,
                "country": None,
                "sector": "crypto",
                "metadata": {"source": "alias_map", "raw": t},
            }
        )

    for m in TICKER_RE.findall(text):
        if m in {"THE", "WITH", "FROM", "THIS", "THAT"}:
            continue
        normalized = ALIAS_MAP.get(m, m)
        if normalized in seen:
            continue
        seen.add(normalized)
        entities.append(
            {
                "entity_type": "asset",
                "name": normalized,
                "symbol": normalized,
                "country": None,
                "sector": None,
                "metadata": {"source": "regex_ticker", "raw": m},
            }
        )
    return entities[:5]
