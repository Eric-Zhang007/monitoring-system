from __future__ import annotations

import re
from typing import Dict, List

TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")


def extract_entities(title: str, summary: str = "") -> List[Dict]:
    text = f"{title} {summary}".strip()
    if not text:
        return []
    entities: List[Dict] = []
    seen = set()
    for m in TICKER_RE.findall(text):
        if m in {"THE", "WITH", "FROM", "THIS", "THAT"}:
            continue
        if m in seen:
            continue
        seen.add(m)
        entities.append(
            {
                "entity_type": "asset",
                "name": m,
                "symbol": m,
                "country": "US",
                "sector": None,
                "metadata": {"source": "regex_ticker"},
            }
        )
    return entities[:5]
