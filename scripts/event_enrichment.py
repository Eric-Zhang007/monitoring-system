from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_llm_enricher(*, force_enable: bool = False):
    repo_root = Path(__file__).resolve().parents[1]
    collector_dir = repo_root / "collector"
    if str(collector_dir) not in sys.path:
        sys.path.insert(0, str(collector_dir))
    from llm_enrichment import LLMEnricher  # type: ignore

    enricher = LLMEnricher()
    if force_enable:
        enricher.enabled = True
    return enricher


def _entity_key(entity: Dict[str, Any]) -> Tuple[str, str]:
    et = str(entity.get("entity_type") or "").strip().lower()
    symbol = str(entity.get("symbol") or "").strip().upper()
    name = str(entity.get("name") or "").strip().lower()
    return et, symbol or name


def _to_canonical_llm_entities(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in raw[:20]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        entity_type = str(row.get("entity_type") or "company").strip().lower()
        if entity_type not in {"asset", "company", "investor"}:
            entity_type = "company"
        symbol = str(row.get("symbol") or "").strip().upper().replace("$", "")
        if entity_type != "asset":
            symbol = ""
        item = {
            "entity_type": entity_type,
            "name": name[:120],
            "symbol": symbol or None,
            "country": None,
            "sector": "crypto" if entity_type == "asset" else None,
            "metadata": {"source": "llm_enrichment", "confidence": float(row.get("confidence", 0.5) or 0.5)},
        }
        key = _entity_key(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def apply_llm_enrichment(
    events: List[Dict[str, Any]],
    *,
    enricher,
    max_events: int = 0,
) -> Dict[str, Any]:
    if not events:
        return {"attempted": 0, "enriched": 0, "statuses": {}}
    limit = len(events) if int(max_events) <= 0 else min(len(events), int(max_events))
    status_counts: Dict[str, int] = {}
    enriched = 0
    for idx, event in enumerate(events):
        if idx >= limit:
            break
        llm = enricher.enrich_event(event)
        status = str(llm.get("status") or "unknown")
        status_counts[status] = int(status_counts.get(status, 0)) + 1

        payload = dict(event.get("payload") or {})
        payload["llm_enrichment"] = llm
        payload["llm_sentiment"] = float(llm.get("sentiment", 0.0) or 0.0)
        payload["llm_confidence"] = float(llm.get("confidence", 0.0) or 0.0)
        language = str(llm.get("language") or "").strip().lower()
        if language in {"en", "zh"}:
            payload["language"] = language
        if not str(payload.get("summary") or "").strip():
            summary = str(llm.get("summary") or "").strip()
            if summary:
                payload["summary"] = summary[:1200]
        event["payload"] = payload

        llm_entities = _to_canonical_llm_entities(llm.get("entities"))
        if llm_entities:
            existing = event.get("entities")
            if not isinstance(existing, list):
                existing = []
            existing_keys = {_entity_key(e) for e in existing if isinstance(e, dict)}
            for entity in llm_entities:
                key = _entity_key(entity)
                if key in existing_keys:
                    continue
                existing.append(entity)
                existing_keys.add(key)
            event["entities"] = existing
        enriched += 1
    return {"attempted": limit, "enriched": enriched, "statuses": status_counts}

