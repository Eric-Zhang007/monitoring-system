from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from schemas_v2 import Entity, Event, SocialIngestItem


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_utc(ts: datetime | None) -> datetime:
    if ts is None:
        return _utcnow()
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _clamp(v: float, low: float, high: float) -> float:
    return max(low, min(high, float(v)))


def _normalize_symbol(raw: str | None) -> str:
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    return "".join(ch for ch in text if ch.isalnum())[:16]


def _build_title(item: SocialIngestItem) -> str:
    title = str(item.title or "").strip()
    if title:
        return title[:320]
    text = str(item.text or "").strip()
    if not text:
        return f"social_{item.platform}_{item.kind}"
    return text[:320]


def _coverage_score(item: SocialIngestItem) -> float:
    followers = max(0.0, float(item.author_followers))
    engagement = max(0.0, float(item.engagement_score))
    has_text = 1.0 if str(item.text or "").strip() else 0.0
    score = 0.45 * min(1.0, followers / 100000.0) + 0.45 * min(1.0, engagement / 5000.0) + 0.10 * has_text
    return round(_clamp(score, 0.0, 1.0), 6)


def social_items_to_events(items: List[SocialIngestItem]) -> Tuple[List[Event], List[Dict[str, Any]]]:
    events: List[Event] = []
    enrich_rows: List[Dict[str, Any]] = []

    for item in items:
        platform = str(item.platform or "other").strip().lower() or "other"
        kind = str(item.kind or "post").strip().lower() or "post"
        symbol = _normalize_symbol(item.symbol)
        occurred_at = _as_utc(item.occurred_at)
        published_at = _as_utc(item.published_at) if item.published_at else occurred_at
        ingested_at = _as_utc(item.ingested_at) if item.ingested_at else _utcnow()
        available_at = _as_utc(item.available_at) if item.available_at else ingested_at
        sentiment = _clamp(0.5 * float(item.post_sentiment) + 0.5 * float(item.comment_sentiment), -1.0, 1.0)

        payload = {
            "social_platform": platform,
            "social_kind": kind,
            "author": str(item.author or ""),
            "author_followers": int(item.author_followers),
            "engagement_score": float(item.engagement_score),
            "post_sentiment": float(item.post_sentiment),
            "comment_sentiment": float(item.comment_sentiment),
            "symbol_mentions": [symbol] if symbol else [],
            **dict(item.metadata or {}),
        }

        entities: List[Entity] = list(item.entities or [])
        if symbol and not any((e.entity_type == "asset" and str(e.symbol or "").strip().upper() == symbol) for e in entities):
            entities.append(
                Entity(
                    entity_type="asset",
                    name=symbol,
                    symbol=symbol,
                    country=None,
                    sector="crypto",
                    metadata={"source": "social_symbol"},
                )
            )

        events.append(
            Event(
                event_type="market",
                title=_build_title(item),
                occurred_at=occurred_at,
                published_at=published_at,
                ingested_at=ingested_at,
                available_at=available_at,
                effective_at=available_at,
                source_url=item.source_url,
                source_name=f"social:{platform}",
                source_timezone="UTC",
                source_tier=int(item.source_tier),
                confidence_score=float(item.confidence_score),
                event_importance=max(0.0, min(1.0, 0.35 + 0.65 * abs(sentiment))),
                novelty_score=float(_clamp(float((item.metadata or {}).get("novelty_score", 0.5) or 0.5), 0.0, 1.0)),
                entity_confidence=max(0.0, min(1.0, 0.4 + 0.6 * float(item.confidence_score))),
                latency_ms=max(0, int((available_at - occurred_at).total_seconds() * 1000.0)),
                source_latency_ms=max(0, int((ingested_at - occurred_at).total_seconds() * 1000.0)),
                dedup_cluster_id=str((item.metadata or {}).get("dedup_cluster_id") or ""),
                market_scope="crypto",
                payload=payload,
                entities=entities,
            )
        )

        embedding = (item.metadata or {}).get("embedding")
        enrich_rows.append(
            {
                "social_platform": platform,
                "social_kind": kind,
                "language": str((item.metadata or {}).get("language") or ""),
                "summary": str(item.text or item.title or "")[:600],
                "sentiment": float(sentiment),
                "author": str(item.author or ""),
                "author_followers": int(item.author_followers),
                "engagement_score": float(item.engagement_score),
                "embedding": embedding if isinstance(embedding, list) else [],
                "observed_at": available_at,
                "event_time": occurred_at,
                "ingest_lag_sec": max(0.0, (available_at - occurred_at).total_seconds()),
                "coverage_score": _coverage_score(item),
                "payload": dict(item.metadata or {}),
            }
        )

    return events, enrich_rows

