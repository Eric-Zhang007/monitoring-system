#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


SOCIAL_PAYLOAD_KEYS = (
    "social_platform",
    "author",
    "author_followers",
    "engagement_score",
    "comment_sentiment",
    "post_sentiment",
    "n_comments",
    "n_replies",
    "is_verified",
    "influence_tier",
    "symbol_mentions",
)


def _parse_datetime(raw: object) -> str:
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(float(raw), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if text.isdigit():
        return datetime.fromtimestamp(float(text), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        if "." in text:
            return datetime.fromtimestamp(float(text), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except ValueError:
        pass
    candidate = text.replace(" ", "T")
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_symbols(raw: object) -> List[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
    else:
        values = []
    seen = set()
    out: List[str] = []
    for val in values:
        token = str(val or "").strip().upper().replace("$", "")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _guess_sentiment_from_text(text: str) -> float:
    content = str(text or "").lower()
    pos = sum(k in content for k in ("bull", "bullish", "surge", "rally", "upgrade", "approval"))
    neg = sum(k in content for k in ("bear", "bearish", "crash", "drop", "hack", "lawsuit", "ban"))
    if pos == 0 and neg == 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg)))


def _average_comment_sentiment(raw_comments: object) -> float:
    if not isinstance(raw_comments, list):
        return 0.0
    vals: List[float] = []
    for item in raw_comments:
        if isinstance(item, dict) and item.get("sentiment") is not None:
            vals.append(_safe_float(item.get("sentiment"), default=0.0))
    if not vals:
        return 0.0
    return max(-1.0, min(1.0, sum(vals) / len(vals)))


def _payload_social_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    out.setdefault("social_platform", "none")
    out.setdefault("author", "")
    out["author_followers"] = max(0, _safe_int(out.get("author_followers"), default=0))
    out["engagement_score"] = max(0.0, _safe_float(out.get("engagement_score"), default=0.0))
    out["comment_sentiment"] = max(-1.0, min(1.0, _safe_float(out.get("comment_sentiment"), default=0.0)))
    out["post_sentiment"] = max(-1.0, min(1.0, _safe_float(out.get("post_sentiment"), default=0.0)))
    out["n_comments"] = max(0, _safe_int(out.get("n_comments"), default=0))
    out["n_replies"] = max(0, _safe_int(out.get("n_replies"), default=0))
    out["is_verified"] = bool(out.get("is_verified", False))
    out["influence_tier"] = str(out.get("influence_tier") or "unknown")
    out["symbol_mentions"] = _normalize_symbols(out.get("symbol_mentions"))
    return out


def _ensure_alignment_and_provenance_payload(payload: Dict[str, Any], event_times: Dict[str, str]) -> Dict[str, Any]:
    out = dict(payload or {})
    out["time_alignment"] = dict(out.get("time_alignment") or {})
    out["time_alignment"].setdefault("alignment_mode", "strict_asof_v1")
    out["time_alignment"].setdefault("occurred_at", str(event_times.get("occurred_at") or ""))
    out["time_alignment"].setdefault("published_at", str(event_times.get("published_at") or ""))
    out["time_alignment"].setdefault("available_at", str(event_times.get("available_at") or ""))
    out["time_alignment"].setdefault("effective_at", str(event_times.get("effective_at") or ""))
    out["time_alignment"].setdefault("monotonic_non_decreasing", True)
    out["provenance"] = dict(out.get("provenance") or {})
    out["provenance"].setdefault("source_script", "scripts/import_social_events_jsonl.py")
    out["provenance"].setdefault("ingest_mode", "social_jsonl_import")
    out["provenance"].setdefault("ingested_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    return out


def _symbols_to_entities(symbols: Iterable[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for symbol in _normalize_symbols(list(symbols)):
        out.append(
            {
                "entity_type": "asset",
                "name": symbol,
                "symbol": symbol,
                "country": None,
                "sector": "crypto",
                "metadata": {"source": "social_import"},
            }
        )
    return out


def to_canonical_event(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("row_must_be_object")

    payload = dict(raw.get("payload") or {}) if isinstance(raw.get("payload"), dict) else {}

    # Canonical row path: keep schema and force social payload normalization.
    if raw.get("event_type") and raw.get("title") and raw.get("occurred_at"):
        canonical = dict(raw)
        canonical_payload = _payload_social_defaults(payload)
        if "summary" not in canonical_payload:
            canonical_payload["summary"] = str(canonical.get("title") or "")
        canonical["payload"] = canonical_payload
        canonical["entities"] = canonical.get("entities") or _symbols_to_entities(canonical_payload.get("symbol_mentions") or [])
        canonical["occurred_at"] = _parse_datetime(canonical.get("occurred_at"))
        canonical["published_at"] = _parse_datetime(canonical.get("published_at") or canonical["occurred_at"])
        canonical["available_at"] = _parse_datetime(canonical.get("available_at") or canonical["published_at"])
        canonical["effective_at"] = _parse_datetime(canonical.get("effective_at") or canonical["available_at"])
        canonical_payload = _ensure_alignment_and_provenance_payload(
            canonical_payload,
            event_times={
                "occurred_at": str(canonical.get("occurred_at") or ""),
                "published_at": str(canonical.get("published_at") or ""),
                "available_at": str(canonical.get("available_at") or ""),
                "effective_at": str(canonical.get("effective_at") or ""),
            },
        )
        canonical["payload"] = canonical_payload
        canonical["event_type"] = str(canonical.get("event_type") or "market")
        canonical["market_scope"] = str(canonical.get("market_scope") or "crypto")
        return canonical

    text = str(raw.get("text") or raw.get("content") or raw.get("body") or raw.get("title") or "").strip()
    title = str(raw.get("title") or text[:220] or "social post")
    platform = str(raw.get("social_platform") or raw.get("platform") or payload.get("social_platform") or "unknown").strip().lower()
    author = str(raw.get("author") or payload.get("author") or "unknown").strip()
    symbols = _normalize_symbols(raw.get("symbol_mentions") or payload.get("symbol_mentions") or [])

    post_sentiment = raw.get("post_sentiment")
    if post_sentiment is None:
        post_sentiment = payload.get("post_sentiment")
    if post_sentiment is None:
        post_sentiment = _guess_sentiment_from_text(text)

    comment_sentiment = raw.get("comment_sentiment")
    if comment_sentiment is None:
        comment_sentiment = payload.get("comment_sentiment")
    if comment_sentiment is None:
        comment_sentiment = _average_comment_sentiment(raw.get("comments"))

    occurred_at = _parse_datetime(raw.get("occurred_at") or raw.get("created_at") or raw.get("published_at"))
    followers = raw.get("author_followers")
    if followers is None:
        followers = payload.get("author_followers")

    n_comments = raw.get("n_comments")
    if n_comments is None:
        n_comments = payload.get("n_comments")
    if n_comments is None:
        n_comments = raw.get("comment_count")

    n_replies = raw.get("n_replies")
    if n_replies is None:
        n_replies = payload.get("n_replies")
    if n_replies is None:
        n_replies = raw.get("reply_count")

    engagement_score = raw.get("engagement_score")
    if engagement_score is None:
        engagement_score = payload.get("engagement_score")
    if engagement_score is None:
        engagement_score = _safe_float(raw.get("likes"), 0.0) + _safe_float(n_comments, 0.0) * 2.0 + _safe_float(raw.get("shares"), 0.0) * 1.8

    payload_in = {
        "summary": text[:1200],
        "social_platform": platform,
        "author": author,
        "author_followers": followers if followers is not None else 0,
        "engagement_score": engagement_score,
        "comment_sentiment": comment_sentiment,
        "post_sentiment": post_sentiment,
        "n_comments": n_comments if n_comments is not None else 0,
        "n_replies": n_replies if n_replies is not None else 0,
        "is_verified": raw.get("is_verified") if raw.get("is_verified") is not None else payload.get("is_verified", False),
        "influence_tier": raw.get("influence_tier") or payload.get("influence_tier") or "unknown",
        "symbol_mentions": symbols,
    }
    for key in ("source_id", "subreddit", "video_id", "tweet_id"):
        if raw.get(key) is not None:
            payload_in[key] = raw.get(key)

    out_payload = _payload_social_defaults(payload_in)

    event_type = str(raw.get("event_type") or "market").strip().lower()
    if event_type not in {"funding", "mna", "product", "regulatory", "market"}:
        event_type = "market"

    market_scope = str(raw.get("market_scope") or "crypto").strip().lower()
    if market_scope not in {"crypto", "equity", "macro"}:
        market_scope = "crypto"

    source_tier = max(1, min(5, _safe_int(raw.get("source_tier"), default=3)))
    confidence = max(0.0, min(1.0, _safe_float(raw.get("confidence_score"), default=0.55)))
    importance = max(0.0, min(1.0, _safe_float(raw.get("event_importance"), default=confidence)))
    novelty = max(0.0, min(1.0, _safe_float(raw.get("novelty_score"), default=0.5)))
    entity_conf = max(0.0, min(1.0, _safe_float(raw.get("entity_confidence"), default=0.5)))

    occurred_at_iso = occurred_at
    published_at_iso = _parse_datetime(raw.get("published_at") or occurred_at_iso)
    available_at_iso = _parse_datetime(raw.get("available_at") or occurred_at_iso)
    effective_at_iso = _parse_datetime(raw.get("effective_at") or occurred_at_iso)
    out_payload = _ensure_alignment_and_provenance_payload(
        out_payload,
        event_times={
            "occurred_at": occurred_at_iso,
            "published_at": published_at_iso,
            "available_at": available_at_iso,
            "effective_at": effective_at_iso,
        },
    )

    return {
        "event_type": event_type,
        "title": title[:500],
        "occurred_at": occurred_at_iso,
        "published_at": published_at_iso,
        "available_at": available_at_iso,
        "effective_at": effective_at_iso,
        "source_url": str(raw.get("source_url") or raw.get("url") or ""),
        "source_name": str(raw.get("source_name") or f"social_{platform}"),
        "source_timezone": str(raw.get("source_timezone") or "UTC"),
        "source_tier": source_tier,
        "confidence_score": confidence,
        "event_importance": importance,
        "novelty_score": novelty,
        "entity_confidence": entity_conf,
        "latency_ms": max(0, _safe_int(raw.get("latency_ms"), default=0)),
        "source_latency_ms": max(0, _safe_int(raw.get("source_latency_ms"), default=0)),
        "market_scope": market_scope,
        "payload": out_payload,
        "entities": _symbols_to_entities(out_payload.get("symbol_mentions") or []),
    }


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _batch(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    n = max(1, int(size))
    return [items[i : i + n] for i in range(0, len(items), n)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Import social JSONL into canonical events/entities/event_links")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    jsonl = str(args.jsonl).strip()
    if not os.path.exists(jsonl):
        raise FileNotFoundError(jsonl)

    repo_root = Path(__file__).resolve().parents[1]
    backend_dir = repo_root / "backend"
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    from schemas_v2 import Event  # type: ignore
    from v2_repository import V2Repository  # type: ignore

    rows = _load_jsonl(jsonl)
    repo = V2Repository(db_url=args.database_url)

    accepted = 0
    inserted = 0
    deduped = 0
    event_ids: List[int] = []
    converted = 0

    for chunk in _batch(rows, size=int(args.batch_size)):
        events = []
        for raw in chunk:
            try:
                canonical = to_canonical_event(raw)
                events.append(Event(**canonical))
                converted += 1
            except Exception:
                continue
        if not events:
            continue
        a, i, d, ids = repo.ingest_events(events)
        accepted += int(a)
        inserted += int(i)
        deduped += int(d)
        event_ids.extend([int(x) for x in ids])

    print(
        json.dumps(
            {
                "status": "ok",
                "jsonl": jsonl,
                "rows_loaded": len(rows),
                "rows_converted": converted,
                "accepted": accepted,
                "inserted": inserted,
                "deduplicated": deduped,
                "event_ids_count": len(event_ids),
                "social_payload_keys": list(SOCIAL_PAYLOAD_KEYS),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
