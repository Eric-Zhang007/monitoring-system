#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Set, Tuple

import requests


DEFAULT_QUERIES = [
    "bitcoin OR ethereum OR solana OR crypto market",
    "federal reserve OR FOMC OR inflation OR CPI OR interest rate crypto",
    "donald trump crypto OR us election crypto policy",
    "SEC crypto OR ETF approval bitcoin",
    "blackrock bitcoin etf inflow",
]


def _parse_dt_utc(raw: str) -> datetime:
    s = str(raw or "").strip()
    if not s:
        raise ValueError("empty_datetime")
    norm = s.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt_obj = datetime.fromisoformat(norm)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def _to_iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_hash_01(text: str) -> float:
    raw = str(text or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    base = float((16 ** 12) - 1)
    return float(int(digest, 16) / base) if base > 0 else 0.0


def _google_latency_minutes(title: str, link: str, occurred_at: datetime, source_tier: int = 2) -> float:
    jitter = _stable_hash_01(f"{title}|{link}|{_to_iso_z(occurred_at)}")
    base = 25.0 + (110.0 - 25.0) * jitter
    tier_penalty = max(0, int(source_tier) - 1) * 3.5
    return max(0.5, base + tier_penalty)


def _iter_windows(start: datetime, end: datetime, day_step: int) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    step = max(1, int(day_step))
    while cur < end:
        nxt = min(end, cur + timedelta(days=step))
        yield cur, nxt
        cur = nxt


def _build_google_rss_url(query: str, start: datetime, end: datetime) -> str:
    q = f"{query} after:{start.date().isoformat()} before:{end.date().isoformat()}"
    encoded = urllib.parse.quote(q)
    return f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"


def _extract_entities(text: str) -> List[Dict[str, object]]:
    t = str(text or "").lower()
    mapping = {
        "BTC": ("bitcoin", "btc"),
        "ETH": ("ethereum", "eth"),
        "SOL": ("solana", "sol"),
        "BNB": ("bnb", "binance"),
        "XRP": ("xrp", "ripple"),
        "ADA": ("ada", "cardano"),
        "DOGE": ("doge", "dogecoin"),
        "TRX": ("trx", "tron"),
        "AVAX": ("avax", "avalanche"),
        "LINK": ("link", "chainlink"),
    }
    out: List[Dict[str, object]] = []
    for symbol, kws in mapping.items():
        if any(k in t for k in kws):
            out.append(
                {
                    "entity_type": "asset",
                    "name": symbol,
                    "symbol": symbol,
                    "country": None,
                    "sector": "crypto",
                    "metadata": {"from": "keyword"},
                }
            )
    if not out:
        out.append(
            {
                "entity_type": "asset",
                "name": "US_MACRO",
                "symbol": "US_MACRO",
                "country": "US",
                "sector": "macro",
                "metadata": {"from": "fallback_macro"},
            }
        )
    return out


def _classify(title: str, query: str) -> Tuple[str, str, bool]:
    t = f"{title} {query}".lower()
    if any(k in t for k in ("sec", "regulation", "policy", "ban", "law")):
        event_type = "regulatory"
    else:
        event_type = "market"
    is_macro = any(
        k in t
        for k in (
            "federal reserve",
            "fomc",
            "cpi",
            "inflation",
            "interest rate",
            "unemployment",
            "trump",
            "election",
            "etf",
            "blackrock",
            "fidelity",
        )
    )
    return event_type, ("macro" if is_macro else "crypto"), bool(is_macro)


def _normalize_source(title: str) -> Tuple[str, str]:
    text = str(title or "").strip()
    if " - " in text:
        head, tail = text.rsplit(" - ", 1)
        source_name = tail.strip() or "google_news"
        clean_title = head.strip() or text
        return clean_title[:500], source_name[:120]
    return text[:500], "google_news"


def _fetch_rss_items(sess: requests.Session, url: str) -> List[Dict[str, str]]:
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        out.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pubDate": (item.findtext("pubDate") or "").strip(),
                "description": (item.findtext("description") or "").strip(),
            }
        )
    return out


def _dedup(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: Set[str] = set()
    out: List[Dict[str, object]] = []
    for ev in events:
        key = "|".join(
            [
                str(ev.get("title") or "").strip().lower(),
                str(ev.get("source_url") or "").strip().lower(),
                str(ev.get("occurred_at") or "").strip(),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Google News event backfill JSONL for 2025-now")
    ap.add_argument("--start", default="2025-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--day-step", type=int, default=7)
    ap.add_argument("--queries", default="||".join(DEFAULT_QUERIES), help="split by ||")
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--out-jsonl", default="artifacts/server_bundle/events_google_news_2025_now.jsonl")
    args = ap.parse_args()

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")

    queries = [q.strip() for q in str(args.queries).split("||") if q.strip()]
    if not queries:
        queries = list(DEFAULT_QUERIES)

    sess = requests.Session()
    all_events: List[Dict[str, object]] = []
    windows = 0
    reqs = 0
    for w_start, w_end in _iter_windows(start_dt, end_dt, day_step=int(args.day_step)):
        for q in queries:
            url = _build_google_rss_url(q, w_start, w_end)
            try:
                items = _fetch_rss_items(sess, url)
            except Exception:
                items = []
            reqs += 1
            for it in items:
                raw_title = str(it.get("title") or "")
                title, source_name = _normalize_source(raw_title)
                pub = str(it.get("pubDate") or "").strip()
                if not pub:
                    continue
                try:
                    occurred_at = parsedate_to_datetime(pub).astimezone(timezone.utc)
                except Exception:
                    continue
                event_type, market_scope, global_impact = _classify(title=title, query=q)
                lag_minutes = _google_latency_minutes(title=title, link=str(it.get("link") or ""), occurred_at=occurred_at, source_tier=2)
                available_at = occurred_at + timedelta(minutes=lag_minutes)
                source_latency_ms = int(max(0.0, lag_minutes * 60_000.0))
                all_events.append(
                    {
                        "event_type": event_type,
                        "title": title,
                        "occurred_at": _to_iso_z(occurred_at),
                        "published_at": _to_iso_z(occurred_at),
                        "available_at": _to_iso_z(available_at),
                        "effective_at": _to_iso_z(available_at),
                        "source_url": str(it.get("link") or ""),
                        "source_name": source_name,
                        "source_timezone": "UTC",
                        "source_tier": 2,
                        "confidence_score": 0.65 if market_scope == "crypto" else 0.75,
                        "event_importance": 0.7 if global_impact else 0.58,
                        "novelty_score": 0.55,
                        "entity_confidence": 0.5,
                        "latency_ms": source_latency_ms,
                        "source_latency_ms": source_latency_ms,
                        "market_scope": market_scope,
                        "payload": {
                            "provider": "google_news_rss",
                            "query": q,
                            "description": str(it.get("description") or "")[:1200],
                            "global_impact": global_impact,
                            "availability_lag_minutes": round(float(lag_minutes), 3),
                            "availability_model": "provider_tier_hash_v1",
                        },
                        "entities": _extract_entities(f"{title} {it.get('description') or ''}"),
                    }
                )
            if float(args.sleep_sec) > 0:
                time.sleep(float(args.sleep_sec))
        windows += 1

    deduped = _dedup(all_events)
    deduped.sort(key=lambda x: str(x.get("occurred_at") or ""))
    out_path = str(args.out_jsonl).strip()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in deduped:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "start": _to_iso_z(start_dt),
                "end": _to_iso_z(end_dt),
                "queries": len(queries),
                "windows": windows,
                "requests": reqs,
                "events_raw": len(all_events),
                "events_deduped": len(deduped),
                "out_jsonl": out_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
