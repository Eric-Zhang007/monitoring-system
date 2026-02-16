#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.parse
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Set, Tuple

import requests


GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_SEARCH_QUERIES = [
    "bitcoin OR ethereum OR solana OR crypto market",
    "federal reserve OR FOMC OR inflation OR CPI OR interest rate crypto",
    "donald trump crypto OR us election crypto policy",
    "SEC crypto OR ETF approval bitcoin",
    "blackrock bitcoin etf inflow",
]
DEFAULT_OFFICIAL_FEEDS = {
    "federal_reserve": [
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "https://www.federalreserve.gov/feeds/press_all.xml",
    ],
    "sec": [
        "https://www.sec.gov/news/pressreleases.rss",
    ],
    "cftc": [
        "https://www.cftc.gov/PressRoom/PressReleases/rss.xml",
    ],
    "whitehouse": [
        "https://www.whitehouse.gov/briefing-room/statements-releases/feed/",
    ],
    "crypto_media": [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://cryptonews.com/news/feed/",
    ],
}


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


def _iter_windows(start: datetime, end: datetime, day_step: int) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    step = max(1, int(day_step))
    while cur < end:
        nxt = min(end, cur + timedelta(days=step))
        yield cur, nxt
        cur = nxt


def _normalize_title_source(title: str) -> Tuple[str, str]:
    t = str(title or "").strip()
    if " - " in t:
        head, tail = t.rsplit(" - ", 1)
        return (head.strip() or t)[:500], (tail.strip() or "unknown")[:120]
    return t[:500], "unknown"


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


def _classify(title: str, query: str = "") -> Tuple[str, str, bool]:
    t = f"{title} {query}".lower()
    if any(k in t for k in ("sec", "regulation", "policy", "ban", "law", "enforcement")):
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
            "treasury",
            "fed",
        )
    )
    return event_type, ("macro" if is_macro else "crypto"), bool(is_macro)


def _is_relevant(title: str, summary: str = "") -> bool:
    t = f"{title} {summary}".lower()
    keys = (
        "crypto",
        "bitcoin",
        "ethereum",
        "solana",
        "altcoin",
        "etf",
        "federal reserve",
        "fomc",
        "interest rate",
        "inflation",
        "cpi",
        "unemployment",
        "trump",
        "sec",
        "regulation",
        "stablecoin",
    )
    return any(k in t for k in keys)


def _rss_items(sess: requests.Session, url: str) -> List[Dict[str, str]]:
    r = sess.get(url, timeout=30)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    out: List[Dict[str, str]] = []
    for item in root.findall(".//item"):
        out.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pubDate": (item.findtext("pubDate") or item.findtext("dc:date") or "").strip(),
                "description": (item.findtext("description") or "").strip(),
            }
        )
    return out


def _build_event(
    *,
    title: str,
    link: str,
    ts: datetime,
    source_name: str,
    provider: str,
    query: str,
    summary: str,
    source_tier: int,
) -> Dict[str, object]:
    event_type, market_scope, global_impact = _classify(title=title, query=query)
    confidence = 0.7 if market_scope == "macro" else 0.65
    lag_minutes = _estimate_source_latency_minutes(
        provider=provider,
        source_name=source_name,
        title=title,
        link=link,
        occurred_at=ts,
        source_tier=source_tier,
    )
    available_at = ts + timedelta(minutes=lag_minutes)
    source_latency_ms = int(max(0.0, lag_minutes * 60_000.0))
    return {
        "event_type": event_type,
        "title": title[:500],
        "occurred_at": _to_iso_z(ts),
        "published_at": _to_iso_z(ts),
        "available_at": _to_iso_z(available_at),
        "effective_at": _to_iso_z(available_at),
        "source_url": link,
        "source_name": source_name[:120],
        "source_timezone": "UTC",
        "source_tier": int(source_tier),
        "confidence_score": confidence,
        "event_importance": 0.72 if global_impact else 0.58,
        "novelty_score": 0.55,
        "entity_confidence": 0.5,
        "latency_ms": source_latency_ms,
        "source_latency_ms": source_latency_ms,
        "market_scope": market_scope,
        "payload": {
            "provider": provider,
            "query": query,
            "description": summary[:1200],
            "global_impact": global_impact,
            "availability_lag_minutes": round(float(lag_minutes), 3),
            "availability_model": "provider_tier_hash_v1",
        },
        "entities": _extract_entities(f"{title} {summary}"),
    }


def _stable_hash_01(text: str) -> float:
    raw = str(text or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    base = float((16 ** 12) - 1)
    return float(int(digest, 16) / base) if base > 0 else 0.0


def _estimate_source_latency_minutes(
    *,
    provider: str,
    source_name: str,
    title: str,
    link: str,
    occurred_at: datetime,
    source_tier: int,
) -> float:
    p = str(provider or "").strip().lower()
    s = str(source_name or "").strip().lower()
    # Conservative latency priors: official feeds are fastest, broad aggregators slower.
    if p == "google_news_rss":
        low, high = 25.0, 110.0
    elif p == "gdelt":
        low, high = 18.0, 70.0
    elif p.startswith("rss:federal_reserve") or p.startswith("rss:sec") or p.startswith("rss:cftc") or p.startswith("rss:whitehouse"):
        low, high = 2.0, 18.0
    elif p.startswith("rss:crypto_media"):
        low, high = 8.0, 35.0
    elif p.startswith("rss:"):
        low, high = 6.0, 30.0
    else:
        low, high = 10.0, 45.0
    tier_penalty = max(0, int(source_tier) - 1) * 3.5
    seed = f"{p}|{s}|{title}|{link}|{_to_iso_z(occurred_at)}"
    jitter = _stable_hash_01(seed)
    base = low + (high - low) * jitter
    return max(0.5, base + tier_penalty)


def _collect_google_news(
    sess: requests.Session,
    start: datetime,
    end: datetime,
    queries: List[str],
    day_step: int,
    sleep_sec: float,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for w_start, w_end in _iter_windows(start, end, day_step):
        for q in queries:
            qq = f"{q} after:{w_start.date().isoformat()} before:{w_end.date().isoformat()}"
            url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(qq) + "&hl=en-US&gl=US&ceid=US:en"
            try:
                items = _rss_items(sess, url)
            except Exception:
                items = []
            for it in items:
                raw_title = str(it.get("title") or "")
                title, src_tail = _normalize_title_source(raw_title)
                summary = str(it.get("description") or "")
                if not _is_relevant(title, summary):
                    continue
                try:
                    ts = parsedate_to_datetime(str(it.get("pubDate") or "")).astimezone(timezone.utc)
                except Exception:
                    continue
                events.append(
                    _build_event(
                        title=title,
                        link=str(it.get("link") or ""),
                        ts=ts,
                        source_name=src_tail or "google_news",
                        provider="google_news_rss",
                        query=q,
                        summary=summary,
                        source_tier=2,
                    )
                )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return events


def _gdelt_ts(raw: str) -> datetime:
    s = str(raw or "").strip()
    if "T" in s and len(s) >= 16:
        try:
            return datetime.strptime(s[:16], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if len(s) >= 14 and s[:14].isdigit():
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _collect_gdelt(
    sess: requests.Session,
    start: datetime,
    end: datetime,
    queries: List[str],
    day_step: int,
    max_records: int,
    sleep_sec: float,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for w_start, w_end in _iter_windows(start, end, day_step):
        for q in queries:
            params = {
                "query": q,
                "mode": "ArtList",
                "maxrecords": str(max(10, int(max_records))),
                "format": "json",
                "sort": "datedesc",
                "startdatetime": w_start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": w_end.strftime("%Y%m%d%H%M%S"),
            }
            try:
                r = sess.get(GDELT_URL, params=params, timeout=30)
                if r.status_code == 429:
                    if sleep_sec > 0:
                        time.sleep(sleep_sec * 2.0)
                    continue
                r.raise_for_status()
                rows = (r.json() or {}).get("articles") or []
            except Exception:
                rows = []
            for row in rows:
                title = str((row or {}).get("title") or "").strip()
                if not _is_relevant(title):
                    continue
                ts = _gdelt_ts(str((row or {}).get("seendate") or ""))
                events.append(
                    _build_event(
                        title=title,
                        link=str((row or {}).get("url") or ""),
                        ts=ts,
                        source_name=str((row or {}).get("domain") or "gdelt"),
                        provider="gdelt",
                        query=q,
                        summary=str((row or {}).get("socialimage") or ""),
                        source_tier=2,
                    )
                )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return events


def _collect_official_rss(
    sess: requests.Session,
    start: datetime,
    end: datetime,
    feeds: Dict[str, List[str]],
    sleep_sec: float,
) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    tier_map = {
        "federal_reserve": 1,
        "sec": 1,
        "cftc": 1,
        "whitehouse": 1,
        "crypto_media": 2,
    }
    for source_group, urls in feeds.items():
        for url in urls:
            try:
                items = _rss_items(sess, url)
            except Exception:
                items = []
            for it in items:
                title = str(it.get("title") or "").strip()
                summary = str(it.get("description") or "").strip()
                if not _is_relevant(title, summary):
                    continue
                pub = str(it.get("pubDate") or "").strip()
                if not pub:
                    continue
                try:
                    ts = parsedate_to_datetime(pub).astimezone(timezone.utc)
                except Exception:
                    continue
                if ts < start or ts > end:
                    continue
                source_name = source_group
                events.append(
                    _build_event(
                        title=title,
                        link=str(it.get("link") or ""),
                        ts=ts,
                        source_name=source_name,
                        provider=f"rss:{source_group}",
                        query=source_group,
                        summary=summary,
                        source_tier=tier_map.get(source_group, 2),
                    )
                )
            if sleep_sec > 0:
                time.sleep(sleep_sec)
    return events


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


def _provider_of(event: Dict[str, object]) -> str:
    payload = event.get("payload")
    if isinstance(payload, dict):
        provider = str(payload.get("provider") or "").strip().lower()
        if provider:
            return provider
    return "unknown"


def _provider_counts(events: List[Dict[str, object]]) -> Dict[str, int]:
    cnt = Counter()
    for ev in events:
        cnt[_provider_of(ev)] += 1
    return dict(sorted(cnt.items(), key=lambda x: (-x[1], x[0])))


def _event_pick_score(event: Dict[str, object]) -> float:
    title = str(event.get("title") or "")
    src = str(event.get("source_url") or "")
    ts = str(event.get("occurred_at") or "")
    jitter = _stable_hash_01(f"{title}|{src}|{ts}")
    imp = float(event.get("event_importance") or 0.0)
    conf = float(event.get("confidence_score") or 0.0)
    nov = float(event.get("novelty_score") or 0.0)
    return 0.6 * imp + 0.25 * conf + 0.1 * nov + 0.05 * jitter


def _cap_provider_share(
    events: List[Dict[str, object]],
    *,
    provider: str,
    max_share: float,
    min_keep: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not events:
        return [], {"status": "empty"}
    provider_norm = str(provider or "").strip().lower()
    max_share = max(0.05, min(0.95, float(max_share)))
    min_keep = max(0, int(min_keep))
    target: List[Dict[str, object]] = []
    others: List[Dict[str, object]] = []
    for ev in events:
        if _provider_of(ev) == provider_norm:
            target.append(ev)
        else:
            others.append(ev)
    if not target:
        return events, {"status": "noop_provider_absent", "provider": provider_norm}
    other_n = len(others)
    if other_n > 0:
        allowed = int((max_share / max(1e-9, 1.0 - max_share)) * other_n)
    else:
        allowed = int(max_share * len(events))
    keep_n = max(min_keep, allowed)
    keep_n = max(1, min(len(target), keep_n))
    if keep_n >= len(target):
        before_share = len(target) / max(1, len(events))
        return events, {
            "status": "noop_under_cap",
            "provider": provider_norm,
            "provider_before": len(target),
            "provider_after": len(target),
            "share_before": round(before_share, 4),
            "share_after": round(before_share, 4),
        }
    day_buckets: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for ev in target:
        day_key = str(ev.get("occurred_at") or "")[:10] or "unknown"
        day_buckets[day_key].append(ev)
    ordered_days = sorted(day_buckets.keys())
    for day in ordered_days:
        day_buckets[day].sort(key=_event_pick_score, reverse=True)
    selected: List[Dict[str, object]] = []
    while len(selected) < keep_n:
        progressed = False
        for day in ordered_days:
            bucket = day_buckets[day]
            if not bucket:
                continue
            selected.append(bucket.pop(0))
            progressed = True
            if len(selected) >= keep_n:
                break
        if not progressed:
            break
    merged = others + selected
    merged.sort(key=lambda x: str(x.get("occurred_at") or ""))
    before_share = len(target) / max(1, len(events))
    after_share = len(selected) / max(1, len(merged))
    meta = {
        "status": "capped",
        "provider": provider_norm,
        "provider_before": len(target),
        "provider_after": len(selected),
        "share_before": round(before_share, 4),
        "share_after": round(after_share, 4),
        "max_share_target": round(max_share, 4),
        "min_keep": min_keep,
    }
    return merged, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Build multi-source events backfill JSONL for 2025-now")
    ap.add_argument("--start", default="2025-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--day-step", type=int, default=7)
    ap.add_argument("--queries", default="||".join(DEFAULT_SEARCH_QUERIES), help="split by ||")
    ap.add_argument("--gdelt-max-records", type=int, default=50)
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--disable-google", action="store_true")
    ap.add_argument("--disable-gdelt", action="store_true")
    ap.add_argument("--disable-official-rss", action="store_true")
    ap.add_argument("--disable-source-balance", action="store_true")
    ap.add_argument("--max-google-share", type=float, default=0.68)
    ap.add_argument("--min-google-events", type=int, default=2500)
    ap.add_argument("--out-jsonl", default="artifacts/server_bundle/events_multisource_2025_now.jsonl")
    args = ap.parse_args()

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")
    queries = [q.strip() for q in str(args.queries).split("||") if q.strip()] or list(DEFAULT_SEARCH_QUERIES)

    sess = requests.Session()
    all_events: List[Dict[str, object]] = []
    source_stats: Dict[str, int] = {}

    if not bool(args.disable_google):
        ev = _collect_google_news(
            sess=sess,
            start=start_dt,
            end=end_dt,
            queries=queries,
            day_step=int(args.day_step),
            sleep_sec=float(args.sleep_sec),
        )
        all_events.extend(ev)
        source_stats["google_news"] = len(ev)

    if not bool(args.disable_gdelt):
        ev = _collect_gdelt(
            sess=sess,
            start=start_dt,
            end=end_dt,
            queries=queries,
            day_step=int(args.day_step),
            max_records=int(args.gdelt_max_records),
            sleep_sec=float(args.sleep_sec),
        )
        all_events.extend(ev)
        source_stats["gdelt"] = len(ev)

    if not bool(args.disable_official_rss):
        ev = _collect_official_rss(
            sess=sess,
            start=start_dt,
            end=end_dt,
            feeds=DEFAULT_OFFICIAL_FEEDS,
            sleep_sec=float(args.sleep_sec),
        )
        all_events.extend(ev)
        source_stats["official_rss"] = len(ev)

    deduped = _dedup(all_events)
    deduped.sort(key=lambda x: str(x.get("occurred_at") or ""))
    deduped_count = len(deduped)
    provider_counts_before = _provider_counts(deduped)
    source_balance_meta: Dict[str, object] = {"status": "disabled_or_not_needed"}
    if (not bool(args.disable_source_balance)) and (not bool(args.disable_google)):
        deduped, source_balance_meta = _cap_provider_share(
            deduped,
            provider="google_news_rss",
            max_share=float(args.max_google_share),
            min_keep=int(args.min_google_events),
        )
    provider_counts_after = _provider_counts(deduped)

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
                "source_stats": source_stats,
                "events_raw": len(all_events),
                "events_deduped": deduped_count,
                "events_final": len(deduped),
                "provider_counts_before_balance": provider_counts_before,
                "provider_counts_after_balance": provider_counts_after,
                "source_balance": source_balance_meta,
                "out_jsonl": out_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
