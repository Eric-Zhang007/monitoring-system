#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import requests


GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERIES = [
    "federal reserve AND rate",
    "美联储 AND 利率",
    "bitcoin ETF",
    "比特币 ETF",
    "crypto regulation",
    "加密货币 监管",
    "donald trump crypto",
    "特朗普 加密货币",
]
DEFAULT_RELEASE_IDS = ["10", "53", "54"]


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


def _fmt_gdelt_dt(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _parse_gdelt_ts(raw: str) -> datetime:
    s = str(raw or "").strip()
    if "T" in s and len(s) >= 16:
        try:
            return datetime.strptime(s[:16], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if len(s) >= 14 and s[:14].isdigit():
        return datetime.strptime(s[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _to_iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _stable_hash_01(text: str) -> float:
    raw = str(text or "").encode("utf-8", errors="ignore")
    digest = hashlib.sha1(raw).hexdigest()[:12]
    base = float((16 ** 12) - 1)
    return float(int(digest, 16) / base) if base > 0 else 0.0


def _provider_latency_minutes(provider: str, title: str, link: str, occurred_at: datetime, source_tier: int) -> float:
    p = str(provider or "").strip().lower()
    if p == "gdelt":
        low, high = 18.0, 70.0
    elif p == "fred":
        low, high = 1.0, 8.0
    else:
        low, high = 5.0, 35.0
    jitter = _stable_hash_01(f"{p}|{title}|{link}|{_to_iso_z(occurred_at)}")
    tier_penalty = max(0, int(source_tier) - 1) * 3.0
    return max(0.5, low + (high - low) * jitter + tier_penalty)


def _iter_day_windows(start: datetime, end: datetime, day_step: int) -> Iterable[Tuple[datetime, datetime]]:
    cur = start
    step = max(1, int(day_step))
    while cur < end:
        nxt = min(end, cur + timedelta(days=step))
        yield cur, nxt
        cur = nxt


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _language_hint(text: str) -> str:
    body = str(text or "")
    if not body.strip():
        return "other"
    if re.search(r"[\u4e00-\u9fff]", body):
        return "zh"
    if re.search(r"[A-Za-z]", body):
        return "en"
    return "other"


def _classify_event(title: str, query: str) -> Tuple[str, str, bool]:
    t = f"{title} {query}".lower()
    if any(k in t for k in ("sec", "regulation", "law", "ban", "policy", "tariff", "监管", "禁令", "关税")):
        event_type = "regulatory"
    else:
        event_type = "market"
    is_macro = any(
        k in t
        for k in (
            "federal reserve",
            "fomc",
            "interest rate",
            "inflation",
            "cpi",
            "unemployment",
            "treasury",
            "trump",
            "etf",
            "institutional",
            "blackrock",
            "fidelity",
            "美联储",
            "通胀",
            "利率",
            "特朗普",
            "贝莱德",
        )
    )
    market_scope = "macro" if is_macro else "crypto"
    global_impact = bool(is_macro)
    return event_type, market_scope, global_impact


def _extract_entities(title: str) -> List[Dict[str, object]]:
    mapping = {
        "BTC": ("bitcoin", "btc", "比特币"),
        "ETH": ("ethereum", "eth", "以太坊"),
        "SOL": ("solana", "sol", "索拉纳"),
        "BNB": ("bnb", "binance coin", "binancecoin"),
        "XRP": ("xrp", "ripple"),
        "ADA": ("ada", "cardano", "艾达"),
        "DOGE": ("doge", "dogecoin", "狗狗币"),
        "TRX": ("trx", "tron"),
        "AVAX": ("avax", "avalanche"),
        "LINK": ("link", "chainlink"),
    }
    t = str(title or "").lower()
    out: List[Dict[str, object]] = []
    for symbol, keywords in mapping.items():
        if any(k in t for k in keywords):
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


def _title_is_relevant(title: str) -> bool:
    t = str(title or "").lower()
    keywords = (
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
        "比特币",
        "以太坊",
        "索拉纳",
        "加密货币",
        "美联储",
        "通胀",
        "利率",
        "监管",
    )
    return any(k in t for k in keywords)


def _build_event_from_gdelt(row: Dict[str, object], query: str) -> Dict[str, object]:
    title = str(row.get("title") or "").strip() or "GDELT event"
    occurred_at = _parse_gdelt_ts(str(row.get("seendate") or ""))
    published_at = occurred_at
    source_tier = 2 if str(row.get("domain") or "").strip() else 3
    lag_minutes = _provider_latency_minutes(
        provider="gdelt",
        title=title,
        link=str(row.get("url") or ""),
        occurred_at=occurred_at,
        source_tier=source_tier,
    )
    available_at = occurred_at + timedelta(minutes=lag_minutes)
    source_latency_ms = int(max(0.0, lag_minutes * 60_000.0))
    event_type, market_scope, global_impact = _classify_event(title=title, query=query)
    confidence = 0.7 if market_scope == "macro" else 0.6
    return {
        "event_type": event_type,
        "title": title[:500],
        "occurred_at": _to_iso_z(occurred_at),
        "published_at": _to_iso_z(published_at),
        "available_at": _to_iso_z(available_at),
        "effective_at": _to_iso_z(available_at),
        "source_url": str(row.get("url") or ""),
        "source_name": str(row.get("domain") or "gdelt"),
        "source_timezone": "UTC",
        "source_tier": source_tier,
        "confidence_score": confidence,
        "event_importance": 0.75 if global_impact else 0.6,
        "novelty_score": 0.6,
        "entity_confidence": 0.5,
        "latency_ms": source_latency_ms,
        "source_latency_ms": source_latency_ms,
        "market_scope": market_scope,
        "payload": {
            "provider": "gdelt",
            "query": query,
            "language": row.get("language"),
            "language_hint": _language_hint(f"{title}\n{query}"),
            "tone": row.get("tone"),
            "global_impact": global_impact,
            "availability_lag_minutes": round(float(lag_minutes), 3),
            "availability_model": "provider_tier_hash_v1",
        },
        "entities": _extract_entities(title),
    }


def _fetch_gdelt_rows(
    sess: requests.Session,
    query: str,
    start: datetime,
    end: datetime,
    max_records: int,
    sleep_sec: float,
) -> List[Dict[str, object]]:
    params = {
        "query": query,
        "mode": "ArtList",
        "maxrecords": str(max(10, int(max_records))),
        "format": "json",
        "sort": "datedesc",
        "startdatetime": _fmt_gdelt_dt(start),
        "enddatetime": _fmt_gdelt_dt(end),
    }
    r = sess.get(GDELT_URL, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(max(1.0, float(sleep_sec) * 2.0))
        return []
    r.raise_for_status()
    try:
        body = r.json()
    except Exception:
        return []
    rows = body.get("articles") or []
    if sleep_sec > 0:
        time.sleep(float(sleep_sec))
    return [dict(x) for x in rows if isinstance(x, dict)]


def _fetch_fred_events(start: datetime, end: datetime, release_ids: List[str]) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for rid in release_ids:
        url = f"https://fred.stlouisfed.org/releases?rid={rid}&output=1"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            root = ET.fromstring(r.text)
        except Exception:
            continue
        for item in root.findall(".//item"):
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            published = (item.findtext("pubDate") or "").strip()
            if not published:
                continue
            try:
                ts = parsedate_to_datetime(str(published)).astimezone(timezone.utc)
            except Exception:
                continue
            if ts < start or ts > end:
                continue
            lag_minutes = _provider_latency_minutes(
                provider="fred",
                title=title,
                link=link,
                occurred_at=ts,
                source_tier=1,
            )
            available_at = ts + timedelta(minutes=lag_minutes)
            source_latency_ms = int(max(0.0, lag_minutes * 60_000.0))
            events.append(
                {
                    "event_type": "market",
                    "title": f"Macro release: {title}"[:500],
                    "occurred_at": _to_iso_z(ts),
                    "published_at": _to_iso_z(ts),
                    "available_at": _to_iso_z(available_at),
                    "effective_at": _to_iso_z(available_at),
                    "source_url": link,
                    "source_name": "fred",
                    "source_timezone": "UTC",
                    "source_tier": 1,
                    "confidence_score": 0.9,
                    "event_importance": 0.85,
                    "novelty_score": 0.6,
                    "entity_confidence": 0.6,
                    "latency_ms": source_latency_ms,
                    "source_latency_ms": source_latency_ms,
                    "market_scope": "macro",
                    "payload": {
                        "provider": "fred",
                        "release_id": rid,
                        "language": "en",
                        "summary": (item.findtext("description") or "")[:1200],
                        "global_impact": True,
                        "availability_lag_minutes": round(float(lag_minutes), 3),
                        "availability_model": "provider_tier_hash_v1",
                    },
                    "entities": [
                        {
                            "entity_type": "asset",
                            "name": "US_MACRO",
                            "symbol": "US_MACRO",
                            "country": "US",
                            "sector": "macro",
                            "metadata": {"release_id": rid},
                        }
                    ],
                }
            )
    return events


def _dedup_events(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen: Set[str] = set()
    out: List[Dict[str, object]] = []
    for ev in events:
        key = "|".join(
            [
                str(ev.get("source_name") or "").strip().lower(),
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
    ap = argparse.ArgumentParser(description="Build macro/news backfill event JSONL for 2025-now")
    ap.add_argument("--start", default="2025-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--day-step", type=int, default=3)
    ap.add_argument("--max-records", type=int, default=75)
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--queries", default="||".join(DEFAULT_QUERIES), help="split by ||")
    ap.add_argument("--skip-gdelt", action="store_true")
    ap.add_argument("--include-fred", action="store_true")
    ap.add_argument("--release-ids", default="10,53,54", help="FRED release ids, comma-separated")
    ap.add_argument("--llm-enrich", action="store_true", default=_env_flag("LLM_ENRICHMENT_ENABLED", default=False))
    ap.add_argument("--llm-max-events", type=int, default=int(os.getenv("LLM_ENRICH_MAX_EVENTS", "0")))
    ap.add_argument("--out-jsonl", default="artifacts/server_bundle/events_2025_now.jsonl")
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
    fetched_windows = 0
    if not bool(args.skip_gdelt):
        for w_start, w_end in _iter_day_windows(start_dt, end_dt, day_step=int(args.day_step)):
            for q in queries:
                rows = _fetch_gdelt_rows(
                    sess,
                    query=q,
                    start=w_start,
                    end=w_end,
                    max_records=int(args.max_records),
                    sleep_sec=float(args.sleep_sec),
                )
                for row in rows:
                    if not _title_is_relevant(str(row.get("title") or "")):
                        continue
                    all_events.append(_build_event_from_gdelt(row=row, query=q))
            fetched_windows += 1

    if bool(args.include_fred):
        rid = [x.strip() for x in str(args.release_ids).split(",") if x.strip()]
        all_events.extend(_fetch_fred_events(start=start_dt, end=end_dt, release_ids=rid or DEFAULT_RELEASE_IDS))

    deduped = _dedup_events(all_events)
    deduped.sort(key=lambda x: str(x.get("occurred_at") or ""))
    llm_meta = {"status": "disabled"}
    if bool(args.llm_enrich):
        helper_dir = os.path.dirname(__file__)
        if helper_dir not in sys.path:
            sys.path.insert(0, helper_dir)
        from event_enrichment import apply_llm_enrichment, build_llm_enricher  # type: ignore

        llm_meta = apply_llm_enrichment(
            deduped,
            enricher=build_llm_enricher(force_enable=True),
            max_events=int(args.llm_max_events),
        )
        llm_meta["status"] = "enabled"

    out_path = str(args.out_jsonl).strip()
    if not out_path:
        raise RuntimeError("out_jsonl_required")

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
                "windows": fetched_windows,
                "events_raw": len(all_events),
                "events_deduped": len(deduped),
                "llm_enrichment": llm_meta,
                "out_jsonl": out_path,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
