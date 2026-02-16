#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Set, Tuple

import requests


DEFAULT_QUERIES = [
    "bitcoin OR ethereum OR solana OR crypto market",
    "比特币 OR 以太坊 OR 索拉纳 OR 加密货币 市场",
    "federal reserve OR FOMC OR inflation OR CPI OR interest rate crypto",
    "美联储 OR 通胀 OR CPI OR 利率 OR 加密货币",
    "donald trump crypto OR us election crypto policy",
    "特朗普 加密货币 政策 OR 美国大选 加密货币",
    "SEC crypto OR ETF approval bitcoin",
    "SEC 加密 监管 OR 比特币 ETF 批准",
    "blackrock bitcoin etf inflow",
    "贝莱德 比特币 ETF 资金流入",
]
DEFAULT_GOOGLE_LOCALES = ["US:en", "CN:zh-Hans"]


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


def _parse_google_locale(raw: str) -> Tuple[str, str, str]:
    text = str(raw or "").strip()
    if not text:
        return "en-US", "US", "US:en"
    if ":" in text:
        gl, lang = text.split(":", 1)
        gl = gl.strip().upper() or "US"
        lang = lang.strip() or "en"
    else:
        gl, lang = "US", text
    lang_lower = lang.lower()
    if lang_lower == "en":
        hl = "en-US"
        ceid_lang = "en"
    elif lang_lower.startswith("zh"):
        hl = "zh-CN"
        ceid_lang = "zh-Hans"
    else:
        hl = lang
        ceid_lang = lang
    return hl, gl, f"{gl}:{ceid_lang}"


def _build_google_rss_url(query: str, start: datetime, end: datetime, locale: str) -> str:
    hl, gl, ceid = _parse_google_locale(locale)
    q = f"{query} after:{start.date().isoformat()} before:{end.date().isoformat()}"
    encoded = urllib.parse.quote(q)
    return (
        "https://news.google.com/rss/search?q="
        f"{encoded}&hl={urllib.parse.quote(hl)}&gl={urllib.parse.quote(gl)}&ceid={urllib.parse.quote(ceid)}"
    )


def _extract_entities(text: str) -> List[Dict[str, object]]:
    t = str(text or "").lower()
    mapping = {
        "BTC": ("bitcoin", "btc", "比特币"),
        "ETH": ("ethereum", "eth", "以太坊"),
        "SOL": ("solana", "sol", "索拉纳"),
        "BNB": ("bnb", "binance"),
        "XRP": ("xrp", "ripple"),
        "ADA": ("ada", "cardano", "艾达"),
        "DOGE": ("doge", "dogecoin", "狗狗币"),
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
    if any(k in t for k in ("sec", "regulation", "policy", "ban", "law", "监管", "禁令", "执法")):
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
            "美联储",
            "通胀",
            "利率",
            "大选",
            "特朗普",
            "贝莱德",
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
    ap.add_argument("--google-locales", default=",".join(DEFAULT_GOOGLE_LOCALES), help="e.g. US:en,CN:zh-Hans")
    ap.add_argument("--sleep-sec", type=float, default=0.2)
    ap.add_argument("--llm-enrich", action="store_true", default=_env_flag("LLM_ENRICHMENT_ENABLED", default=False))
    ap.add_argument("--llm-max-events", type=int, default=int(os.getenv("LLM_ENRICH_MAX_EVENTS", "0")))
    ap.add_argument("--out-jsonl", default="artifacts/server_bundle/events_google_news_2025_now.jsonl")
    args = ap.parse_args()

    start_dt = _parse_dt_utc(args.start)
    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")

    queries = [q.strip() for q in str(args.queries).split("||") if q.strip()]
    if not queries:
        queries = list(DEFAULT_QUERIES)
    locales = [x.strip() for x in str(args.google_locales).split(",") if x.strip()] or list(DEFAULT_GOOGLE_LOCALES)

    sess = requests.Session()
    all_events: List[Dict[str, object]] = []
    windows = 0
    reqs = 0
    for w_start, w_end in _iter_windows(start_dt, end_dt, day_step=int(args.day_step)):
        for q in queries:
            for locale in locales:
                url = _build_google_rss_url(q, w_start, w_end, locale=locale)
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
                                "google_locale": locale,
                                "language": _language_hint(f"{title}\n{str(it.get('description') or '')}"),
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
                "google_locales": locales,
                "windows": windows,
                "requests": reqs,
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
