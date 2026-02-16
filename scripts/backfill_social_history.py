#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _to_bool(raw: str, default: bool = False) -> bool:
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


def _normalize_symbols(raw: Iterable[object]) -> List[str]:
    seen = set()
    out: List[str] = []
    for sym in raw:
        token = str(sym or "").strip().upper().replace("$", "")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _ensure_social_payload(payload: Dict[str, object]) -> Dict[str, object]:
    defaults = {
        "social_platform": "none",
        "author": "",
        "author_followers": 0,
        "engagement_score": 0.0,
        "comment_sentiment": 0.0,
        "post_sentiment": 0.0,
        "n_comments": 0,
        "n_replies": 0,
        "is_verified": False,
        "influence_tier": "unknown",
        "symbol_mentions": [],
    }
    out = dict(payload or {})
    for key, value in defaults.items():
        out.setdefault(key, list(value) if isinstance(value, list) else value)
    out["symbol_mentions"] = _normalize_symbols(out.get("symbol_mentions") or [])
    return out


def _dedup(events: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for ev in events:
        key = "|".join(
            [
                str(ev.get("title") or "").strip().lower(),
                str(ev.get("source_url") or "").strip().lower(),
                str(ev.get("occurred_at") or ""),
            ]
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(ev)
    return out


def _build_connectors(args, liquid_symbols: Sequence[str]):
    repo_root = Path(__file__).resolve().parents[1]
    collector_dir = repo_root / "collector"
    if str(collector_dir) not in sys.path:
        sys.path.insert(0, str(collector_dir))

    from connectors import RedditConnector, TelegramConnector, XTwitterConnector, YouTubeConnector  # type: ignore

    selected = {s.strip().lower() for s in str(args.sources).split(",") if s.strip()}
    if not selected:
        selected = {"twitter", "reddit", "youtube", "telegram"}

    out = []
    if "twitter" in selected or "x" in selected:
        query = str(args.twitter_query).strip()
        if not args.twitter_include_replies:
            query = f"{query} -is:reply"
        out.append(
            XTwitterConnector(
                bearer_token=str(args.twitter_bearer_token).strip(),
                query=query,
                max_results=int(args.twitter_max_results),
                known_symbols=list(liquid_symbols),
            )
        )
    if "reddit" in selected:
        out.append(
            RedditConnector(
                subreddits=[s.strip() for s in str(args.reddit_subreddits).split(",") if s.strip()],
                limit=int(args.reddit_limit),
                sort=str(args.reddit_sort),
                user_agent=str(args.reddit_user_agent),
                mode=str(args.reddit_mode),
                client_id=str(args.reddit_client_id),
                client_secret=str(args.reddit_client_secret),
                fetch_comments=True,
                comments_per_post=20,
                max_posts_for_comments=max(20, int(args.reddit_limit)),
            )
        )
    if "youtube" in selected:
        out.append(
            YouTubeConnector(
                mode=str(args.youtube_mode),
                channel_ids=[s.strip() for s in str(args.youtube_channel_ids).split(",") if s.strip()],
                query=str(args.youtube_query),
                api_key=str(args.youtube_api_key),
                max_results=int(args.youtube_max_results),
            )
        )
    if "telegram" in selected:
        out.append(
            TelegramConnector(
                feeds=[s.strip() for s in str(args.telegram_feeds).split(",") if s.strip()],
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill social-source events into canonical JSONL")
    ap.add_argument("--sources", default=os.getenv("SOCIAL_CONNECTORS", "twitter,reddit,youtube,telegram"))
    ap.add_argument("--out-jsonl", default="artifacts/social_history_backfill.jsonl")

    ap.add_argument("--twitter-bearer-token", default=os.getenv("TWITTER_BEARER_TOKEN", ""))
    ap.add_argument(
        "--twitter-query",
        default=os.getenv(
            "TWITTER_QUERY",
            "(bitcoin OR ethereum OR solana OR 比特币 OR 以太坊 OR 索拉纳 OR $BTC OR $ETH OR $SOL) -is:retweet",
        ),
    )
    ap.add_argument("--twitter-max-results", type=int, default=int(os.getenv("TWITTER_MAX_RESULTS", "25")))
    ap.add_argument("--twitter-include-replies", action="store_true", default=_to_bool(os.getenv("TWITTER_INCLUDE_REPLIES", "0")))

    ap.add_argument("--reddit-subreddits", default=os.getenv("REDDIT_SUBREDDITS", "cryptocurrency,bitcoin,ethereum,solana"))
    ap.add_argument("--reddit-limit", type=int, default=int(os.getenv("REDDIT_LIMIT", "25")))
    ap.add_argument("--reddit-sort", default=os.getenv("REDDIT_SORT", "new"))
    ap.add_argument("--reddit-user-agent", default=os.getenv("REDDIT_USER_AGENT", "monitoring-system/2.0"))
    ap.add_argument("--reddit-mode", default=os.getenv("REDDIT_MODE", "public"))
    ap.add_argument("--reddit-client-id", default=os.getenv("REDDIT_CLIENT_ID", ""))
    ap.add_argument("--reddit-client-secret", default=os.getenv("REDDIT_CLIENT_SECRET", ""))

    ap.add_argument("--youtube-mode", default=os.getenv("YOUTUBE_MODE", "rss"))
    ap.add_argument("--youtube-channel-ids", default=os.getenv("YOUTUBE_CHANNEL_IDS", ""))
    ap.add_argument("--youtube-query", default=os.getenv("YOUTUBE_QUERY", "crypto bitcoin ethereum solana 比特币 以太坊 索拉纳"))
    ap.add_argument("--youtube-api-key", default=os.getenv("YOUTUBE_API_KEY", ""))
    ap.add_argument("--youtube-max-results", type=int, default=int(os.getenv("YOUTUBE_MAX_RESULTS", "20")))

    ap.add_argument("--telegram-feeds", default=os.getenv("TELEGRAM_FEEDS", ""))

    ap.add_argument("--liquid-symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK"))
    ap.add_argument("--dedup", action="store_true", default=_to_bool(os.getenv("SOCIAL_DEDUP", "1"), default=True))
    ap.add_argument("--no-dedup", action="store_false", dest="dedup")
    ap.add_argument("--llm-enrich", action="store_true", default=_to_bool(os.getenv("LLM_ENRICHMENT_ENABLED", "0")))
    ap.add_argument("--llm-max-events", type=int, default=int(os.getenv("LLM_ENRICH_MAX_EVENTS", "0")))
    args = ap.parse_args()

    liquid_symbols = [s.strip().upper() for s in str(args.liquid_symbols).split(",") if s.strip()]
    connectors = _build_connectors(args, liquid_symbols=liquid_symbols)

    events: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}

    for connector in connectors:
        try:
            raw_rows = connector.fetch()
        except Exception as exc:
            failures.append({"connector": connector.name, "stage": "fetch", "error": str(exc)[:240]})
            continue
        counts[f"{connector.name}_raw"] = len(raw_rows)
        for raw in raw_rows:
            try:
                event = connector.normalize(raw)
            except Exception as exc:
                failures.append({"connector": connector.name, "stage": "normalize", "error": str(exc)[:240]})
                continue
            payload = _ensure_social_payload(dict(event.get("payload") or {}))
            occurred_at = str(event.get("occurred_at") or datetime.utcnow().isoformat() + "Z")
            event["payload"] = payload
            event.setdefault("published_at", occurred_at)
            event.setdefault("available_at", occurred_at)
            event.setdefault("effective_at", occurred_at)
            events.append(event)

    if args.dedup:
        events = _dedup(events)

    llm_meta = {"status": "disabled"}
    if bool(args.llm_enrich):
        helper_dir = Path(__file__).resolve().parent
        if str(helper_dir) not in sys.path:
            sys.path.insert(0, str(helper_dir))
        from event_enrichment import apply_llm_enrichment, build_llm_enricher  # type: ignore

        llm_meta = apply_llm_enrichment(
            events,
            enricher=build_llm_enricher(force_enable=True),
            max_events=int(args.llm_max_events),
        )
        llm_meta["status"] = "enabled"

    out_jsonl = str(args.out_jsonl).strip()
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "status": "ok",
                "connectors": [c.name for c in connectors],
                "counts": counts,
                "events": len(events),
                "failures": failures[:20],
                "llm_enrichment": llm_meta,
                "out_jsonl": out_jsonl,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
