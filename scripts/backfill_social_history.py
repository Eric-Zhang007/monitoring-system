#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


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


def _detect_language(text: str) -> str:
    body = str(text or "").strip()
    if not body:
        return "other"
    if re.search(r"[\u4e00-\u9fff]", body):
        return "zh"
    if re.search(r"[A-Za-z]", body):
        return "en"
    return "other"


def _parse_dt_utc(raw: object) -> datetime:
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            raw = raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_datetime")
    norm = text.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt_obj = datetime.fromisoformat(norm)
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)
    return dt_obj.astimezone(timezone.utc)


def _to_iso_z(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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


def _normalize_time_alignment(event: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    occurred_at = _parse_dt_utc(event.get("occurred_at") or datetime.now(timezone.utc))
    published_at = _parse_dt_utc(event.get("published_at") or occurred_at)
    if published_at < occurred_at:
        published_at = occurred_at
    available_at = _parse_dt_utc(event.get("available_at") or published_at)
    if available_at < published_at:
        available_at = published_at
    effective_at = _parse_dt_utc(event.get("effective_at") or available_at)
    if effective_at < available_at:
        effective_at = available_at
    event["occurred_at"] = _to_iso_z(occurred_at)
    event["published_at"] = _to_iso_z(published_at)
    event["available_at"] = _to_iso_z(available_at)
    event["effective_at"] = _to_iso_z(effective_at)
    source_latency_ms = int(max(0.0, (available_at - occurred_at).total_seconds() * 1000.0))
    event["source_latency_ms"] = int(max(int(event.get("source_latency_ms") or 0), source_latency_ms))
    event["latency_ms"] = int(max(int(event.get("latency_ms") or 0), source_latency_ms))
    monotonic = bool(occurred_at <= published_at <= available_at <= effective_at)
    return event, monotonic


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
    ap.add_argument("--start", default="", help="inclusive UTC start, e.g. 2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="", help="inclusive UTC end, default now")
    ap.add_argument("--language-targets", default=os.getenv("SOCIAL_LANGUAGE_TARGETS", "en,zh"))
    ap.add_argument("--pipeline-tag", default=os.getenv("SOCIAL_PIPELINE_TAG", "task_h_social_2018_now"))
    ap.add_argument("--provenance-run-id", default=os.getenv("SOCIAL_PROVENANCE_RUN_ID", ""))
    ap.add_argument("--max-fetch-retries", type=int, default=max(1, int(os.getenv("SOCIAL_FETCH_MAX_RETRIES", "3"))))
    ap.add_argument("--retry-backoff-sec", type=float, default=float(os.getenv("SOCIAL_FETCH_RETRY_BACKOFF_SEC", "1.0")))
    ap.add_argument("--dry-run", action="store_true", help="plan connector run only; no fetch")

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

    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    start_dt = _parse_dt_utc(args.start) if str(args.start).strip() else datetime(2018, 1, 1, tzinfo=timezone.utc)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range_end_must_be_gt_start")
    language_targets = {x.strip().lower() for x in str(args.language_targets).split(",") if x.strip()}
    if not language_targets:
        language_targets = {"en", "zh"}
    selected_sources = [s.strip().lower() for s in str(args.sources).split(",") if s.strip()]

    liquid_symbols = [s.strip().upper() for s in str(args.liquid_symbols).split(",") if s.strip()]
    connectors = _build_connectors(args, liquid_symbols=liquid_symbols)
    provenance_run_id = str(args.provenance_run_id).strip() or f"social-backfill-{int(time.time())}"

    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "pipeline_tag": str(args.pipeline_tag),
                    "provenance_run_id": provenance_run_id,
                    "start": _to_iso_z(start_dt),
                    "end": _to_iso_z(end_dt),
                    "sources": selected_sources,
                    "connectors": [c.name for c in connectors],
                    "language_targets": sorted(language_targets),
                    "llm_enrich": bool(args.llm_enrich),
                },
                ensure_ascii=False,
            )
        )
        return 0

    events: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    counts: Dict[str, int] = {}
    alignment_violations = 0
    language_counts: Dict[str, int] = {}
    dropped_outside_window = 0
    dropped_language = 0

    for connector in connectors:
        raw_rows: List[Dict[str, Any]] = []
        last_err: Exception | None = None
        for attempt in range(1, max(1, int(args.max_fetch_retries)) + 1):
            try:
                raw_rows = connector.fetch()
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt >= max(1, int(args.max_fetch_retries)):
                    break
                sleep_s = min(90.0, float(args.retry_backoff_sec) * (2 ** (attempt - 1)))
                time.sleep(max(0.1, sleep_s))
        if last_err is not None:
            failures.append({"connector": connector.name, "stage": "fetch", "error": str(last_err)[:240]})
            continue

        counts[f"{connector.name}_raw"] = len(raw_rows)
        for raw in raw_rows:
            try:
                event = connector.normalize(raw)
            except Exception as exc:
                failures.append({"connector": connector.name, "stage": "normalize", "error": str(exc)[:240]})
                continue
            payload = _ensure_social_payload(dict(event.get("payload") or {}))
            base_text = f"{event.get('title') or ''}\n{payload.get('summary') or ''}"
            detected_language = str(payload.get("language") or "").strip().lower() or _detect_language(base_text)
            payload["language"] = detected_language
            if detected_language not in language_targets:
                dropped_language += 1
                continue
            language_counts[detected_language] = int(language_counts.get(detected_language, 0)) + 1
            event["payload"] = payload
            event.setdefault("published_at", event.get("occurred_at"))
            event.setdefault("available_at", event.get("published_at"))
            event.setdefault("effective_at", event.get("available_at"))
            try:
                event, monotonic = _normalize_time_alignment(event)
            except Exception as exc:
                failures.append({"connector": connector.name, "stage": "time_normalize", "error": str(exc)[:240]})
                continue
            if not monotonic:
                alignment_violations += 1
            occurred_dt = _parse_dt_utc(event.get("occurred_at"))
            if occurred_dt < start_dt or occurred_dt > end_dt:
                dropped_outside_window += 1
                continue
            provenance = {
                "pipeline_tag": str(args.pipeline_tag),
                "provenance_run_id": provenance_run_id,
                "collector_connector": connector.name,
                "window_start": _to_iso_z(start_dt),
                "window_end": _to_iso_z(end_dt),
                "language_targets": sorted(language_targets),
                "source_script": "scripts/backfill_social_history.py",
                "generated_at": _to_iso_z(datetime.now(timezone.utc)),
            }
            time_alignment_meta = {
                "alignment_mode": "strict_asof_v1",
                "occurred_at": str(event.get("occurred_at")),
                "published_at": str(event.get("published_at")),
                "available_at": str(event.get("available_at")),
                "effective_at": str(event.get("effective_at")),
                "monotonic_non_decreasing": bool(monotonic),
            }
            event_payload = dict(event.get("payload") or {})
            event_payload["provenance"] = provenance
            event_payload["time_alignment"] = time_alignment_meta
            event["payload"] = event_payload
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
                "start": _to_iso_z(start_dt),
                "end": _to_iso_z(end_dt),
                "language_targets": sorted(language_targets),
                "language_counts": language_counts,
                "dropped_outside_window": dropped_outside_window,
                "dropped_language": dropped_language,
                "alignment_violations": alignment_violations,
                "pipeline_tag": str(args.pipeline_tag),
                "provenance_run_id": provenance_run_id,
                "llm_enrichment": llm_meta,
                "out_jsonl": out_jsonl,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
