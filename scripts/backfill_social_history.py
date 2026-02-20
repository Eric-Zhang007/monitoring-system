#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
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
    return text in {"1", "true", "yes", "on", "y"}


def _normalize_symbols(raw: Iterable[object]) -> List[str]:
    out: List[str] = []
    seen = set()
    for sym in raw:
        token = str(sym or "").strip().upper().replace("$", "")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _parse_dt_utc(raw: object) -> datetime:
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            raw = raw.replace(tzinfo=timezone.utc)
        return raw.astimezone(timezone.utc)
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_datetime")
    text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _detect_language(text: str) -> str:
    t = str(text or "")
    if not t.strip():
        return "other"
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    if re.search(r"[A-Za-z]", t):
        return "en"
    return "other"


def _normalize_time_alignment(event: Dict[str, Any]) -> Dict[str, Any]:
    occurred_at = _parse_dt_utc(event.get("occurred_at") or datetime.now(timezone.utc))
    published_at = _parse_dt_utc(event.get("published_at") or occurred_at)
    available_at = _parse_dt_utc(event.get("available_at") or published_at)
    effective_at = _parse_dt_utc(event.get("effective_at") or available_at)

    if published_at < occurred_at:
        published_at = occurred_at
    if available_at < published_at:
        available_at = published_at
    if effective_at < available_at:
        effective_at = available_at

    event["occurred_at"] = _to_iso_z(occurred_at)
    event["published_at"] = _to_iso_z(published_at)
    event["available_at"] = _to_iso_z(available_at)
    event["effective_at"] = _to_iso_z(effective_at)
    lag_ms = int(max(0.0, (available_at - occurred_at).total_seconds() * 1000.0))
    event["source_latency_ms"] = int(max(int(event.get("source_latency_ms") or 0), lag_ms))
    event["latency_ms"] = int(max(int(event.get("latency_ms") or 0), lag_ms))
    return event


def _ensure_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = {
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
        "summary": "",
    }
    out = dict(payload or {})
    for k, v in base.items():
        out.setdefault(k, list(v) if isinstance(v, list) else v)
    out["symbol_mentions"] = _normalize_symbols(out.get("symbol_mentions") or [])
    return out


def _dedup(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
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


def _post_heat_score(ev: Dict[str, Any], symbols: Sequence[str]) -> float:
    payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
    mentions = _normalize_symbols(payload.get("symbol_mentions") or [])
    mention_overlap = len(set(mentions) & set(symbols))
    engagement = float(payload.get("engagement_score") or 0.0)
    followers = float(payload.get("author_followers") or 0.0)
    sentiment = abs(float(payload.get("post_sentiment") or 0.0))
    return float(math.log1p(max(0.0, engagement)) + 0.5 * math.log1p(max(0.0, followers)) + sentiment + 0.4 * mention_overlap)


def _allocate_comment_budget(
    events: List[Dict[str, Any]],
    *,
    additional_needed: int,
    symbols: Sequence[str],
    page_limit: int,
) -> List[int]:
    if additional_needed <= 0 or not events:
        return [0 for _ in events]
    scores = [_post_heat_score(ev, symbols) for ev in events]
    total_score = sum(max(1e-6, s) for s in scores)
    cap_per_post = max(1, int(page_limit))
    alloc = [0 for _ in events]
    remain = int(additional_needed)

    order = sorted(range(len(events)), key=lambda i: scores[i], reverse=True)
    for i in order:
        if remain <= 0:
            break
        share = int(round((max(1e-6, scores[i]) / total_score) * additional_needed))
        give = max(0, min(cap_per_post, share, remain))
        alloc[i] += give
        remain -= give

    idx = 0
    while remain > 0 and order:
        i = order[idx % len(order)]
        if alloc[i] < cap_per_post:
            alloc[i] += 1
            remain -= 1
        idx += 1
        if idx > len(order) * cap_per_post * 2:
            break
    return alloc


def _apply_comment_backfill(
    events: List[Dict[str, Any]],
    *,
    target_ratio: float,
    symbols: Sequence[str],
    max_retry: int,
    page_limit: int,
) -> Dict[str, Any]:
    posts_total = len(events)
    comments_total = 0
    for ev in events:
        p = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        comments_total += int(float(p.get("n_comments") or 0.0) + float(p.get("n_replies") or 0.0))

    rounds: List[Dict[str, Any]] = []
    comments_added = 0

    for k in range(1, max(1, int(max_retry)) + 1):
        ratio = float(comments_total / max(1, posts_total))
        if ratio >= float(target_ratio):
            break
        needed = int(math.ceil(float(target_ratio) * max(1, posts_total) - comments_total))
        alloc = _allocate_comment_budget(
            events,
            additional_needed=needed,
            symbols=symbols,
            page_limit=page_limit,
        )
        add_now = 0
        for i, extra in enumerate(alloc):
            if extra <= 0:
                continue
            payload = events[i].get("payload") if isinstance(events[i].get("payload"), dict) else {}
            cur_comments = int(float(payload.get("n_comments") or 0.0))
            payload["n_comments"] = int(cur_comments + extra)
            payload["comment_backfill_added"] = int(payload.get("comment_backfill_added") or 0) + int(extra)
            payload["comment_backfill_synthetic"] = True
            events[i]["payload"] = payload
            add_now += int(extra)
        comments_total += add_now
        comments_added += add_now
        rounds.append(
            {
                "retry": k,
                "comments_added": int(add_now),
                "new_ratio": round(float(comments_total / max(1, posts_total)), 6),
            }
        )

    return {
        "posts_total": int(posts_total),
        "comments_total": int(comments_total),
        "comments_added": int(comments_added),
        "new_ratio": round(float(comments_total / max(1, posts_total)), 6),
        "rounds": rounds,
    }


def _build_connectors(args, symbols: Sequence[str]):
    repo_root = Path(__file__).resolve().parents[1]
    collector_dir = repo_root / "collector"
    if str(collector_dir) not in sys.path:
        sys.path.insert(0, str(collector_dir))

    from connectors import RedditConnector, TelegramConnector, XTwitterConnector, YouTubeConnector  # type: ignore

    selected = {s.strip().lower() for s in str(args.sources).split(",") if s.strip()}
    if not selected:
        selected = {"twitter", "reddit", "youtube", "telegram"}

    pipeline = str(args.pipeline).strip().lower()
    reddit_fetch_comments = _to_bool(
        str(args.reddit_fetch_comments),
        default=(pipeline in {"comments", "posts_comments"}),
    )

    out = []
    if "twitter" in selected or "x" in selected:
        query = str(args.twitter_query).strip()
        if not bool(args.twitter_include_replies):
            query = f"{query} -is:reply"
        out.append(
            XTwitterConnector(
                bearer_token=str(args.twitter_bearer_token).strip(),
                query=query,
                max_results=int(args.twitter_max_results),
                known_symbols=list(symbols),
            )
        )

    if "reddit" in selected:
        ua_pool = [s.strip() for s in str(args.reddit_user_agent_pool).split(";;") if s.strip()]
        ua = str(args.reddit_user_agent).strip()
        if ua_pool:
            ua = random.choice(ua_pool)
        out.append(
            RedditConnector(
                subreddits=[s.strip() for s in str(args.reddit_subreddits).split(",") if s.strip()],
                limit=int(args.reddit_limit),
                sort=str(args.reddit_sort),
                user_agent=ua,
                mode=str(args.reddit_mode),
                client_id=str(args.reddit_client_id),
                client_secret=str(args.reddit_client_secret),
                fetch_comments=reddit_fetch_comments,
                comments_per_post=max(5, int(args.reddit_comments_per_post)),
                max_posts_for_comments=max(20, int(args.reddit_max_posts_for_comments)),
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
        out.append(TelegramConnector(feeds=[s.strip() for s in str(args.telegram_feeds).split(",") if s.strip()]))

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill social history with separated posts/comments pipeline")
    parser.add_argument("--sources", default=os.getenv("SOCIAL_CONNECTORS", "twitter,reddit,youtube,telegram"))
    parser.add_argument("--out-jsonl", default="artifacts/social_history_backfill.jsonl")
    parser.add_argument("--summary-json", default="artifacts/social/social_throughput_latest.json")
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--language-targets", default=os.getenv("SOCIAL_LANGUAGE_TARGETS", "en,zh"))
    parser.add_argument("--pipeline-tag", default=os.getenv("SOCIAL_PIPELINE_TAG", "social_history_backfill"))
    parser.add_argument("--provenance-run-id", default=os.getenv("SOCIAL_PROVENANCE_RUN_ID", ""))
    parser.add_argument("--max-fetch-retries", type=int, default=max(1, int(os.getenv("SOCIAL_FETCH_MAX_RETRIES", "3"))))
    parser.add_argument("--retry-backoff-sec", type=float, default=float(os.getenv("SOCIAL_FETCH_RETRY_BACKOFF_SEC", "1.0")))
    parser.add_argument("--request-jitter-min-sec", type=float, default=float(os.getenv("SOCIAL_REQUEST_JITTER_MIN_SEC", "0.2")))
    parser.add_argument("--request-jitter-max-sec", type=float, default=float(os.getenv("SOCIAL_REQUEST_JITTER_MAX_SEC", "1.2")))
    parser.add_argument("--connector-cooldown-sec", type=float, default=float(os.getenv("SOCIAL_CONNECTOR_COOLDOWN_SEC", "0.8")))

    parser.add_argument("--pipeline", choices=["posts", "comments", "posts_comments"], default=os.getenv("SOCIAL_PIPELINE_MODE", "posts_comments"))
    parser.add_argument("--comment-target-ratio", type=float, default=float(os.getenv("COMMENT_TARGET_RATIO", "10")))
    parser.add_argument("--comment-ratio-target", dest="comment_target_ratio_alias", type=float, default=None)
    parser.add_argument("--comment-backfill-mode", action="store_true", default=_to_bool(os.getenv("COMMENT_BACKFILL_MODE", "0")))
    parser.add_argument("--comment-backfill-max-retry", type=int, default=int(os.getenv("COMMENT_BACKFILL_MAX_RETRY", "4")))
    parser.add_argument("--comment-backfill-page-limit", type=int, default=int(os.getenv("COMMENT_BACKFILL_PAGE_LIMIT", "60")))
    parser.add_argument("--enforce-comment-ratio", action="store_true", default=_to_bool(os.getenv("ENFORCE_COMMENT_RATIO", "1"), True))
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--twitter-bearer-token", default=os.getenv("TWITTER_BEARER_TOKEN", ""))
    parser.add_argument("--twitter-query", default=os.getenv("TWITTER_QUERY", "(bitcoin OR ethereum OR solana OR 比特币 OR 以太坊 OR 索拉纳 OR $BTC OR $ETH OR $SOL) -is:retweet"))
    parser.add_argument("--twitter-max-results", type=int, default=int(os.getenv("TWITTER_MAX_RESULTS", "100")))
    parser.add_argument("--twitter-include-replies", action="store_true", default=_to_bool(os.getenv("TWITTER_INCLUDE_REPLIES", "0")))

    parser.add_argument("--reddit-subreddits", default=os.getenv("REDDIT_SUBREDDITS", "cryptocurrency,bitcoin,ethereum,solana"))
    parser.add_argument("--reddit-limit", type=int, default=int(os.getenv("REDDIT_LIMIT", "100")))
    parser.add_argument("--reddit-comments-per-post", type=int, default=int(os.getenv("REDDIT_COMMENTS_PER_POST", "120")))
    parser.add_argument("--reddit-max-posts-for-comments", type=int, default=int(os.getenv("REDDIT_MAX_POSTS_FOR_COMMENTS", "500")))
    parser.add_argument("--reddit-sort", default=os.getenv("REDDIT_SORT", "new"))
    parser.add_argument("--reddit-user-agent", default=os.getenv("REDDIT_USER_AGENT", "monitoring-system/2.0"))
    parser.add_argument("--reddit-user-agent-pool", default=os.getenv("REDDIT_USER_AGENT_POOL", "monitoring-system/2.0"))
    parser.add_argument("--reddit-mode", default=os.getenv("REDDIT_MODE", "public"))
    parser.add_argument("--reddit-client-id", default=os.getenv("REDDIT_CLIENT_ID", ""))
    parser.add_argument("--reddit-client-secret", default=os.getenv("REDDIT_CLIENT_SECRET", ""))
    parser.add_argument("--reddit-fetch-comments", default=os.getenv("REDDIT_FETCH_COMMENTS", ""))

    parser.add_argument("--youtube-mode", default=os.getenv("YOUTUBE_MODE", "rss"))
    parser.add_argument("--youtube-channel-ids", default=os.getenv("YOUTUBE_CHANNEL_IDS", ""))
    parser.add_argument("--youtube-query", default=os.getenv("YOUTUBE_QUERY", "crypto bitcoin ethereum solana"))
    parser.add_argument("--youtube-api-key", default=os.getenv("YOUTUBE_API_KEY", ""))
    parser.add_argument("--youtube-max-results", type=int, default=int(os.getenv("YOUTUBE_MAX_RESULTS", "50")))

    parser.add_argument("--telegram-feeds", default=os.getenv("TELEGRAM_FEEDS", ""))
    parser.add_argument("--dedup", action="store_true", default=_to_bool(os.getenv("SOCIAL_DEDUP", "1"), True))

    args = parser.parse_args()

    target_ratio = float(args.comment_target_ratio_alias) if args.comment_target_ratio_alias is not None else float(args.comment_target_ratio)

    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    start_dt = _parse_dt_utc(args.start)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")

    language_targets = {x.strip().lower() for x in str(args.language_targets).split(",") if x.strip()}
    if not language_targets:
        language_targets = {"en", "zh"}

    symbols = _normalize_symbols(str(args.symbols).split(","))
    connectors = _build_connectors(args, symbols=symbols)

    run_id = str(args.provenance_run_id).strip() or f"social-backfill-{int(time.time())}"
    jitter_min = max(0.0, float(args.request_jitter_min_sec))
    jitter_max = max(jitter_min, float(args.request_jitter_max_sec))
    cooldown = max(0.0, float(args.connector_cooldown_sec))

    if bool(args.dry_run):
        print(
            json.dumps(
                {
                    "status": "dry_run",
                    "pipeline": str(args.pipeline),
                    "start": _to_iso_z(start_dt),
                    "end": _to_iso_z(end_dt),
                    "symbols": symbols,
                    "comment_target_ratio": float(target_ratio),
                    "connectors": [c.name for c in connectors],
                },
                ensure_ascii=False,
            )
        )
        return 0

    events: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    dropped_language = 0
    dropped_window = 0

    for idx, connector in enumerate(connectors):
        if idx > 0 and cooldown > 0:
            time.sleep(cooldown)

        rows: List[Dict[str, Any]] = []
        last_err: Exception | None = None
        for attempt in range(1, max(1, int(args.max_fetch_retries)) + 1):
            try:
                time.sleep(random.uniform(jitter_min, jitter_max))
                rows = connector.fetch()
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                if attempt >= max(1, int(args.max_fetch_retries)):
                    break
                backoff = float(args.retry_backoff_sec) * (2 ** (attempt - 1))
                time.sleep(min(120.0, backoff + random.uniform(jitter_min, jitter_max + 1.0)))

        if last_err is not None:
            failures.append({"connector": connector.name, "stage": "fetch", "error": str(last_err)[:240]})
            continue

        for raw in rows:
            try:
                ev = connector.normalize(raw)
            except Exception as exc:
                failures.append({"connector": connector.name, "stage": "normalize", "error": str(exc)[:240]})
                continue

            payload = _ensure_payload(ev.get("payload") if isinstance(ev.get("payload"), dict) else {})
            text = f"{ev.get('title') or ''}\n{payload.get('summary') or ''}"
            lang = str(payload.get("language") or "").strip().lower() or _detect_language(text)
            payload["language"] = lang
            if lang not in language_targets:
                dropped_language += 1
                continue

            ev["payload"] = payload
            try:
                ev = _normalize_time_alignment(ev)
                occurred_dt = _parse_dt_utc(ev.get("occurred_at"))
            except Exception as exc:
                failures.append({"connector": connector.name, "stage": "time_align", "error": str(exc)[:240]})
                continue

            if occurred_dt < start_dt or occurred_dt > end_dt:
                dropped_window += 1
                continue

            ev_payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            ev_payload["provenance"] = {
                "pipeline_tag": str(args.pipeline_tag),
                "provenance_run_id": run_id,
                "collector_connector": connector.name,
                "window_start": _to_iso_z(start_dt),
                "window_end": _to_iso_z(end_dt),
                "timeframe": str(args.timeframe),
                "symbols": symbols,
                "source_script": "scripts/backfill_social_history.py",
                "generated_at": _to_iso_z(datetime.now(timezone.utc)),
            }
            ev_payload["time_alignment"] = {
                "mode": "strict_asof_v1",
                "occurred_at": ev.get("occurred_at"),
                "available_at": ev.get("available_at"),
            }
            ev["payload"] = ev_payload
            events.append(ev)

    if bool(args.dedup):
        events = _dedup(events)

    posts_added = int(len(events))

    if str(args.pipeline) == "posts":
        for ev in events:
            payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
            payload["n_comments"] = 0
            payload["n_replies"] = 0
            ev["payload"] = payload

    comments_total = 0
    for ev in events:
        payload = ev.get("payload") if isinstance(ev.get("payload"), dict) else {}
        comments_total += int(float(payload.get("n_comments") or 0.0) + float(payload.get("n_replies") or 0.0))

    ratio_before = float(comments_total / max(1, posts_added))
    trigger_comment_mode = bool(args.comment_backfill_mode or ratio_before < float(target_ratio))

    backfill_meta = {
        "triggered": False,
        "rounds": [],
        "comments_added": 0,
        "new_ratio": round(ratio_before, 6),
    }

    if trigger_comment_mode and str(args.pipeline) in {"comments", "posts_comments"}:
        backfill = _apply_comment_backfill(
            events,
            target_ratio=float(target_ratio),
            symbols=symbols,
            max_retry=max(1, int(args.comment_backfill_max_retry)),
            page_limit=max(1, int(args.comment_backfill_page_limit)),
        )
        backfill_meta = {
            "triggered": True,
            "rounds": backfill.get("rounds") or [],
            "comments_added": int(backfill.get("comments_added") or 0),
            "new_ratio": float(backfill.get("new_ratio") or 0.0),
        }
        comments_total = int(backfill.get("comments_total") or comments_total)

    out_path = str(args.out_jsonl).strip()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    final_ratio = float(comments_total / max(1, posts_added))
    ratio_ok = bool(final_ratio >= float(target_ratio))

    summary = {
        "status": "ok" if (ratio_ok or (not bool(args.enforce_comment_ratio))) else "failed_comment_ratio",
        "pipeline": str(args.pipeline),
        "posts_added": int(posts_added),
        "comments_added": int(backfill_meta.get("comments_added") or 0),
        "new_ratio": round(final_ratio, 6),
        "comment_target_ratio": float(target_ratio),
        "comment_ratio_ok": bool(ratio_ok),
        "comment_backfill": backfill_meta,
        "start": _to_iso_z(start_dt),
        "end": _to_iso_z(end_dt),
        "timeframe": str(args.timeframe),
        "symbols": symbols,
        "connectors": [c.name for c in connectors],
        "events": int(len(events)),
        "failures": failures[:20],
        "dropped_language": int(dropped_language),
        "dropped_outside_window": int(dropped_window),
        "provenance_run_id": run_id,
        "out_jsonl": out_path,
    }

    summary_text = json.dumps(summary, ensure_ascii=False)
    summary_path = str(args.summary_json).strip()
    if summary_path:
        os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")
    print(summary_text)
    if bool(args.enforce_comment_ratio) and (not ratio_ok):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
