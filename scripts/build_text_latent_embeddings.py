#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _tokenize(text: str) -> List[str]:
    raw = str(text or "").lower()
    word_tokens = re.findall(r"[a-z0-9_]+", raw)
    # Include short CJK character ngrams for mixed-language social text.
    cjk_chars = [ch for ch in raw if "\u4e00" <= ch <= "\u9fff"]
    cjk_bigrams = [cjk_chars[i] + cjk_chars[i + 1] for i in range(0, max(0, len(cjk_chars) - 1))]
    return word_tokens + cjk_chars + cjk_bigrams


def _text_vec(text: str, dim: int, seed: str) -> List[float]:
    tokens = _tokenize(text)
    if not tokens:
        return [0.0] * dim
    bins = [0.0] * dim
    for tok in tokens:
        body = (seed + "||" + tok).encode("utf-8", errors="ignore")
        h = hashlib.blake2b(body, digest_size=16).digest()
        idx = int.from_bytes(h[:8], "big", signed=False) % dim
        sign = 1.0 if (h[8] % 2 == 0) else -1.0
        bins[idx] += sign * 1.0
    scale = max(1.0, float(len(tokens)) ** 0.5)
    return [float(v / scale) for v in bins]


def _norm_symbols(raw: Any) -> List[str]:
    if isinstance(raw, list):
        vals = raw
    else:
        vals = []
    out: List[str] = []
    seen = set()
    for x in vals:
        token = str(x or "").strip().upper().replace("$", "")
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _payload_symbols(payload: Dict[str, Any]) -> List[str]:
    symbols = _norm_symbols(payload.get("symbol_mentions") or [])
    if symbols:
        return symbols
    fallback_candidates = [
        payload.get("symbol"),
        payload.get("asset"),
        payload.get("ticker"),
    ]
    return _norm_symbols([x for x in fallback_candidates if x])


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _bucket_5m_end(ts: datetime) -> datetime:
    ts_utc = ts.astimezone(timezone.utc)
    step = 5 * 60
    epoch = int(ts_utc.timestamp())
    end_epoch = epoch if (epoch % step == 0) else ((epoch // step) + 1) * step
    return datetime.fromtimestamp(end_epoch, tz=timezone.utc)


def _clip_unit(v: float) -> float:
    return max(-1.0, min(1.0, float(v)))


def _merge_vec(target: List[float], source: List[float], weight: float) -> None:
    w = max(0.0, float(weight))
    if w <= 0.0:
        return
    for i in range(min(len(target), len(source))):
        target[i] += float(source[i]) * w


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _load_social_items_from_events(cur, *, start_dt: datetime, end_dt: datetime, max_rows: int) -> List[Dict[str, Any]]:
    sql = """
        SELECT
            e.id,
            COALESCE(e.available_at, e.occurred_at) AS ts,
            e.title,
            e.payload
        FROM events e
        WHERE COALESCE(e.payload->>'social_platform','none') NOT IN ('none','unknown','')
          AND COALESCE(e.available_at, e.occurred_at) >= %s
          AND COALESCE(e.available_at, e.occurred_at) <= %s
        ORDER BY COALESCE(e.available_at, e.occurred_at) ASC
    """
    params: List[object] = [start_dt, end_dt]
    if int(max_rows) > 0:
        sql += " LIMIT %s"
        params.append(int(max_rows))
    cur.execute(sql, tuple(params))
    rows = [dict(r) for r in cur.fetchall()]
    out: List[Dict[str, Any]] = []
    for r in rows:
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        ts = r.get("ts")
        if not isinstance(ts, datetime):
            continue
        symbols = _payload_symbols(payload)
        if not symbols:
            continue
        kind = str(payload.get("social_kind") or "post").strip().lower()
        title = str(r.get("title") or "").strip()
        summary = str(payload.get("summary") or "").strip()
        body = str(payload.get("text") or "").strip()
        out.append(
            {
                "event_id": r.get("id"),
                "ts": ts,
                "symbols": symbols,
                "kind": kind,
                "title": title,
                "summary": summary,
                "body": body,
                "comment_text": str(payload.get("top_comment") or payload.get("comment_excerpt") or summary or body).strip(),
                "engagement_score": max(0.0, _safe_float(payload.get("engagement_score"), 0.0)),
                "post_sentiment": _clip_unit(_safe_float(payload.get("post_sentiment"), 0.0)),
                "comment_sentiment": _clip_unit(_safe_float(payload.get("comment_sentiment"), 0.0)),
                "author": str(payload.get("author") or "").strip().lower(),
            }
        )
    return out


def _load_social_items_from_raw(cur, *, start_dt: datetime, end_dt: datetime, max_rows: int) -> List[Dict[str, Any]]:
    if (not _table_exists(cur, "social_posts_raw")) and (not _table_exists(cur, "social_comments_raw")):
        return []
    out: List[Dict[str, Any]] = []

    if _table_exists(cur, "social_posts_raw"):
        sql_posts = """
            SELECT
                available_at AS ts,
                platform,
                source_id,
                title,
                text,
                author,
                engagement_score,
                post_sentiment,
                comment_sentiment,
                symbol_mentions,
                payload
            FROM social_posts_raw
            WHERE available_at >= %s
              AND available_at <= %s
            ORDER BY available_at ASC
        """
        params_posts: List[object] = [start_dt, end_dt]
        if int(max_rows) > 0:
            sql_posts += " LIMIT %s"
            params_posts.append(int(max_rows))
        cur.execute(sql_posts, tuple(params_posts))
        for r in cur.fetchall() or []:
            row = dict(r)
            ts = row.get("ts")
            if not isinstance(ts, datetime):
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            symbols = _norm_symbols(row.get("symbol_mentions") or payload.get("symbol_mentions") or [])
            if not symbols:
                symbols = _payload_symbols(payload)
            if not symbols:
                continue
            title = str(row.get("title") or "").strip()
            body = str(row.get("text") or payload.get("text") or payload.get("summary") or "").strip()
            out.append(
                {
                    "event_id": None,
                    "ts": ts,
                    "symbols": symbols,
                    "kind": "post",
                    "title": title,
                    "summary": str(payload.get("summary") or "").strip(),
                    "body": body,
                    "comment_text": str(payload.get("top_comment") or payload.get("comment_excerpt") or body).strip(),
                    "engagement_score": max(0.0, _safe_float(row.get("engagement_score"), 0.0)),
                    "post_sentiment": _clip_unit(_safe_float(row.get("post_sentiment"), 0.0)),
                    "comment_sentiment": _clip_unit(_safe_float(row.get("comment_sentiment"), 0.0)),
                    "author": str(row.get("author") or "").strip().lower(),
                }
            )

    if _table_exists(cur, "social_comments_raw"):
        sql_comments = """
            SELECT
                available_at AS ts,
                platform,
                source_id,
                text,
                author,
                engagement_score,
                comment_sentiment,
                symbol_mentions,
                payload
            FROM social_comments_raw
            WHERE available_at >= %s
              AND available_at <= %s
            ORDER BY available_at ASC
        """
        params_comments: List[object] = [start_dt, end_dt]
        if int(max_rows) > 0:
            sql_comments += " LIMIT %s"
            params_comments.append(int(max_rows))
        cur.execute(sql_comments, tuple(params_comments))
        for r in cur.fetchall() or []:
            row = dict(r)
            ts = row.get("ts")
            if not isinstance(ts, datetime):
                continue
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            symbols = _norm_symbols(row.get("symbol_mentions") or payload.get("symbol_mentions") or [])
            if not symbols:
                symbols = _payload_symbols(payload)
            if not symbols:
                continue
            body = str(row.get("text") or payload.get("text") or payload.get("summary") or "").strip()
            out.append(
                {
                    "event_id": None,
                    "ts": ts,
                    "symbols": symbols,
                    "kind": "comment",
                    "title": "",
                    "summary": "",
                    "body": body,
                    "comment_text": body,
                    "engagement_score": max(0.0, _safe_float(row.get("engagement_score"), 0.0)),
                    "post_sentiment": _clip_unit(_safe_float(payload.get("post_sentiment"), 0.0)),
                    "comment_sentiment": _clip_unit(_safe_float(row.get("comment_sentiment"), payload.get("comment_sentiment"))),
                    "author": str(row.get("author") or "").strip().lower(),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build social text latent embeddings")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("TEXT_LATENT_MAX_ROWS", "0")))
    parser.add_argument("--post-dim", type=int, default=int(os.getenv("SOCIAL_POST_LATENT_DIM", "32")))
    parser.add_argument("--comment-dim", type=int, default=int(os.getenv("SOCIAL_COMMENT_LATENT_DIM", "32")))
    parser.add_argument("--input-source", choices=["auto", "raw", "events"], default=os.getenv("TEXT_LATENT_INPUT_SOURCE", "auto"))
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    post_dim = max(8, int(args.post_dim))
    comment_dim = max(8, int(args.comment_dim))

    inserted = 0
    source_used = "events"

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS social_text_latent (
                    id BIGSERIAL PRIMARY KEY,
                    event_id BIGINT,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    symbols TEXT[] NOT NULL,
                    post_latent JSONB NOT NULL,
                    comment_latent JSONB NOT NULL,
                    latent JSONB NOT NULL,
                    agg_features JSONB NOT NULL DEFAULT '{}'::jsonb,
                    latent_dim INTEGER NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("ALTER TABLE social_text_latent ADD COLUMN IF NOT EXISTS agg_features JSONB NOT NULL DEFAULT '{}'::jsonb")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_social_text_latent_ts ON social_text_latent(as_of_ts DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_social_text_latent_event ON social_text_latent(event_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_social_text_latent_symbols_gin ON social_text_latent USING GIN(symbols)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE social_text_latent")
            else:
                cur.execute(
                    """
                    DELETE FROM social_text_latent
                    WHERE as_of_ts >= %s AND as_of_ts <= %s
                    """,
                    (start_dt, end_dt),
                )

            social_items: List[Dict[str, Any]] = []
            source_mode = str(args.input_source or "auto").strip().lower()
            if source_mode in {"auto", "raw"}:
                social_items = _load_social_items_from_raw(
                    cur,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_rows=int(args.max_rows),
                )
                if social_items:
                    source_used = "raw"
            if (not social_items) and source_mode in {"auto", "events"}:
                social_items = _load_social_items_from_events(
                    cur,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_rows=int(args.max_rows),
                )
                source_used = "events"

            agg_map: Dict[Tuple[str, datetime], Dict[str, Any]] = {}
            for item_raw in social_items:
                ts = item_raw.get("ts")
                if not isinstance(ts, datetime):
                    continue
                symbols = _norm_symbols(item_raw.get("symbols") or [])
                if not symbols:
                    continue
                kind = str(item_raw.get("kind") or "post").strip().lower()
                title = str(item_raw.get("title") or "").strip()
                summary = str(item_raw.get("summary") or "").strip()
                body = str(item_raw.get("body") or "").strip()
                post_text = "\n".join(x for x in [title, summary, body] if x).strip()
                comment_text = str(item_raw.get("comment_text") or summary or body).strip()
                post_latent = _text_vec(post_text, post_dim, "post")
                comment_latent = _text_vec(comment_text, comment_dim, "comment")
                engagement = max(0.0, _safe_float(item_raw.get("engagement_score"), 0.0))
                post_sent = _clip_unit(_safe_float(item_raw.get("post_sentiment"), 0.0))
                comment_sent = _clip_unit(_safe_float(item_raw.get("comment_sentiment"), 0.0))
                author = str(item_raw.get("author") or "").strip().lower()
                bucket_end = _bucket_5m_end(ts)
                base_weight = max(1.0, 1.0 + (engagement ** 0.5))

                for symbol in symbols:
                    key = (symbol, bucket_end)
                    item = agg_map.get(key)
                    if item is None:
                        item = {
                            "symbol": symbol,
                            "as_of_ts": bucket_end,
                            "post_sum": [0.0] * post_dim,
                            "comment_sum": [0.0] * comment_dim,
                            "vector_weight": 0.0,
                            "post_weight": 0.0,
                            "comment_weight": 0.0,
                            "post_count": 0,
                            "comment_count": 0,
                            "item_count": 0,
                            "engagement_sum": 0.0,
                            "post_sent_num": 0.0,
                            "comment_sent_num": 0.0,
                            "authors": set(),
                        }
                        agg_map[key] = item

                    item["item_count"] += 1
                    item["engagement_sum"] += engagement
                    item["vector_weight"] += base_weight
                    item["post_sent_num"] += base_weight * post_sent
                    item["comment_sent_num"] += base_weight * comment_sent
                    item["authors"].add(author or f"anon_{len(item['authors'])}")
                    if kind == "comment":
                        item["comment_count"] += 1
                        item["comment_weight"] += base_weight
                    else:
                        item["post_count"] += 1
                        item["post_weight"] += base_weight
                    _merge_vec(item["post_sum"], post_latent, base_weight)
                    _merge_vec(item["comment_sum"], comment_latent, base_weight)

            insert_rows = []
            for _, item in sorted(agg_map.items(), key=lambda x: (x[0][0], x[0][1])):
                post_w = max(1e-9, float(item["post_weight"]))
                comment_w = max(1e-9, float(item["comment_weight"]))
                vector_w = max(1e-9, float(item["vector_weight"]))
                total_w = max(1e-9, float(post_w + comment_w))
                post_vec = [float(v / vector_w) for v in item["post_sum"]]
                comment_vec = [float(v / vector_w) for v in item["comment_sum"]]
                latent = post_vec + comment_vec
                item_count = int(item["item_count"])
                post_count = int(item["post_count"])
                comment_count = int(item["comment_count"])
                engagement_sum = float(item["engagement_sum"])
                agg_features = {
                    "bucket_size_min": 5,
                    "item_count": item_count,
                    "post_count": post_count,
                    "comment_count": comment_count,
                    "post_ratio": float(post_count / max(1, item_count)),
                    "comment_ratio": float(comment_count / max(1, item_count)),
                    "unique_author_count": int(len(item["authors"])),
                    "engagement_sum": engagement_sum,
                    "engagement_mean": float(engagement_sum / max(1, item_count)),
                    "post_sentiment_weighted": float(item["post_sent_num"] / total_w),
                    "comment_sentiment_weighted": float(item["comment_sent_num"] / total_w),
                    "text_buzz_log": float((1.0 + item_count) ** 0.5 - 1.0),
                }
                insert_rows.append(
                    (
                        None,
                        item["as_of_ts"],
                        [item["symbol"]],
                        json.dumps(post_vec),
                        json.dumps(comment_vec),
                        json.dumps(latent),
                        json.dumps(agg_features),
                        len(latent),
                    )
                )

            if insert_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO social_text_latent(
                        event_id, as_of_ts, symbols, post_latent, comment_latent, latent, agg_features, latent_dim
                    )
                    VALUES %s
                    """,
                    insert_rows,
                    template="(%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s::jsonb, %s)",
                )
                inserted = len(insert_rows)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "social_text_latent",
                "rows_inserted": int(inserted),
                "aggregation_mode": "symbol_5m_bucket_set_pooling_v1",
                "latent_dim": int(post_dim + comment_dim),
                "post_dim": int(post_dim),
                "comment_dim": int(comment_dim),
                "bucket_minutes": 5,
                "source_used": source_used,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
