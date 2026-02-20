#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import psycopg2
from psycopg2.extras import execute_values

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


def _parse_dt(raw: Any) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw or "").strip()
        if not text:
            dt = datetime.now(timezone.utc)
        else:
            text = text.replace(" ", "T")
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        return int(float(raw))
    except Exception:
        return int(default)


def _norm_symbols(raw: Any) -> List[str]:
    vals: Sequence[Any]
    if isinstance(raw, list):
        vals = raw
    elif isinstance(raw, str):
        vals = [x.strip() for x in raw.split(",") if x.strip()]
    else:
        vals = []
    out: List[str] = []
    seen = set()
    for x in vals:
        sym = str(x or "").strip().upper().replace("$", "")
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _stable_id(*parts: Any) -> str:
    body = "||".join(str(x or "") for x in parts)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _ensure_tables(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS social_posts_raw (
            id BIGSERIAL PRIMARY KEY,
            platform TEXT NOT NULL,
            source_id TEXT NOT NULL,
            occurred_at TIMESTAMPTZ NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            available_at TIMESTAMPTZ NOT NULL,
            author TEXT,
            author_id TEXT,
            title TEXT,
            text TEXT,
            engagement_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            post_sentiment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            comment_sentiment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            n_comments INTEGER NOT NULL DEFAULT 0,
            n_replies INTEGER NOT NULL DEFAULT 0,
            symbol_mentions TEXT[] NOT NULL DEFAULT '{}'::text[],
            entity_links JSONB NOT NULL DEFAULT '[]'::jsonb,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(platform, source_id, available_at)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS social_comments_raw (
            id BIGSERIAL PRIMARY KEY,
            platform TEXT NOT NULL,
            source_id TEXT NOT NULL,
            parent_source_id TEXT,
            occurred_at TIMESTAMPTZ NOT NULL,
            published_at TIMESTAMPTZ NOT NULL,
            available_at TIMESTAMPTZ NOT NULL,
            author TEXT,
            author_id TEXT,
            text TEXT,
            engagement_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            comment_sentiment DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            symbol_mentions TEXT[] NOT NULL DEFAULT '{}'::text[],
            entity_links JSONB NOT NULL DEFAULT '[]'::jsonb,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            source_url TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE(platform, source_id, available_at)
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_social_posts_raw_available_at ON social_posts_raw(available_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_social_comments_raw_available_at ON social_comments_raw(available_at DESC)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_social_posts_raw_symbols_gin ON social_posts_raw USING GIN(symbol_mentions)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_social_comments_raw_symbols_gin ON social_comments_raw USING GIN(symbol_mentions)")


def _to_raw_rows(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    post_rows: List[Tuple[Any, ...]] = []
    comment_rows: List[Tuple[Any, ...]] = []
    for obj in rows:
        payload = obj.get("payload") if isinstance(obj.get("payload"), dict) else {}
        platform = str(payload.get("social_platform") or obj.get("social_platform") or "unknown").strip().lower() or "unknown"
        title = str(obj.get("title") or payload.get("summary") or "").strip()
        text = str(payload.get("text") or payload.get("summary") or obj.get("content") or title).strip()
        source_url = str(obj.get("source_url") or payload.get("source_url") or "")
        source_id = str(payload.get("source_id") or payload.get("tweet_id") or payload.get("video_id") or "").strip()
        if not source_id:
            source_id = _stable_id(platform, source_url, obj.get("occurred_at"), title)

        occurred_at = _parse_dt(obj.get("occurred_at"))
        published_at = _parse_dt(obj.get("published_at") or occurred_at)
        available_at = _parse_dt(obj.get("available_at") or published_at)
        if published_at < occurred_at:
            published_at = occurred_at
        if available_at < published_at:
            available_at = published_at

        symbols = _norm_symbols(payload.get("symbol_mentions") or obj.get("symbol_mentions") or [])
        entities = obj.get("entities") if isinstance(obj.get("entities"), list) else []
        engagement = max(0.0, _safe_float(payload.get("engagement_score"), 0.0))
        post_sent = max(-1.0, min(1.0, _safe_float(payload.get("post_sentiment"), 0.0)))
        comment_sent = max(-1.0, min(1.0, _safe_float(payload.get("comment_sentiment"), 0.0)))
        n_comments = max(0, _safe_int(payload.get("n_comments"), 0))
        n_replies = max(0, _safe_int(payload.get("n_replies"), 0))
        author = str(payload.get("author") or obj.get("author") or "").strip()
        author_id = str(payload.get("author_id") or obj.get("author_id") or "").strip()

        post_rows.append(
            (
                platform,
                source_id,
                occurred_at,
                published_at,
                available_at,
                author,
                author_id,
                title,
                text,
                engagement,
                post_sent,
                comment_sent,
                n_comments,
                n_replies,
                symbols,
                json.dumps(entities, ensure_ascii=False),
                json.dumps(payload, ensure_ascii=False),
                source_url,
            )
        )

        comments_raw = obj.get("comments") if isinstance(obj.get("comments"), list) else []
        if comments_raw:
            for idx, c in enumerate(comments_raw):
                if not isinstance(c, dict):
                    continue
                c_text = str(c.get("text") or c.get("content") or "").strip()
                if not c_text:
                    continue
                c_id = str(c.get("id") or "").strip() or _stable_id(source_id, "comment", idx, c_text)
                c_author = str(c.get("author") or "").strip()
                c_author_id = str(c.get("author_id") or "").strip()
                c_sent = max(-1.0, min(1.0, _safe_float(c.get("sentiment"), comment_sent)))
                c_eng = max(0.0, _safe_float(c.get("engagement_score"), 0.0))
                comment_rows.append(
                    (
                        platform,
                        c_id,
                        source_id,
                        occurred_at,
                        published_at,
                        available_at,
                        c_author,
                        c_author_id,
                        c_text,
                        c_eng,
                        c_sent,
                        symbols,
                        json.dumps(entities, ensure_ascii=False),
                        json.dumps(c, ensure_ascii=False),
                        source_url,
                    )
                )
        elif (n_comments + n_replies) > 0:
            synthetic_id = _stable_id(source_id, "synthetic", available_at.isoformat())
            comment_rows.append(
                (
                    platform,
                    synthetic_id,
                    source_id,
                    occurred_at,
                    published_at,
                    available_at,
                    author,
                    author_id,
                    str(payload.get("top_comment") or payload.get("comment_excerpt") or text[:500]).strip(),
                    float(n_comments + n_replies),
                    comment_sent,
                    symbols,
                    json.dumps(entities, ensure_ascii=False),
                    json.dumps({"synthetic": True, "n_comments": n_comments, "n_replies": n_replies}, ensure_ascii=False),
                    source_url,
                )
            )
    return post_rows, comment_rows


def _insert_rows(cur, post_rows: List[Tuple[Any, ...]], comment_rows: List[Tuple[Any, ...]]) -> Tuple[int, int]:
    posts_inserted = 0
    comments_inserted = 0
    if post_rows:
        execute_values(
            cur,
            """
            INSERT INTO social_posts_raw(
                platform, source_id, occurred_at, published_at, available_at, author, author_id, title, text,
                engagement_score, post_sentiment, comment_sentiment, n_comments, n_replies,
                symbol_mentions, entity_links, payload, source_url, created_at
            ) VALUES %s
            ON CONFLICT (platform, source_id, available_at) DO UPDATE SET
                title = EXCLUDED.title,
                text = EXCLUDED.text,
                engagement_score = EXCLUDED.engagement_score,
                post_sentiment = EXCLUDED.post_sentiment,
                comment_sentiment = EXCLUDED.comment_sentiment,
                n_comments = EXCLUDED.n_comments,
                n_replies = EXCLUDED.n_replies,
                symbol_mentions = EXCLUDED.symbol_mentions,
                entity_links = EXCLUDED.entity_links,
                payload = EXCLUDED.payload,
                source_url = EXCLUDED.source_url
            """,
            post_rows,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,NOW())",
        )
        posts_inserted = len(post_rows)
    if comment_rows:
        execute_values(
            cur,
            """
            INSERT INTO social_comments_raw(
                platform, source_id, parent_source_id, occurred_at, published_at, available_at, author, author_id, text,
                engagement_score, comment_sentiment, symbol_mentions, entity_links, payload, source_url, created_at
            ) VALUES %s
            ON CONFLICT (platform, source_id, available_at) DO UPDATE SET
                parent_source_id = EXCLUDED.parent_source_id,
                text = EXCLUDED.text,
                engagement_score = EXCLUDED.engagement_score,
                comment_sentiment = EXCLUDED.comment_sentiment,
                symbol_mentions = EXCLUDED.symbol_mentions,
                entity_links = EXCLUDED.entity_links,
                payload = EXCLUDED.payload,
                source_url = EXCLUDED.source_url
            """,
            comment_rows,
            template="(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,NOW())",
        )
        comments_inserted = len(comment_rows)
    return posts_inserted, comments_inserted


def main() -> int:
    ap = argparse.ArgumentParser(description="Import social JSONL into social_posts_raw/social_comments_raw")
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rows = _read_jsonl(str(args.jsonl))
    post_rows, comment_rows = _to_raw_rows(rows)

    posts_inserted = 0
    comments_inserted = 0
    if not bool(args.dry_run):
        with psycopg2.connect(args.database_url) as conn:
            conn.autocommit = False
            with conn.cursor() as cur:
                _ensure_tables(cur)
                posts_inserted, comments_inserted = _insert_rows(cur, post_rows, comment_rows)
            conn.commit()

    out = {
        "status": "ok",
        "jsonl": str(args.jsonl),
        "rows_loaded": int(len(rows)),
        "posts_converted": int(len(post_rows)),
        "comments_converted": int(len(comment_rows)),
        "posts_inserted": int(posts_inserted),
        "comments_inserted": int(comments_inserted),
        "dry_run": bool(args.dry_run),
        "schema_tables": ["social_posts_raw", "social_comments_raw"],
    }
    text = json.dumps(out, ensure_ascii=False)
    if str(args.out_json).strip():
        os.makedirs(os.path.dirname(str(args.out_json)) or ".", exist_ok=True)
        with open(str(args.out_json), "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
