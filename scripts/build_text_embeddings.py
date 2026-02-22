#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from features.feature_contract import SCHEMA_HASH
from text.embedder import encode_texts, load_model

DEFAULT_TEXT_EMBED_MODEL_PATH = "artifacts/models/text_encoder/multilingual-e5-small"


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _bucket_5m(ts: datetime) -> datetime:
    step = 300
    ep = int(ts.timestamp())
    return datetime.fromtimestamp((ep // step) * step, tz=timezone.utc)


def _norm_symbol(raw: object) -> str:
    return str(raw or "").strip().upper().replace("$", "")


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _load_rows(cur, start_dt: datetime, end_dt: datetime, max_rows: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if _table_exists(cur, "social_posts_raw"):
        sql = """
            SELECT available_at AS ts, symbol_mentions, text, author, engagement_score, payload
            FROM social_posts_raw
            WHERE available_at >= %s AND available_at <= %s
            ORDER BY available_at ASC
        """
        params: List[Any] = [start_dt, end_dt]
        if max_rows > 0:
            sql += " LIMIT %s"
            params.append(max_rows)
        cur.execute(sql, tuple(params))
        for r in cur.fetchall() or []:
            row = dict(r)
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            syms = row.get("symbol_mentions") if isinstance(row.get("symbol_mentions"), list) else payload.get("symbol_mentions")
            syms = syms if isinstance(syms, list) else []
            text = str(row.get("text") or payload.get("text") or payload.get("summary") or "").strip()
            if not text:
                continue
            for sym in syms:
                s = _norm_symbol(sym)
                if not s:
                    continue
                rows.append(
                    {
                        "symbol": s,
                        "ts": row.get("ts"),
                        "text": text,
                        "author": str(row.get("author") or "").strip().lower(),
                        "engagement": float(row.get("engagement_score") or 0.0),
                    }
                )

    if _table_exists(cur, "social_comments_raw"):
        sql = """
            SELECT available_at AS ts, symbol_mentions, text, author, engagement_score, payload
            FROM social_comments_raw
            WHERE available_at >= %s AND available_at <= %s
            ORDER BY available_at ASC
        """
        params = [start_dt, end_dt]
        if max_rows > 0:
            sql += " LIMIT %s"
            params.append(max_rows)
        cur.execute(sql, tuple(params))
        for r in cur.fetchall() or []:
            row = dict(r)
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            syms = row.get("symbol_mentions") if isinstance(row.get("symbol_mentions"), list) else payload.get("symbol_mentions")
            syms = syms if isinstance(syms, list) else []
            text = str(row.get("text") or payload.get("text") or payload.get("summary") or "").strip()
            if not text:
                continue
            for sym in syms:
                s = _norm_symbol(sym)
                if not s:
                    continue
                rows.append(
                    {
                        "symbol": s,
                        "ts": row.get("ts"),
                        "text": text,
                        "author": str(row.get("author") or "").strip().lower(),
                        "engagement": float(row.get("engagement_score") or 0.0),
                    }
                )

    if not rows and _table_exists(cur, "events"):
        sql = """
            SELECT COALESCE(e.available_at, e.occurred_at) AS ts, e.payload, e.title
            FROM events e
            WHERE COALESCE(e.available_at, e.occurred_at) >= %s
              AND COALESCE(e.available_at, e.occurred_at) <= %s
            ORDER BY COALESCE(e.available_at, e.occurred_at) ASC
        """
        params = [start_dt, end_dt]
        if max_rows > 0:
            sql += " LIMIT %s"
            params.append(max_rows)
        cur.execute(sql, tuple(params))
        for r in cur.fetchall() or []:
            row = dict(r)
            payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
            syms = payload.get("symbol_mentions") if isinstance(payload.get("symbol_mentions"), list) else []
            text = "\n".join([str(row.get("title") or "").strip(), str(payload.get("summary") or "").strip()]).strip()
            if not text:
                continue
            for sym in syms:
                s = _norm_symbol(sym)
                if not s:
                    continue
                rows.append(
                    {
                        "symbol": s,
                        "ts": row.get("ts"),
                        "text": text,
                        "author": str(payload.get("author") or "").strip().lower(),
                        "engagement": float(payload.get("engagement_score") or 0.0),
                    }
                )

    return [r for r in rows if isinstance(r.get("ts"), datetime)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Build semantic text embeddings by 5m bucket")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument(
        "--model-path",
        default=os.getenv("TEXT_EMBED_MODEL_PATH", DEFAULT_TEXT_EMBED_MODEL_PATH),
    )
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--truncate", action="store_true")
    args = ap.parse_args()

    model_path = str(args.model_path or "").strip()
    if not model_path:
        raise RuntimeError("text_embedding_model_path_required")
    if not os.path.isdir(model_path):
        raise RuntimeError(
            "text_embedding_model_missing_local_dir:"
            f"{model_path}; run `python3 scripts/setup_text_encoder.py` first"
        )
    model = load_model(model_path)

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)

    inserted = 0
    emb_dim = 0
    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS social_text_embeddings (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    embedding JSONB NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    text_quality JSONB NOT NULL,
                    schema_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_social_text_embeddings_symbol_ts ON social_text_embeddings(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE social_text_embeddings")
            else:
                cur.execute(
                    "DELETE FROM social_text_embeddings WHERE as_of_ts >= %s AND as_of_ts <= %s",
                    (start_dt, end_dt),
                )

            rows = _load_rows(cur, start_dt=start_dt, end_dt=end_dt, max_rows=max(0, int(args.max_rows)))

            grouped: Dict[Tuple[str, datetime], List[Dict[str, Any]]] = defaultdict(list)
            for r in rows:
                key = (str(r["symbol"]), _bucket_5m(r["ts"]))
                grouped[key].append(r)

            inserts = []
            for (symbol, bts), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
                texts = [str(it["text"]) for it in items]
                vectors = encode_texts(model, texts)
                if vectors.size <= 0:
                    continue
                emb_dim = int(vectors.shape[1])
                weights = np.array([1.0 + max(0.0, float(it.get("engagement") or 0.0)) ** 0.5 for it in items], dtype=np.float32)
                weights = weights / max(1e-9, float(np.sum(weights)))
                pooled = np.sum(vectors * weights.reshape(-1, 1), axis=0)

                authors = [str(it.get("author") or "") for it in items]
                unique_auth = len({a for a in authors if a})
                dup_ratio = 1.0 - (unique_auth / max(1, len(items)))
                disagreement = float(np.mean(np.std(vectors, axis=0)))
                avg_lag = float(np.mean([(bts - it["ts"]).total_seconds() for it in items]))
                quality = {
                    "num_items": int(len(items)),
                    "unique_authors": int(unique_auth),
                    "dup_ratio": float(max(0.0, min(1.0, dup_ratio))),
                    "disagreement": float(disagreement),
                    "avg_lag_sec": float(max(0.0, avg_lag)),
                    "coverage": float(min(1.0, len(items) / 12.0)),
                }
                inserts.append(
                    (
                        symbol,
                        bts,
                        json.dumps([float(x) for x in pooled.tolist()]),
                        int(emb_dim),
                        json.dumps(quality),
                        str(SCHEMA_HASH),
                    )
                )

            if inserts:
                execute_values(
                    cur,
                    """
                    INSERT INTO social_text_embeddings(symbol, as_of_ts, embedding, embedding_dim, text_quality, schema_hash)
                    VALUES %s
                    """,
                    inserts,
                    template="(%s,%s,%s::jsonb,%s,%s::jsonb,%s)",
                )
                inserted = len(inserts)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "social_text_embeddings",
                "rows_inserted": int(inserted),
                "embedding_dim": int(emb_dim),
                "schema_hash": str(SCHEMA_HASH),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
