#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right, bisect_left
import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from inference.liquid_feature_contract import (
    LIQUID_FEATURE_SCHEMA_VERSION,
    LIQUID_FULL_FEATURE_KEYS,
    LIQUID_LATENT_FEATURE_KEYS,
    LIQUID_MANUAL_FEATURE_KEYS,
)


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_list(raw: Any) -> List[float]:
    if isinstance(raw, list):
        vals = raw
    else:
        vals = []
    out: List[float] = []
    for x in vals:
        try:
            out.append(float(x))
        except Exception:
            out.append(0.0)
    return out


def _to_dict(raw: Any) -> Dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _avg_vectors(vectors: List[List[float]], dim: int) -> List[float]:
    if not vectors:
        return [0.0] * dim
    out = [0.0] * dim
    for v in vectors:
        for i in range(min(dim, len(v))):
            out[i] += float(v[i])
    n = float(len(vectors))
    return [x / n for x in out]


def _latest_before(ts_list: List[datetime], vec_list: List[List[float]], as_of_ts: datetime) -> List[float]:
    if not ts_list or not vec_list:
        return []
    idx = bisect_right(ts_list, as_of_ts) - 1
    if idx < 0:
        return []
    return vec_list[idx]


def _window_vectors(
    ts_list: List[datetime],
    vec_list: List[List[float]],
    start_ts: datetime,
    end_ts: datetime,
    max_count: int = 256,
) -> List[List[float]]:
    if not ts_list or not vec_list:
        return []
    left = bisect_left(ts_list, start_ts)
    right = bisect_right(ts_list, end_ts)
    if right <= left:
        return []
    sliced = vec_list[left:right]
    if len(sliced) > max_count:
        sliced = sliced[-max_count:]
    return sliced


def _window_items(
    ts_list: List[datetime],
    val_list: List[Dict[str, Any]],
    start_ts: datetime,
    end_ts: datetime,
    max_count: int = 256,
) -> List[Dict[str, Any]]:
    if not ts_list or not val_list:
        return []
    left = bisect_left(ts_list, start_ts)
    right = bisect_right(ts_list, end_ts)
    if right <= left:
        return []
    sliced = val_list[left:right]
    if len(sliced) > max_count:
        sliced = sliced[-max_count:]
    return sliced


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _social_manual_from_agg(rows: List[Dict[str, Any]], as_of_ts: datetime) -> Dict[str, float]:
    if not rows:
        return {}
    total_items = 0.0
    total_posts = 0.0
    total_comments = 0.0
    total_engagement = 0.0
    post_sent_num = 0.0
    comment_sent_num = 0.0
    count_1h = 0.0
    count_6h = 0.0

    lookback_1h = as_of_ts - timedelta(hours=1)
    lookback_6h = as_of_ts - timedelta(hours=6)
    for row in rows:
        ts = row.get("_as_of_ts")
        item_count = max(0.0, _safe_float(row.get("item_count"), 0.0))
        post_count = max(0.0, _safe_float(row.get("post_count"), 0.0))
        comment_count = max(0.0, _safe_float(row.get("comment_count"), 0.0))
        engagement_sum = max(0.0, _safe_float(row.get("engagement_sum"), 0.0))
        post_sent = max(-1.0, min(1.0, _safe_float(row.get("post_sentiment_weighted"), 0.0)))
        comment_sent = max(-1.0, min(1.0, _safe_float(row.get("comment_sentiment_weighted"), 0.0)))

        w = max(1.0, item_count)
        total_items += item_count
        total_posts += post_count
        total_comments += comment_count
        total_engagement += engagement_sum
        post_sent_num += w * post_sent
        comment_sent_num += w * comment_sent
        if isinstance(ts, datetime):
            if ts >= lookback_6h:
                count_6h += item_count
            if ts >= lookback_1h:
                count_1h += item_count

    den = max(1.0, total_items)
    post_sent_mean = post_sent_num / den
    comment_sent_mean = comment_sent_num / den
    social_comment_rate = total_comments / max(1.0, total_posts)
    social_event_ratio = min(1.0, total_items / 72.0)  # 6h window with 5m bins -> 72 buckets
    engagement_norm = math.tanh((math.log1p(total_engagement / den)) / 3.0)
    buzz_norm = math.tanh(math.log1p(total_items) / 4.0)
    event_density = min(1.0, total_items / 72.0)
    return {
        "social_post_sentiment": float(post_sent_mean),
        "social_comment_sentiment": float(comment_sent_mean),
        "social_engagement_norm": float(engagement_norm),
        "social_event_ratio": float(social_event_ratio),
        "social_buzz": float(buzz_norm),
        "social_comment_rate": float(social_comment_rate),
        "sentiment_abs": float(abs(0.5 * (post_sent_mean + comment_sent_mean))),
        "comment_skew": float(comment_sent_mean - post_sent_mean),
        "event_density": float(event_density),
        "event_velocity_1h": float(math.tanh(count_1h / 12.0)),
        "event_velocity_6h": float(math.tanh(count_6h / 72.0)),
    }


def _blend_manual_features(
    final: Dict[str, float],
    derived: Dict[str, float],
    *,
    alpha: float,
) -> None:
    w = max(0.0, min(1.0, float(alpha)))
    for key, dv in derived.items():
        if key not in final:
            continue
        base = _safe_float(final.get(key), 0.0)
        derived_v = _safe_float(dv, 0.0)
        if abs(base) <= 1e-12:
            final[key] = float(derived_v)
        else:
            final[key] = float((1.0 - w) * base + w * derived_v)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge feature views into final training matrix")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--truncate", action="store_true")
    parser.add_argument("--max-rows", type=int, default=int(os.getenv("FEATURE_MATRIX_MAIN_MAX_ROWS", "0")))
    parser.add_argument("--social-agg-blend-alpha", type=float, default=float(os.getenv("SOCIAL_AGG_BLEND_ALPHA", "0.35")))
    args = parser.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)

    inserted = 0

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_matrix_main (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    features JSONB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    feature_version TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_matrix_main_symbol_ts ON feature_matrix_main(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE feature_matrix_main")

            sql = """
                SELECT symbol, as_of_ts, feature_payload
                FROM feature_snapshots_main
                WHERE as_of_ts >= %s AND as_of_ts <= %s
                ORDER BY as_of_ts ASC
            """
            params: List[object] = [start_dt, end_dt]
            if int(args.max_rows) > 0:
                sql += " LIMIT %s"
                params.append(int(args.max_rows))
            cur.execute(sql, tuple(params))
            base_rows = [dict(r) for r in cur.fetchall()]

            symbols = sorted({str(r.get("symbol") or "").upper() for r in base_rows if str(r.get("symbol") or "").strip()})
            if symbols:
                # Idempotency: rebuild same window/version without duplicate accumulation.
                cur.execute(
                    """
                    DELETE FROM feature_matrix_main
                    WHERE symbol = ANY(%s)
                      AND as_of_ts >= %s
                      AND as_of_ts <= %s
                      AND feature_version = %s
                    """,
                    (symbols, start_dt, end_dt, str(LIQUID_FEATURE_SCHEMA_VERSION)),
                )
            market_ts_map: Dict[str, List[datetime]] = {}
            market_vec_map: Dict[str, List[List[float]]] = {}
            social_ts_map: Dict[str, List[datetime]] = {}
            social_vec_map: Dict[str, List[List[float]]] = {}
            social_feat_map: Dict[str, List[Dict[str, Any]]] = {}

            if symbols:
                if _table_exists(cur, "market_latent"):
                    cur.execute(
                        """
                        SELECT symbol, as_of_ts, latent
                        FROM market_latent
                        WHERE symbol = ANY(%s)
                          AND as_of_ts >= %s
                          AND as_of_ts <= %s
                        ORDER BY symbol ASC, as_of_ts ASC
                        """,
                        (symbols, start_dt - timedelta(days=7), end_dt),
                    )
                    for row in cur.fetchall() or []:
                        sym = str(row.get("symbol") or "").upper()
                        ts = row.get("as_of_ts")
                        if not sym or not isinstance(ts, datetime):
                            continue
                        market_ts_map.setdefault(sym, []).append(ts)
                        market_vec_map.setdefault(sym, []).append(_to_list(row.get("latent")))

                if _table_exists(cur, "social_text_latent"):
                    cur.execute(
                        """
                        SELECT as_of_ts, symbols, latent, agg_features
                        FROM social_text_latent
                        WHERE as_of_ts >= %s
                          AND as_of_ts <= %s
                        ORDER BY as_of_ts ASC
                        """,
                        (start_dt - timedelta(hours=6), end_dt),
                    )
                    for row in cur.fetchall() or []:
                        ts = row.get("as_of_ts")
                        syms = row.get("symbols")
                        if not isinstance(ts, datetime) or not isinstance(syms, list):
                            continue
                        vec = _to_list(row.get("latent"))
                        feat = _to_dict(row.get("agg_features"))
                        for s in syms:
                            sym = str(s or "").upper()
                            if sym in symbols:
                                social_ts_map.setdefault(sym, []).append(ts)
                                social_vec_map.setdefault(sym, []).append(vec)
                                social_feat_map.setdefault(sym, []).append({**feat, "_as_of_ts": ts})

            insert_rows = []
            for r in base_rows:
                symbol = str(r.get("symbol") or "").upper()
                as_of_ts = r.get("as_of_ts")
                if not isinstance(as_of_ts, datetime):
                    continue
                manual_payload = r.get("feature_payload") if isinstance(r.get("feature_payload"), dict) else {}

                market_latent = _latest_before(
                    market_ts_map.get(symbol, []),
                    market_vec_map.get(symbol, []),
                    as_of_ts,
                )
                social_vectors = _window_vectors(
                    social_ts_map.get(symbol, []),
                    social_vec_map.get(symbol, []),
                    start_ts=as_of_ts - timedelta(hours=6),
                    end_ts=as_of_ts,
                    max_count=256,
                )
                social_feature_rows = _window_items(
                    social_ts_map.get(symbol, []),
                    social_feat_map.get(symbol, []),
                    start_ts=as_of_ts - timedelta(hours=6),
                    end_ts=as_of_ts,
                    max_count=256,
                )
                social_latent_target_dim = len(LIQUID_LATENT_FEATURE_KEYS) // 2
                social_latent = _avg_vectors(social_vectors, social_latent_target_dim)

                final = {k: 0.0 for k in LIQUID_FULL_FEATURE_KEYS}
                for k in LIQUID_MANUAL_FEATURE_KEYS:
                    final[k] = float(manual_payload.get(k, 0.0) or 0.0)

                half = len(LIQUID_LATENT_FEATURE_KEYS) // 2
                for i in range(half):
                    final[LIQUID_LATENT_FEATURE_KEYS[i]] = float(market_latent[i] if i < len(market_latent) else 0.0)
                for i in range(half, len(LIQUID_LATENT_FEATURE_KEYS)):
                    j = i - half
                    final[LIQUID_LATENT_FEATURE_KEYS[i]] = float(social_latent[j] if j < len(social_latent) else 0.0)
                derived_social = _social_manual_from_agg(social_feature_rows, as_of_ts=as_of_ts)
                _blend_manual_features(final, derived_social, alpha=float(args.social_agg_blend_alpha))

                insert_rows.append(
                    (
                        symbol,
                        as_of_ts,
                        json.dumps(final),
                        len(LIQUID_FULL_FEATURE_KEYS),
                        str(LIQUID_FEATURE_SCHEMA_VERSION),
                    )
                )

            if insert_rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO feature_matrix_main(symbol, as_of_ts, features, feature_dim, feature_version)
                    VALUES %s
                    """,
                    insert_rows,
                    template="(%s, %s, %s::jsonb, %s, %s)",
                )
                inserted += len(insert_rows)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "feature_matrix_main",
                "rows_inserted": int(inserted),
                "manual_dim": len(LIQUID_MANUAL_FEATURE_KEYS),
                "latent_dim": len(LIQUID_LATENT_FEATURE_KEYS),
                "total_dim": len(LIQUID_FULL_FEATURE_KEYS),
                "feature_version": str(LIQUID_FEATURE_SCHEMA_VERSION),
                "social_agg_blend_alpha": float(args.social_agg_blend_alpha),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
