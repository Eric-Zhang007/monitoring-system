#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]

FIXED_START_ISO = "2018-01-01T00:00:00Z"
FIXED_START_DT = datetime(2018, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
FIXED_TIMEFRAME = "5m"
TARGET_RATIO = 10.0


def _parse_iso_utc(raw: str) -> datetime:
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


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for piece in str(raw or "").split(","):
        sym = piece.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _bucket_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").strip().lower()
    if tf != FIXED_TIMEFRAME:
        raise ValueError(f"timeframe_must_be_{FIXED_TIMEFRAME}")
    return 300


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema='public' AND table_name=%s
        """,
        (table_name,),
    )
    rows = cur.fetchall() or []
    return {str(r.get("column_name")) for r in rows if r.get("column_name")}


def _pick_col(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _expected_bucket_count(start_dt: datetime, end_dt: datetime, bucket_sec: int) -> int:
    if end_dt < start_dt:
        return 0
    return int((end_dt - start_dt).total_seconds() // bucket_sec) + 1


def _bucket_epoch(ts: datetime, bucket_sec: int) -> int:
    return int(ts.timestamp() // bucket_sec) * bucket_sec


def _fetch_bucket_epochs(
    cur,
    *,
    table_name: str,
    ts_col: str,
    start_dt: datetime,
    end_dt: datetime,
    bucket_sec: int,
    symbol_col: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe_col: Optional[str] = None,
    timeframe: Optional[str] = None,
    extra_filter_sql: str = "",
    extra_params: Sequence[Any] = (),
) -> List[int]:
    where = [f"{ts_col} >= %s", f"{ts_col} <= %s"]
    params: List[Any] = [start_dt, end_dt]
    if symbol_col and symbol:
        where.append(f"UPPER({symbol_col}) = %s")
        params.append(str(symbol).upper())
    if timeframe_col and timeframe:
        where.append(f"{timeframe_col} = %s")
        params.append(str(timeframe))
    if extra_filter_sql.strip():
        where.append(f"({extra_filter_sql})")
        params.extend(list(extra_params))
    sql = (
        "SELECT DISTINCT FLOOR(EXTRACT(EPOCH FROM "
        + ts_col
        + ") / %s)::bigint AS bucket_epoch "
        + f"FROM {table_name} WHERE "
        + " AND ".join(where)
        + " ORDER BY bucket_epoch ASC"
    )
    cur.execute(sql, tuple([bucket_sec] + params))
    return [int(r.get("bucket_epoch")) for r in (cur.fetchall() or []) if r.get("bucket_epoch") is not None]


def _gap_ranges_from_epochs(
    *,
    epochs: Sequence[int],
    start_epoch: int,
    end_epoch: int,
    bucket_sec: int,
) -> List[Tuple[int, int, int]]:
    unique_epochs = sorted(set(int(x) for x in epochs if start_epoch <= int(x) <= end_epoch))
    out: List[Tuple[int, int, int]] = []
    cursor = start_epoch
    for e in unique_epochs:
        if e > cursor:
            out.append((cursor, e - bucket_sec, ((e - cursor) // bucket_sec) + 1))
        cursor = e + bucket_sec
    if cursor <= end_epoch:
        out.append((cursor, end_epoch, ((end_epoch - cursor) // bucket_sec) + 1))
    return out


def _coverage_block(actual: int, expected: int) -> Dict[str, Any]:
    return {
        "expected_buckets": int(expected),
        "actual_buckets": int(actual),
        "coverage_ratio": round(float(actual / max(1, expected)), 8),
    }


def _latency_quantiles(
    cur,
    *,
    table_name: str,
    ts_col: str,
    start_dt: datetime,
    end_dt: datetime,
    available_col: Optional[str],
    occurred_col: Optional[str],
    extra_filter_sql: str = "",
    extra_params: Sequence[Any] = (),
) -> Dict[str, Optional[float]]:
    if not available_col and not occurred_col:
        return {"p50_sec": 0.0, "p90_sec": 0.0, "p99_sec": 0.0}
    a_expr = available_col or ts_col
    o_expr = occurred_col or ts_col
    where = [f"{ts_col} >= %s", f"{ts_col} <= %s"]
    params: List[Any] = [start_dt, end_dt]
    if extra_filter_sql.strip():
        where.append(f"({extra_filter_sql})")
        params.extend(list(extra_params))
    sql = f"""
    WITH lag AS (
      SELECT GREATEST(0.0, EXTRACT(EPOCH FROM (COALESCE({a_expr}, {ts_col}) - COALESCE({o_expr}, {ts_col}))))::double precision AS lag_sec
      FROM {table_name}
      WHERE {' AND '.join(where)}
    )
    SELECT
      percentile_cont(0.5) WITHIN GROUP (ORDER BY lag_sec) AS p50,
      percentile_cont(0.9) WITHIN GROUP (ORDER BY lag_sec) AS p90,
      percentile_cont(0.99) WITHIN GROUP (ORDER BY lag_sec) AS p99
    FROM lag
    """
    cur.execute(sql, tuple(params))
    row = cur.fetchone() or {}
    return {
        "p50_sec": round(float(row.get("p50")), 6) if row.get("p50") is not None else None,
        "p90_sec": round(float(row.get("p90")), 6) if row.get("p90") is not None else None,
        "p99_sec": round(float(row.get("p99")), 6) if row.get("p99") is not None else None,
    }


def _social_sources(cur) -> Dict[str, Any]:
    posts_raw_exists = _table_exists(cur, "social_posts_raw")
    comments_raw_exists = _table_exists(cur, "social_comments_raw")
    events_exists = _table_exists(cur, "events")
    return {
        "posts_table": "social_posts_raw" if posts_raw_exists else ("events" if events_exists else None),
        "comments_table": "social_comments_raw" if comments_raw_exists else ("events" if events_exists else None),
        "events_filter": "COALESCE(payload->>'social_platform','none') NOT IN ('none','unknown','')",
    }


def _social_counts(
    cur,
    *,
    source: Dict[str, Any],
    start_dt: datetime,
    end_dt: datetime,
    bucket_sec: int,
) -> Dict[str, Any]:
    posts_table = source.get("posts_table")
    comments_table = source.get("comments_table")
    events_filter = str(source.get("events_filter") or "")

    posts_total = 0
    post_bucket_map: Dict[int, int] = {}
    daily_posts: Dict[str, int] = {}

    if posts_table == "social_posts_raw":
        p_cols = _table_columns(cur, "social_posts_raw")
        p_ts_col = _pick_col(p_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
        if p_ts_col:
            cur.execute(
                f"""
                SELECT COUNT(*)::bigint AS c
                FROM social_posts_raw
                WHERE {p_ts_col} >= %s AND {p_ts_col} <= %s
                """,
                (start_dt, end_dt),
            )
            posts_total = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute(
                f"""
                SELECT FLOOR(EXTRACT(EPOCH FROM {p_ts_col}) / %s)::bigint AS b, COUNT(*)::bigint AS c
                FROM social_posts_raw
                WHERE {p_ts_col} >= %s AND {p_ts_col} <= %s
                GROUP BY 1
                """,
                (bucket_sec, start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                post_bucket_map[int(r.get("b"))] = int(r.get("c") or 0)
            cur.execute(
                f"""
                SELECT DATE_TRUNC('day', {p_ts_col})::date AS d, COUNT(*)::bigint AS c
                FROM social_posts_raw
                WHERE {p_ts_col} >= %s AND {p_ts_col} <= %s
                GROUP BY 1
                """,
                (start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                daily_posts[str(r.get("d"))] = int(r.get("c") or 0)

    elif posts_table == "events":
        e_cols = _table_columns(cur, "events")
        e_ts_col = _pick_col(e_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
        if e_ts_col:
            cur.execute(
                f"""
                SELECT COUNT(*)::bigint AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                """,
                (start_dt, end_dt),
            )
            posts_total = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute(
                f"""
                SELECT FLOOR(EXTRACT(EPOCH FROM {e_ts_col}) / %s)::bigint AS b, COUNT(*)::bigint AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                GROUP BY 1
                """,
                (bucket_sec, start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                post_bucket_map[int(r.get("b"))] = int(r.get("c") or 0)
            cur.execute(
                f"""
                SELECT DATE_TRUNC('day', {e_ts_col})::date AS d, COUNT(*)::bigint AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                GROUP BY 1
                """,
                (start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                daily_posts[str(r.get("d"))] = int(r.get("c") or 0)

    comments_total = 0
    comment_bucket_map: Dict[int, int] = {}
    daily_comments: Dict[str, int] = {}

    if comments_table == "social_comments_raw":
        c_cols = _table_columns(cur, "social_comments_raw")
        c_ts_col = _pick_col(c_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
        if c_ts_col:
            cur.execute(
                f"""
                SELECT COUNT(*)::bigint AS c
                FROM social_comments_raw
                WHERE {c_ts_col} >= %s AND {c_ts_col} <= %s
                """,
                (start_dt, end_dt),
            )
            comments_total = int((cur.fetchone() or {}).get("c") or 0)
            cur.execute(
                f"""
                SELECT FLOOR(EXTRACT(EPOCH FROM {c_ts_col}) / %s)::bigint AS b, COUNT(*)::bigint AS c
                FROM social_comments_raw
                WHERE {c_ts_col} >= %s AND {c_ts_col} <= %s
                GROUP BY 1
                """,
                (bucket_sec, start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                comment_bucket_map[int(r.get("b"))] = int(r.get("c") or 0)
            cur.execute(
                f"""
                SELECT DATE_TRUNC('day', {c_ts_col})::date AS d, COUNT(*)::bigint AS c
                FROM social_comments_raw
                WHERE {c_ts_col} >= %s AND {c_ts_col} <= %s
                GROUP BY 1
                """,
                (start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                daily_comments[str(r.get("d"))] = int(r.get("c") or 0)

    elif comments_table == "events":
        e_cols = _table_columns(cur, "events")
        e_ts_col = _pick_col(e_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
        if e_ts_col:
            real_comment_expr = (
                "GREATEST(0.0, "
                "COALESCE((payload->>'n_comments')::double precision,0) + "
                "COALESCE((payload->>'n_replies')::double precision,0) - "
                "COALESCE((payload->>'comment_backfill_added')::double precision,0)"
                ")"
            )
            cur.execute(
                f"""
                SELECT COALESCE(SUM({real_comment_expr}),0)::double precision AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                """,
                (start_dt, end_dt),
            )
            comments_total = int(float((cur.fetchone() or {}).get("c") or 0.0))
            cur.execute(
                f"""
                SELECT FLOOR(EXTRACT(EPOCH FROM {e_ts_col}) / %s)::bigint AS b,
                       COALESCE(SUM({real_comment_expr}),0)::double precision AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                GROUP BY 1
                """,
                (bucket_sec, start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                comment_bucket_map[int(r.get("b"))] = int(float(r.get("c") or 0.0))
            cur.execute(
                f"""
                SELECT DATE_TRUNC('day', {e_ts_col})::date AS d,
                       COALESCE(SUM({real_comment_expr}),0)::double precision AS c
                FROM events
                WHERE {events_filter}
                  AND {e_ts_col} >= %s AND {e_ts_col} <= %s
                GROUP BY 1
                """,
                (start_dt, end_dt),
            )
            for r in cur.fetchall() or []:
                daily_comments[str(r.get("d"))] = int(float(r.get("c") or 0.0))

    full_ratio = float(comments_total / max(1, posts_total))

    days = sorted(set(daily_posts.keys()) | set(daily_comments.keys()))
    daily_ratio: List[Dict[str, Any]] = []
    for day in days:
        p = int(daily_posts.get(day, 0))
        c = int(daily_comments.get(day, 0))
        ratio = float(c / p) if p > 0 else None
        daily_ratio.append({"day": day, "posts": p, "comments": c, "ratio": round(ratio, 6) if ratio is not None else None})

    buckets = sorted(set(post_bucket_map.keys()) | set(comment_bucket_map.keys()))
    bucket_ratio_all: List[Dict[str, Any]] = []
    bucket_ratio_abnormal: List[Dict[str, Any]] = []
    for b in buckets:
        p = int(post_bucket_map.get(b, 0))
        c = int(comment_bucket_map.get(b, 0))
        ratio = float(c / p) if p > 0 else None
        row = {
            "bucket_ts": _to_iso_z(datetime.fromtimestamp(b, tz=timezone.utc)),
            "posts": p,
            "comments": c,
            "ratio": round(ratio, 6) if ratio is not None else None,
        }
        bucket_ratio_all.append(row)
        if p > 0 and (ratio is None or ratio < TARGET_RATIO):
            bucket_ratio_abnormal.append(row)

    sample_size = 800
    if len(bucket_ratio_all) <= sample_size:
        bucket_ratio_sample = bucket_ratio_all
    else:
        step = max(1, len(bucket_ratio_all) // sample_size)
        bucket_ratio_sample = bucket_ratio_all[::step][:sample_size]

    bucket_pass = len(bucket_ratio_abnormal) == 0

    return {
        "posts_total": int(posts_total),
        "comments_total": int(comments_total),
        "full_window_ratio": round(full_ratio, 6),
        "daily_ratio": daily_ratio,
        "bucket_ratio": {
            "sample": bucket_ratio_sample,
            "abnormal_buckets_full": bucket_ratio_abnormal,
            "bucket_pass": bool(bucket_pass),
        },
        "sources": {
            "posts_table": posts_table,
            "comments_table": comments_table,
        },
        "post_bucket_map": post_bucket_map,
        "comment_bucket_map": comment_bucket_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit full-history completeness from 2018-01-01 to now")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default=FIXED_START_ISO)
    parser.add_argument("--end", default="")
    parser.add_argument("--timeframe", default=FIXED_TIMEFRAME)
    parser.add_argument("--symbols", default="BTC,ETH,SOL")
    parser.add_argument("--out-json", default="artifacts/audit/full_history_latest.json")
    args = parser.parse_args()

    if psycopg2 is None or RealDictCursor is None:
        raise SystemExit("missing psycopg2 dependency; run: PROFILE=runtime bash scripts/bootstrap_env.sh")

    start_dt = _parse_iso_utc(args.start)
    if start_dt != FIXED_START_DT:
        raise SystemExit("start_must_be_2018-01-01T00:00:00Z")
    end_dt = _parse_iso_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    if end_dt <= start_dt:
        raise SystemExit("invalid_time_window")

    bucket_sec = _bucket_seconds(args.timeframe)
    expected_buckets = _expected_bucket_count(start_dt, end_dt, bucket_sec)
    start_epoch = _bucket_epoch(start_dt, bucket_sec)
    end_epoch = _bucket_epoch(end_dt, bucket_sec)
    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("empty_symbols")

    out: Dict[str, Any] = {
        "window": {
            "start": _to_iso_z(start_dt),
            "end": _to_iso_z(end_dt),
            "timeframe": FIXED_TIMEFRAME,
            "bucket_seconds": bucket_sec,
            "expected_buckets_per_symbol": expected_buckets,
        },
        "symbols": symbols,
        "per_symbol": {},
        "modality_coverage": {},
        "modality_latency_quantiles": {},
        "comment_post_ratio": {},
        "summary": {},
    }

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            table_flags = {
                "market_bars": _table_exists(cur, "market_bars"),
                "orderbook_l2": _table_exists(cur, "orderbook_l2"),
                "funding_rates": _table_exists(cur, "funding_rates"),
                "onchain_signals": _table_exists(cur, "onchain_signals"),
            }

            market_full_ok = 0

            social_source = _social_sources(cur)
            social_stats = _social_counts(
                cur,
                source=social_source,
                start_dt=start_dt,
                end_dt=end_dt,
                bucket_sec=bucket_sec,
            )

            social_post_epochs = sorted(set(int(x) for x in (social_stats.get("post_bucket_map") or {}).keys()))
            social_comment_epochs = sorted(set(int(x) for x in (social_stats.get("comment_bucket_map") or {}).keys()))

            social_posts_cov = _coverage_block(len(social_post_epochs), expected_buckets)
            social_comments_cov = _coverage_block(len(social_comment_epochs), expected_buckets)

            for sym in symbols:
                sym_block: Dict[str, Any] = {}

                if table_flags["market_bars"]:
                    m_cols = _table_columns(cur, "market_bars")
                    m_ts = _pick_col(m_cols, ("ts", "timestamp", "bar_ts"))
                    m_symbol = _pick_col(m_cols, ("symbol", "asset_symbol"))
                    m_tf = _pick_col(m_cols, ("timeframe",))
                    if m_ts and m_symbol:
                        m_epochs = _fetch_bucket_epochs(
                            cur,
                            table_name="market_bars",
                            ts_col=m_ts,
                            start_dt=start_dt,
                            end_dt=end_dt,
                            bucket_sec=bucket_sec,
                            symbol_col=m_symbol,
                            symbol=sym,
                            timeframe_col=m_tf,
                            timeframe=FIXED_TIMEFRAME,
                        )
                    else:
                        m_epochs = []
                else:
                    m_epochs = []

                m_actual = len(sorted(set(m_epochs)))
                m_cov = _coverage_block(m_actual, expected_buckets)
                gaps = _gap_ranges_from_epochs(
                    epochs=m_epochs,
                    start_epoch=start_epoch,
                    end_epoch=end_epoch,
                    bucket_sec=bucket_sec,
                )
                gap_ranges = [
                    {
                        "start": _to_iso_z(datetime.fromtimestamp(g0, tz=timezone.utc)),
                        "end": _to_iso_z(datetime.fromtimestamp(g1, tz=timezone.utc)),
                        "bucket_count": int(gc),
                    }
                    for g0, g1, gc in gaps
                ]
                sym_block["market_bars"] = {
                    **m_cov,
                    "gap_bucket_total": int(sum(int(x[2]) for x in gaps)),
                    "gap_bucket_ranges": gap_ranges,
                }
                if m_actual == expected_buckets:
                    market_full_ok += 1

                if table_flags["orderbook_l2"]:
                    ob_cols = _table_columns(cur, "orderbook_l2")
                    ob_ts = _pick_col(ob_cols, ("ts", "timestamp"))
                    ob_symbol = _pick_col(ob_cols, ("symbol", "asset_symbol"))
                    ob_epochs = (
                        _fetch_bucket_epochs(
                            cur,
                            table_name="orderbook_l2",
                            ts_col=ob_ts,
                            start_dt=start_dt,
                            end_dt=end_dt,
                            bucket_sec=bucket_sec,
                            symbol_col=ob_symbol,
                            symbol=sym,
                        )
                        if ob_ts and ob_symbol
                        else []
                    )
                else:
                    ob_epochs = []
                sym_block["orderbook_l2"] = _coverage_block(len(sorted(set(ob_epochs))), expected_buckets)

                if table_flags["funding_rates"]:
                    fr_cols = _table_columns(cur, "funding_rates")
                    fr_ts = _pick_col(fr_cols, ("ts", "timestamp"))
                    fr_symbol = _pick_col(fr_cols, ("symbol", "asset_symbol"))
                    fr_epochs = (
                        _fetch_bucket_epochs(
                            cur,
                            table_name="funding_rates",
                            ts_col=fr_ts,
                            start_dt=start_dt,
                            end_dt=end_dt,
                            bucket_sec=bucket_sec,
                            symbol_col=fr_symbol,
                            symbol=sym,
                        )
                        if fr_ts and fr_symbol
                        else []
                    )
                else:
                    fr_epochs = []
                sym_block["funding_rates"] = _coverage_block(len(sorted(set(fr_epochs))), expected_buckets)

                if table_flags["onchain_signals"]:
                    oc_cols = _table_columns(cur, "onchain_signals")
                    oc_ts = _pick_col(oc_cols, ("ts", "timestamp"))
                    oc_symbol = _pick_col(oc_cols, ("asset_symbol", "symbol"))
                    oc_epochs = (
                        _fetch_bucket_epochs(
                            cur,
                            table_name="onchain_signals",
                            ts_col=oc_ts,
                            start_dt=start_dt,
                            end_dt=end_dt,
                            bucket_sec=bucket_sec,
                            symbol_col=oc_symbol,
                            symbol=sym,
                        )
                        if oc_ts and oc_symbol
                        else []
                    )
                else:
                    oc_epochs = []
                sym_block["onchain_signals"] = _coverage_block(len(sorted(set(oc_epochs))), expected_buckets)

                sym_block["social_posts"] = dict(social_posts_cov)
                sym_block["social_comments"] = dict(social_comments_cov)

                out["per_symbol"][sym] = {
                    "expected_buckets": expected_buckets,
                    "actual_buckets": int(m_actual),
                    "gap_buckets": gap_ranges,
                    "modalities": sym_block,
                }

            n_sym = len(symbols)
            def _avg_cov(modality: str) -> Dict[str, Any]:
                actual_sum = 0
                expected_sum = expected_buckets * n_sym
                for s in symbols:
                    actual_sum += int((out["per_symbol"][s]["modalities"][modality] or {}).get("actual_buckets") or 0)
                return {
                    "actual_buckets": int(actual_sum),
                    "expected_buckets": int(expected_sum),
                    "coverage_ratio": round(float(actual_sum / max(1, expected_sum)), 8),
                }

            out["modality_coverage"] = {
                "market_bars": _avg_cov("market_bars"),
                "orderbook_l2": _avg_cov("orderbook_l2"),
                "funding_rates": _avg_cov("funding_rates"),
                "onchain_signals": _avg_cov("onchain_signals"),
                "social_posts": dict(social_posts_cov),
                "social_comments": dict(social_comments_cov),
            }

            lat_out: Dict[str, Dict[str, Optional[float]]] = {}
            if table_flags["market_bars"]:
                m_cols = _table_columns(cur, "market_bars")
                m_ts = _pick_col(m_cols, ("ts", "timestamp", "bar_ts"))
                if m_ts:
                    lat_out["market_bars"] = _latency_quantiles(
                        cur,
                        table_name="market_bars",
                        ts_col=m_ts,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        available_col=_pick_col(m_cols, ("available_at", "feature_available_at")),
                        occurred_col=_pick_col(m_cols, ("occurred_at", "published_at")),
                    )
            if table_flags["orderbook_l2"]:
                ob_cols = _table_columns(cur, "orderbook_l2")
                ob_ts = _pick_col(ob_cols, ("ts", "timestamp"))
                if ob_ts:
                    lat_out["orderbook_l2"] = _latency_quantiles(
                        cur,
                        table_name="orderbook_l2",
                        ts_col=ob_ts,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        available_col=_pick_col(ob_cols, ("available_at",)),
                        occurred_col=_pick_col(ob_cols, ("occurred_at", "published_at")),
                    )
            if table_flags["funding_rates"]:
                fr_cols = _table_columns(cur, "funding_rates")
                fr_ts = _pick_col(fr_cols, ("ts", "timestamp"))
                if fr_ts:
                    lat_out["funding_rates"] = _latency_quantiles(
                        cur,
                        table_name="funding_rates",
                        ts_col=fr_ts,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        available_col=_pick_col(fr_cols, ("available_at",)),
                        occurred_col=_pick_col(fr_cols, ("occurred_at", "published_at")),
                    )
            if table_flags["onchain_signals"]:
                oc_cols = _table_columns(cur, "onchain_signals")
                oc_ts = _pick_col(oc_cols, ("ts", "timestamp"))
                if oc_ts:
                    lat_out["onchain_signals"] = _latency_quantiles(
                        cur,
                        table_name="onchain_signals",
                        ts_col=oc_ts,
                        start_dt=start_dt,
                        end_dt=end_dt,
                        available_col=_pick_col(oc_cols, ("available_at",)),
                        occurred_col=_pick_col(oc_cols, ("occurred_at", "published_at")),
                    )

            posts_table = social_stats.get("sources", {}).get("posts_table")
            comments_table = social_stats.get("sources", {}).get("comments_table")

            if posts_table == "social_posts_raw":
                p_cols = _table_columns(cur, "social_posts_raw")
                p_ts = _pick_col(p_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
                lat_out["social_posts"] = _latency_quantiles(
                    cur,
                    table_name="social_posts_raw",
                    ts_col=p_ts or "created_at",
                    start_dt=start_dt,
                    end_dt=end_dt,
                    available_col=_pick_col(p_cols, ("available_at",)),
                    occurred_col=_pick_col(p_cols, ("occurred_at", "published_at", "created_at")),
                ) if p_ts else {"p50_sec": None, "p90_sec": None, "p99_sec": None}
            elif posts_table == "events":
                e_cols = _table_columns(cur, "events")
                e_ts = _pick_col(e_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
                lat_out["social_posts"] = _latency_quantiles(
                    cur,
                    table_name="events",
                    ts_col=e_ts or "created_at",
                    start_dt=start_dt,
                    end_dt=end_dt,
                    available_col=_pick_col(e_cols, ("available_at",)),
                    occurred_col=_pick_col(e_cols, ("occurred_at", "published_at", "created_at")),
                    extra_filter_sql=social_source["events_filter"],
                ) if e_ts else {"p50_sec": None, "p90_sec": None, "p99_sec": None}
            else:
                lat_out["social_posts"] = {"p50_sec": None, "p90_sec": None, "p99_sec": None}

            if comments_table == "social_comments_raw":
                c_cols = _table_columns(cur, "social_comments_raw")
                c_ts = _pick_col(c_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
                lat_out["social_comments"] = _latency_quantiles(
                    cur,
                    table_name="social_comments_raw",
                    ts_col=c_ts or "created_at",
                    start_dt=start_dt,
                    end_dt=end_dt,
                    available_col=_pick_col(c_cols, ("available_at",)),
                    occurred_col=_pick_col(c_cols, ("occurred_at", "published_at", "created_at")),
                ) if c_ts else {"p50_sec": None, "p90_sec": None, "p99_sec": None}
            elif comments_table == "events":
                e_cols = _table_columns(cur, "events")
                e_ts = _pick_col(e_cols, ("available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"))
                lat_out["social_comments"] = _latency_quantiles(
                    cur,
                    table_name="events",
                    ts_col=e_ts or "created_at",
                    start_dt=start_dt,
                    end_dt=end_dt,
                    available_col=_pick_col(e_cols, ("available_at",)),
                    occurred_col=_pick_col(e_cols, ("occurred_at", "published_at", "created_at")),
                    extra_filter_sql=social_source["events_filter"],
                ) if e_ts else {"p50_sec": None, "p90_sec": None, "p99_sec": None}
            else:
                lat_out["social_comments"] = {"p50_sec": None, "p90_sec": None, "p99_sec": None}

            out["modality_latency_quantiles"] = lat_out

            comment_ratio = {
                "full_window_ratio": float(social_stats.get("full_window_ratio") or 0.0),
                "daily_ratio": social_stats.get("daily_ratio") or [],
                "bucket_ratio": social_stats.get("bucket_ratio") or {},
            }
            out["comment_post_ratio"] = comment_ratio

            full_ratio_ok = float(comment_ratio.get("full_window_ratio") or 0.0) >= TARGET_RATIO
            bucket_ok = bool((comment_ratio.get("bucket_ratio") or {}).get("bucket_pass"))
            comment_ratio_ge_10x = bool(full_ratio_ok and bucket_ok)
            modality_window_complete = {}
            for modality in (
                "market_bars",
                "orderbook_l2",
                "funding_rates",
                "onchain_signals",
                "social_posts",
                "social_comments",
            ):
                cov = out["modality_coverage"].get(modality) if isinstance(out.get("modality_coverage"), dict) else {}
                expected = int((cov or {}).get("expected_buckets") or 0)
                actual = int((cov or {}).get("actual_buckets") or 0)
                modality_window_complete[modality] = bool(expected > 0 and actual >= expected)
            history_window_complete = bool(all(bool(v) for v in modality_window_complete.values()))

            out["summary"] = {
                "history_window_complete": history_window_complete,
                "comment_ratio_ge_10x": comment_ratio_ge_10x,
                "modality_window_complete": modality_window_complete,
                "market_symbols_full": int(market_full_ok),
                "market_symbols_total": int(len(symbols)),
                "full_window_ratio": round(float(comment_ratio.get("full_window_ratio") or 0.0), 6),
                "bucket_ratio_pass": bool(bucket_ok),
                "audit_ready_for_next_phase": bool(history_window_complete and comment_ratio_ge_10x),
            }

    out_text = json.dumps(out, ensure_ascii=False)
    out_path = str(args.out_json).strip()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(out_text + "\n")
    print(out_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
