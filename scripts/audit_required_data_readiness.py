#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
try:
    from inference.liquid_feature_contract import DERIVATIVE_METRIC_NAMES as CONTRACT_DERIVATIVE_METRICS
    from inference.liquid_feature_contract import DEFAULT_ONCHAIN_PRIMARY_METRIC as CONTRACT_ONCHAIN_PRIMARY_METRIC
except Exception:
    CONTRACT_DERIVATIVE_METRICS = [
        "long_short_ratio_global_accounts",
        "long_short_ratio_top_accounts",
        "long_short_ratio_top_positions",
        "taker_buy_sell_ratio",
        "basis_rate",
        "annualized_basis_rate",
    ]
    CONTRACT_ONCHAIN_PRIMARY_METRIC = "net_inflow"


def _parse_dt_utc(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty_datetime")
    norm = text.replace(" ", "T")
    if norm.endswith("Z"):
        norm = norm[:-1] + "+00:00"
    dt = datetime.fromisoformat(norm)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for piece in str(raw or "").split(","):
        sym = piece.strip().upper().replace("$", "")
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def _parse_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for piece in str(raw or "").split(","):
        cur = piece.strip()
        if cur and cur not in seen:
            seen.add(cur)
            out.append(cur)
    return out


def _timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe or "").strip().lower()
    if tf.endswith("m"):
        return max(1, int(tf[:-1] or "1")) * 60
    if tf.endswith("h"):
        return max(1, int(tf[:-1] or "1")) * 3600
    if tf.endswith("d"):
        return max(1, int(tf[:-1] or "1")) * 86400
    raise ValueError(f"unsupported_timeframe:{timeframe}")


def _expected_bucket_count(start_dt: datetime, end_dt: datetime, bucket_sec: int) -> int:
    if end_dt < start_dt:
        return 0
    return int((end_dt - start_dt).total_seconds() // bucket_sec) + 1


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


def _pick_col(columns: Iterable[str], candidates: List[str]) -> Optional[str]:
    cols = set(columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def _distinct_bucket_count(
    cur,
    *,
    table_name: str,
    ts_col: str,
    bucket_sec: int,
    start_dt: datetime,
    end_dt: datetime,
    symbol_col: Optional[str] = None,
    symbol: Optional[str] = None,
    timeframe_col: Optional[str] = None,
    timeframe: Optional[str] = None,
    metric_filter: Optional[str] = None,
) -> int:
    where = [f"{ts_col} >= %s", f"{ts_col} <= %s"]
    params: List[Any] = [start_dt, end_dt]
    if symbol_col and symbol:
        where.append(f"UPPER({symbol_col}) = %s")
        params.append(str(symbol).upper())
    if timeframe_col and timeframe:
        where.append(f"{timeframe_col} = %s")
        params.append(str(timeframe))
    if metric_filter:
        where.append("metric_name = %s")
        params.append(str(metric_filter))
    cur.execute(
        f"""
        SELECT COUNT(DISTINCT FLOOR(EXTRACT(EPOCH FROM {ts_col}) / %s)::bigint) AS c
        FROM {table_name}
        WHERE {' AND '.join(where)}
        """,
        tuple([bucket_sec] + params),
    )
    row = cur.fetchone() or {}
    return int(row.get("c") or 0)


def _coverage(actual: int, expected: int) -> float:
    return float(actual / max(1, expected))


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        return float(raw)
    except Exception:
        return float(default)


def _resolve_mode_windows(
    *,
    mode: str,
    start_dt: datetime,
    end_dt: datetime,
    lookback_days: int,
    recent_derivatives_days: int,
) -> tuple[str, datetime, datetime]:
    mode_norm = str(mode or "production").strip().lower()
    if mode_norm not in {"production", "research"}:
        mode_norm = "production"
    if mode_norm == "research":
        lookback_start = start_dt
    else:
        lookback_start = max(start_dt, end_dt - timedelta(days=max(1, int(lookback_days))))
    deriv_start = max(start_dt, end_dt - timedelta(days=max(1, int(recent_derivatives_days))))
    return mode_norm, lookback_start, deriv_start


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit required data readiness for training/deployment")
    ap.add_argument("--mode", default=os.getenv("DATA_READINESS_MODE", "production"), choices=["production", "research"])
    ap.add_argument("--database-url", default=DATABASE_URL)
    ap.add_argument("--symbols", default="BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK")
    ap.add_argument("--primary-timeframe", default="5m")
    ap.add_argument("--secondary-timeframe", default="1h")
    ap.add_argument("--lookback-days", type=int, default=420)
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--recent-derivatives-days", type=int, default=30)
    ap.add_argument("--onchain-primary-metric", default=str(CONTRACT_ONCHAIN_PRIMARY_METRIC))
    ap.add_argument(
        "--required-derivative-metrics",
        default=",".join(CONTRACT_DERIVATIVE_METRICS),
    )
    ap.add_argument("--min-primary-coverage", type=float, default=0.9)
    ap.add_argument("--min-secondary-coverage", type=float, default=0.85)
    ap.add_argument("--min-orderbook-coverage", type=float, default=0.7)
    ap.add_argument("--min-funding-coverage", type=float, default=0.7)
    ap.add_argument("--min-onchain-coverage", type=float, default=0.7)
    ap.add_argument("--min-derivatives-coverage", type=float, default=0.4)
    ap.add_argument("--min-linked-events-per-symbol", type=int, default=20)
    ap.add_argument("--min-social-posts", type=int, default=200)
    ap.add_argument("--min-comment-post-ratio", type=float, default=2.0)
    ap.add_argument("--out-json", default="artifacts/audit/required_data_readiness_latest.json")
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise RuntimeError("empty_symbols")

    end_dt = _parse_dt_utc(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    start_dt = _parse_dt_utc(args.start)
    if end_dt <= start_dt:
        raise RuntimeError("invalid_time_range")
    mode_name, lookback_start, deriv_start = _resolve_mode_windows(
        mode=str(args.mode),
        start_dt=start_dt,
        end_dt=end_dt,
        lookback_days=int(args.lookback_days),
        recent_derivatives_days=int(args.recent_derivatives_days),
    )

    primary_tf = str(args.primary_timeframe).strip().lower()
    secondary_tf = str(args.secondary_timeframe).strip().lower()
    primary_bucket_sec = _timeframe_seconds(primary_tf)
    secondary_bucket_sec = _timeframe_seconds(secondary_tf)
    expected_primary = _expected_bucket_count(lookback_start, end_dt, primary_bucket_sec)
    expected_secondary = _expected_bucket_count(lookback_start, end_dt, secondary_bucket_sec)
    expected_derivatives = _expected_bucket_count(deriv_start, end_dt, primary_bucket_sec)

    out: Dict[str, Any] = {
        "status": "ok",
        "generated_at": _to_iso_z(datetime.now(timezone.utc)),
        "window": {
            "mode": mode_name,
            "start": _to_iso_z(start_dt),
            "end": _to_iso_z(end_dt),
            "lookback_start": _to_iso_z(lookback_start),
            "derivatives_start": _to_iso_z(deriv_start),
            "lookback_days": int(args.lookback_days),
            "recent_derivatives_days": int(args.recent_derivatives_days),
        },
        "symbols": symbols,
        "timeframes": {
            "primary": primary_tf,
            "secondary": secondary_tf,
        },
        "expected_buckets": {
            "primary": int(expected_primary),
            "secondary": int(expected_secondary),
            "derivatives_recent": int(expected_derivatives),
        },
        "tables": {},
        "coverage": {
            "per_symbol": {},
            "avg": {},
        },
        "events_social": {},
        "derivatives_metrics": {},
        "thresholds": {
            "min_primary_coverage": float(args.min_primary_coverage),
            "min_secondary_coverage": float(args.min_secondary_coverage),
            "min_orderbook_coverage": float(args.min_orderbook_coverage),
            "min_funding_coverage": float(args.min_funding_coverage),
            "min_onchain_coverage": float(args.min_onchain_coverage),
            "min_derivatives_coverage": float(args.min_derivatives_coverage),
            "min_linked_events_per_symbol": int(args.min_linked_events_per_symbol),
            "min_social_posts": int(args.min_social_posts),
            "min_comment_post_ratio": float(args.min_comment_post_ratio),
        },
        "gates": {},
        "missing_data_kinds": [],
        "recommended_commands": [],
    }

    req_deriv_metrics = _parse_list(args.required_derivative_metrics)
    onchain_primary_metric = str(args.onchain_primary_metric).strip()

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            tables = {
                "market_bars": _table_exists(cur, "market_bars"),
                "orderbook_l2": _table_exists(cur, "orderbook_l2"),
                "funding_rates": _table_exists(cur, "funding_rates"),
                "onchain_signals": _table_exists(cur, "onchain_signals"),
                "events": _table_exists(cur, "events"),
                "event_links": _table_exists(cur, "event_links"),
                "entities": _table_exists(cur, "entities"),
                "social_posts_raw": _table_exists(cur, "social_posts_raw"),
                "social_comments_raw": _table_exists(cur, "social_comments_raw"),
            }
            out["tables"] = tables

            market_cols = _table_columns(cur, "market_bars") if tables["market_bars"] else set()
            orderbook_cols = _table_columns(cur, "orderbook_l2") if tables["orderbook_l2"] else set()
            funding_cols = _table_columns(cur, "funding_rates") if tables["funding_rates"] else set()
            onchain_cols = _table_columns(cur, "onchain_signals") if tables["onchain_signals"] else set()

            market_ts = _pick_col(market_cols, ["ts", "timestamp", "bar_ts"])
            market_symbol = _pick_col(market_cols, ["symbol", "asset_symbol"])
            market_tf = _pick_col(market_cols, ["timeframe"])
            orderbook_ts = _pick_col(orderbook_cols, ["ts", "timestamp"])
            orderbook_symbol = _pick_col(orderbook_cols, ["symbol", "asset_symbol"])
            funding_ts = _pick_col(funding_cols, ["ts", "timestamp"])
            funding_symbol = _pick_col(funding_cols, ["symbol", "asset_symbol"])
            onchain_ts = _pick_col(onchain_cols, ["ts", "timestamp"])
            onchain_symbol = _pick_col(onchain_cols, ["asset_symbol", "symbol"])

            sum_cov = {
                "market_primary": 0.0,
                "market_secondary": 0.0,
                "orderbook": 0.0,
                "funding": 0.0,
                "onchain_primary_metric": 0.0,
            }

            for sym in symbols:
                block: Dict[str, Any] = {}

                primary_actual = (
                    _distinct_bucket_count(
                        cur,
                        table_name="market_bars",
                        ts_col=str(market_ts),
                        bucket_sec=primary_bucket_sec,
                        start_dt=lookback_start,
                        end_dt=end_dt,
                        symbol_col=str(market_symbol),
                        symbol=sym,
                        timeframe_col=str(market_tf) if market_tf else None,
                        timeframe=primary_tf,
                    )
                    if tables["market_bars"] and market_ts and market_symbol
                    else 0
                )
                primary_cov = _coverage(primary_actual, expected_primary)
                block["market_primary"] = {
                    "actual_buckets": int(primary_actual),
                    "expected_buckets": int(expected_primary),
                    "coverage_ratio": round(primary_cov, 8),
                }
                sum_cov["market_primary"] += primary_cov

                secondary_actual = (
                    _distinct_bucket_count(
                        cur,
                        table_name="market_bars",
                        ts_col=str(market_ts),
                        bucket_sec=secondary_bucket_sec,
                        start_dt=lookback_start,
                        end_dt=end_dt,
                        symbol_col=str(market_symbol),
                        symbol=sym,
                        timeframe_col=str(market_tf) if market_tf else None,
                        timeframe=secondary_tf,
                    )
                    if tables["market_bars"] and market_ts and market_symbol
                    else 0
                )
                secondary_cov = _coverage(secondary_actual, expected_secondary)
                block["market_secondary"] = {
                    "actual_buckets": int(secondary_actual),
                    "expected_buckets": int(expected_secondary),
                    "coverage_ratio": round(secondary_cov, 8),
                }
                sum_cov["market_secondary"] += secondary_cov

                orderbook_actual = (
                    _distinct_bucket_count(
                        cur,
                        table_name="orderbook_l2",
                        ts_col=str(orderbook_ts),
                        bucket_sec=primary_bucket_sec,
                        start_dt=lookback_start,
                        end_dt=end_dt,
                        symbol_col=str(orderbook_symbol),
                        symbol=sym,
                    )
                    if tables["orderbook_l2"] and orderbook_ts and orderbook_symbol
                    else 0
                )
                orderbook_cov = _coverage(orderbook_actual, expected_primary)
                block["orderbook_l2"] = {
                    "actual_buckets": int(orderbook_actual),
                    "expected_buckets": int(expected_primary),
                    "coverage_ratio": round(orderbook_cov, 8),
                }
                sum_cov["orderbook"] += orderbook_cov

                funding_actual = (
                    _distinct_bucket_count(
                        cur,
                        table_name="funding_rates",
                        ts_col=str(funding_ts),
                        bucket_sec=primary_bucket_sec,
                        start_dt=lookback_start,
                        end_dt=end_dt,
                        symbol_col=str(funding_symbol),
                        symbol=sym,
                    )
                    if tables["funding_rates"] and funding_ts and funding_symbol
                    else 0
                )
                funding_cov = _coverage(funding_actual, expected_primary)
                block["funding_rates"] = {
                    "actual_buckets": int(funding_actual),
                    "expected_buckets": int(expected_primary),
                    "coverage_ratio": round(funding_cov, 8),
                }
                sum_cov["funding"] += funding_cov

                onchain_actual = (
                    _distinct_bucket_count(
                        cur,
                        table_name="onchain_signals",
                        ts_col=str(onchain_ts),
                        bucket_sec=primary_bucket_sec,
                        start_dt=lookback_start,
                        end_dt=end_dt,
                        symbol_col=str(onchain_symbol),
                        symbol=sym,
                        metric_filter=onchain_primary_metric,
                    )
                    if tables["onchain_signals"] and onchain_ts and onchain_symbol and onchain_primary_metric
                    else 0
                )
                onchain_cov = _coverage(onchain_actual, expected_primary)
                block["onchain_primary_metric"] = {
                    "metric_name": onchain_primary_metric,
                    "actual_buckets": int(onchain_actual),
                    "expected_buckets": int(expected_primary),
                    "coverage_ratio": round(onchain_cov, 8),
                }
                sum_cov["onchain_primary_metric"] += onchain_cov

                if tables["events"] and tables["event_links"] and tables["entities"]:
                    cur.execute(
                        """
                        SELECT COUNT(*)::bigint AS c
                        FROM events e
                        JOIN event_links el ON el.event_id = e.id
                        JOIN entities en ON en.id = el.entity_id
                        WHERE UPPER(en.symbol) = %s
                          AND COALESCE(e.available_at, e.occurred_at, e.published_at, e.created_at) >= %s
                          AND COALESCE(e.available_at, e.occurred_at, e.published_at, e.created_at) <= %s
                        """,
                        (sym, lookback_start, end_dt),
                    )
                    linked_event_rows = int((cur.fetchone() or {}).get("c") or 0)
                else:
                    linked_event_rows = 0
                block["linked_event_rows"] = int(linked_event_rows)
                out["coverage"]["per_symbol"][sym] = block

            n_sym = max(1, len(symbols))
            out["coverage"]["avg"] = {
                "market_primary": round(sum_cov["market_primary"] / n_sym, 8),
                "market_secondary": round(sum_cov["market_secondary"] / n_sym, 8),
                "orderbook_l2": round(sum_cov["orderbook"] / n_sym, 8),
                "funding_rates": round(sum_cov["funding"] / n_sym, 8),
                "onchain_primary_metric": round(sum_cov["onchain_primary_metric"] / n_sym, 8),
            }

            posts_total = 0
            comments_total = 0
            if tables["social_posts_raw"]:
                p_cols = _table_columns(cur, "social_posts_raw")
                p_ts = _pick_col(p_cols, ["available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"])
                if p_ts:
                    cur.execute(
                        f"""
                        SELECT COUNT(*)::bigint AS c
                        FROM social_posts_raw
                        WHERE {p_ts} >= %s AND {p_ts} <= %s
                        """,
                        (lookback_start, end_dt),
                    )
                    posts_total = int((cur.fetchone() or {}).get("c") or 0)
            elif tables["events"]:
                e_cols = _table_columns(cur, "events")
                e_ts = _pick_col(e_cols, ["available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"])
                if e_ts:
                    cur.execute(
                        f"""
                        SELECT COUNT(*)::bigint AS c
                        FROM events
                        WHERE COALESCE(payload->>'social_platform','none') NOT IN ('none','unknown','')
                          AND {e_ts} >= %s AND {e_ts} <= %s
                        """,
                        (lookback_start, end_dt),
                    )
                    posts_total = int((cur.fetchone() or {}).get("c") or 0)

            if tables["social_comments_raw"]:
                c_cols = _table_columns(cur, "social_comments_raw")
                c_ts = _pick_col(c_cols, ["available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"])
                if c_ts:
                    cur.execute(
                        f"""
                        SELECT COUNT(*)::bigint AS c
                        FROM social_comments_raw
                        WHERE {c_ts} >= %s AND {c_ts} <= %s
                        """,
                        (lookback_start, end_dt),
                    )
                    comments_total = int((cur.fetchone() or {}).get("c") or 0)
            elif tables["events"]:
                e_cols = _table_columns(cur, "events")
                e_ts = _pick_col(e_cols, ["available_at", "occurred_at", "published_at", "created_at", "ts", "timestamp"])
                if e_ts:
                    cur.execute(
                        f"""
                        SELECT COALESCE(
                            SUM(
                                GREATEST(
                                    0.0,
                                    COALESCE((payload->>'n_comments')::double precision,0)
                                    + COALESCE((payload->>'n_replies')::double precision,0)
                                    - COALESCE((payload->>'comment_backfill_added')::double precision,0)
                                )
                            ),
                            0
                        )::double precision AS c
                        FROM events
                        WHERE COALESCE(payload->>'social_platform','none') NOT IN ('none','unknown','')
                          AND {e_ts} >= %s AND {e_ts} <= %s
                        """,
                        (lookback_start, end_dt),
                    )
                    comments_total = int(_safe_float((cur.fetchone() or {}).get("c"), 0.0))

            linked_events_total = int(
                sum(int((out["coverage"]["per_symbol"][s] or {}).get("linked_event_rows") or 0) for s in symbols)
            )
            linked_events_avg = float(linked_events_total / max(1, len(symbols)))
            comment_post_ratio = float(comments_total / max(1, posts_total))
            out["events_social"] = {
                "lookback_start": _to_iso_z(lookback_start),
                "posts_total": int(posts_total),
                "comments_total": int(comments_total),
                "comment_post_ratio": round(comment_post_ratio, 8),
                "linked_events_total": int(linked_events_total),
                "linked_events_avg_per_symbol": round(linked_events_avg, 8),
            }

            deriv_summary: Dict[str, Any] = {}
            if tables["onchain_signals"] and onchain_ts and onchain_symbol and req_deriv_metrics:
                for metric_name in req_deriv_metrics:
                    per_sym_cov: Dict[str, float] = {}
                    cov_sum = 0.0
                    for sym in symbols:
                        actual = _distinct_bucket_count(
                            cur,
                            table_name="onchain_signals",
                            ts_col=str(onchain_ts),
                            bucket_sec=primary_bucket_sec,
                            start_dt=deriv_start,
                            end_dt=end_dt,
                            symbol_col=str(onchain_symbol),
                            symbol=sym,
                            metric_filter=metric_name,
                        )
                        cov = _coverage(actual, expected_derivatives)
                        per_sym_cov[sym] = round(cov, 8)
                        cov_sum += cov
                    deriv_summary[metric_name] = {
                        "avg_coverage_ratio": round(cov_sum / max(1, len(symbols)), 8),
                        "per_symbol_coverage_ratio": per_sym_cov,
                        "expected_buckets": int(expected_derivatives),
                    }
            out["derivatives_metrics"] = deriv_summary

    avg_cov = out["coverage"]["avg"]
    gates = {
        "market_primary": bool(_safe_float(avg_cov.get("market_primary")) >= float(args.min_primary_coverage)),
        "market_secondary": bool(_safe_float(avg_cov.get("market_secondary")) >= float(args.min_secondary_coverage)),
        "orderbook_l2": bool(_safe_float(avg_cov.get("orderbook_l2")) >= float(args.min_orderbook_coverage)),
        "funding_rates": bool(_safe_float(avg_cov.get("funding_rates")) >= float(args.min_funding_coverage)),
        "onchain_primary_metric": bool(_safe_float(avg_cov.get("onchain_primary_metric")) >= float(args.min_onchain_coverage)),
        "events_linked": bool(
            _safe_float(out["events_social"].get("linked_events_avg_per_symbol"))
            >= float(args.min_linked_events_per_symbol)
        ),
        "social_posts": bool(int(out["events_social"].get("posts_total") or 0) >= int(args.min_social_posts)),
        "comment_post_ratio": bool(
            _safe_float(out["events_social"].get("comment_post_ratio")) >= float(args.min_comment_post_ratio)
        ),
        "derivatives_metrics": True,
    }
    for metric_name in req_deriv_metrics:
        block = out["derivatives_metrics"].get(metric_name) if isinstance(out["derivatives_metrics"], dict) else None
        if not isinstance(block, dict):
            gates["derivatives_metrics"] = False
            continue
        if _safe_float(block.get("avg_coverage_ratio")) < float(args.min_derivatives_coverage):
            gates["derivatives_metrics"] = False

    out["gates"] = {
        **gates,
        "ready": bool(all(bool(v) for v in gates.values())),
    }

    missing: List[str] = []
    if not gates["market_primary"]:
        missing.append("market_bars_primary")
    if not gates["market_secondary"]:
        missing.append("market_bars_secondary")
    if not gates["orderbook_l2"]:
        missing.append("orderbook_l2")
    if not gates["funding_rates"]:
        missing.append("funding_rates")
    if not gates["onchain_primary_metric"]:
        missing.append(f"onchain:{onchain_primary_metric}")
    if not gates["events_linked"]:
        missing.append("linked_events")
    if not gates["social_posts"]:
        missing.append("social_posts")
    if not gates["comment_post_ratio"]:
        missing.append("social_comments_ratio")
    if not gates["derivatives_metrics"]:
        missing.append("derivatives_metrics")
    out["missing_data_kinds"] = missing

    recommended: List[str] = []
    if "market_bars_primary" in missing or "orderbook_l2" in missing or "funding_rates" in missing or f"onchain:{onchain_primary_metric}" in missing:
        recommended.append(
            "bash scripts/remediate_liquid_data_gaps.sh"
        )
    if "market_bars_secondary" in missing:
        recommended.append(
            "python3 scripts/ingest_bitget_market_bars.py --market perp --timeframe 1h --start 2018-01-01T00:00:00Z --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT --symbol-map BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK --allow-partial --resume"
        )
    if "linked_events" in missing or "social_posts" in missing or "social_comments_ratio" in missing:
        recommended.append(
            "python3 scripts/orchestrate_event_social_backfill.py --start 2018-01-01T00:00:00Z --chunk-days 30 --resume --out-events-jsonl artifacts/backfill_2018_now/events_2018_now.jsonl --out-social-jsonl artifacts/backfill_2018_now/social_2018_now.jsonl"
        )
        recommended.append(
            "python3 scripts/import_events_jsonl.py --jsonl artifacts/backfill_2018_now/events_2018_now.jsonl && python3 scripts/import_social_events_jsonl.py --jsonl artifacts/backfill_2018_now/social_2018_now.jsonl"
        )
    if "derivatives_metrics" in missing:
        recommended.append(
            "python3 scripts/ingest_binance_derivatives_signals.py --symbols BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK --start 2018-01-01T00:00:00Z --period 5m"
        )
    out["recommended_commands"] = recommended

    text = json.dumps(out, ensure_ascii=False)
    out_path = str(args.out_json).strip()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
