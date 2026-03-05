#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, SCHEMA_HASH
except Exception:  # pragma: no cover
    FEATURE_DIM = 540
    FEATURE_INDEX = {}
    SCHEMA_HASH = ""


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
DEFAULT_SYMBOLS = os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL")
DEFAULT_TRACK = "liquid"
DEFAULT_BUCKET = "5m"
DEFAULT_TOP_N = int(os.getenv("LIQUID_UNIVERSE_TOP_N", "50"))


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if not text:
        raise ValueError("empty_datetime")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_iso(dt: Optional[datetime]) -> Optional[str]:
    if not isinstance(dt, datetime):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
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


def _bucket_seconds(bucket: str) -> int:
    tf = str(bucket or DEFAULT_BUCKET).strip().lower()
    if tf.endswith("m"):
        return max(1, int(tf[:-1] or "5")) * 60
    if tf.endswith("h"):
        return max(1, int(tf[:-1] or "1")) * 3600
    if tf.endswith("d"):
        return max(1, int(tf[:-1] or "1")) * 86400
    raise ValueError(f"unsupported_bucket:{bucket}")


def _expected_buckets(start_dt: datetime, end_dt: datetime, bucket_sec: int) -> int:
    if end_dt <= start_dt:
        return 0
    return int((end_dt - start_dt).total_seconds() // bucket_sec) + 1


def _coverage(actual: int, expected: int) -> float:
    return float(actual / max(1, expected))


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
        (str(table_name),),
    )
    rows = cur.fetchall() or []
    return {str(r.get("column_name") or "") for r in rows if str(r.get("column_name") or "").strip()}


def _pick_col(cols: set[str], names: Sequence[str]) -> Optional[str]:
    for n in names:
        key = str(n)
        if key in cols:
            return key
    return None


def _parse_horizon_steps(bucket_sec: int) -> Dict[str, int]:
    base = {"1h": 12, "4h": 48, "1d": 288, "7d": 2016}
    if bucket_sec == 300:
        return base
    ratio = max(1, int(round(bucket_sec / 300.0)))
    return {k: max(1, int(round(v / ratio))) for k, v in base.items()}


def _distinct_bucket_count(
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
) -> int:
    where = [f"{ts_col} >= %s", f"{ts_col} <= %s"]
    params: List[Any] = [start_dt, end_dt]
    if symbol_col and symbol:
        where.append(f"UPPER({symbol_col}) = %s")
        params.append(str(symbol).upper())
    if timeframe_col and timeframe:
        where.append(f"{timeframe_col} = %s")
        params.append(str(timeframe))
    cur.execute(
        f"""
        SELECT COUNT(DISTINCT FLOOR(EXTRACT(EPOCH FROM {ts_col}) / %s)::bigint)::bigint AS c
        FROM {table_name}
        WHERE {' AND '.join(where)}
        """,
        tuple([bucket_sec] + params),
    )
    row = dict(cur.fetchone() or {})
    return int(row.get("c") or 0)


def _derive_required_tables_from_code() -> Dict[str, Any]:
    files = [
        ROOT / "training" / "datasets" / "liquid_sequence_dataset.py",
        ROOT / "features" / "sequence.py",
        ROOT / "scripts" / "build_feature_store.py",
        ROOT / "scripts" / "build_text_embeddings.py",
        ROOT / "scripts" / "merge_feature_views.py",
        ROOT / "training" / "train_liquid.py",
    ]
    table_pattern = re.compile(
        r"\b(?:FROM|JOIN|INTO|UPDATE|TRUNCATE\s+TABLE|CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS)\s+([a-z_][a-z0-9_]*)",
        flags=re.IGNORECASE,
    )
    discovered: Dict[str, List[str]] = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        tables = sorted({m.group(1).lower() for m in table_pattern.finditer(text)})
        discovered[str(f.relative_to(ROOT))] = tables

    must = ["market_bars", "feature_snapshots_main", "feature_matrix_main"]
    return {
        "files": list(discovered.keys()),
        "discovered_by_file": discovered,
        "required_tables": must,
        "minimal_viable_combo": list(must),
    }


def _ensure_audit_table(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS offline_data_audits (
            id BIGSERIAL PRIMARY KEY,
            task_id VARCHAR(64) NOT NULL UNIQUE,
            status VARCHAR(16) NOT NULL DEFAULT 'completed',
            track VARCHAR(32) NOT NULL DEFAULT 'liquid',
            symbols_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            window_start TIMESTAMPTZ,
            window_end TIMESTAMPTZ,
            lookback INTEGER NOT NULL DEFAULT 96,
            bucket VARCHAR(8) NOT NULL DEFAULT '5m',
            ready BOOLEAN NOT NULL DEFAULT FALSE,
            reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
            payload JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_by VARCHAR(128) NOT NULL DEFAULT 'system',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def _resolve_symbols_from_snapshot(
    cur,
    *,
    track: str,
    as_of: datetime,
    top_n: int,
) -> List[str]:
    if not _table_exists(cur, "asset_universe_snapshots"):
        return []
    cur.execute(
        """
        SELECT symbols_json
        FROM asset_universe_snapshots
        WHERE track = %s AND as_of <= %s
        ORDER BY as_of DESC, created_at DESC
        LIMIT 1
        """,
        (track, as_of),
    )
    row = dict(cur.fetchone() or {})
    symbols = row.get("symbols_json")
    raw_items: List[Any] = []
    if isinstance(symbols, list):
        raw_items = list(symbols)
    elif isinstance(symbols, dict) and isinstance(symbols.get("symbols"), list):
        raw_items = list(symbols.get("symbols"))
    if not raw_items:
        return []
    out = []
    seen = set()
    for item in raw_items:
        s = str(item or "").strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out[: max(1, int(top_n))]


def _safe_feature_zero_fill_anomaly(
    cur,
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    values_col: Optional[str] = None,
    mask_col: Optional[str] = None,
) -> Dict[str, Any]:
    if not _table_exists(cur, "feature_matrix_main"):
        return {"checked": False, "flagged": False, "zero_ratio": None}
    fm_cols = _table_columns(cur, "feature_matrix_main")
    v_col = str(values_col or _pick_col(fm_cols, ("values", "feature_values")) or "")
    m_col = str(mask_col or _pick_col(fm_cols, ("mask", "feature_mask")) or "")
    if not v_col or not m_col:
        return {"checked": False, "flagged": False, "zero_ratio": None, "reason": "feature_matrix_vector_columns_missing"}
    cur.execute(
        f"""
        SELECT {v_col} AS values_col, {m_col} AS mask_col
        FROM feature_matrix_main
        WHERE UPPER(symbol) = %s
          AND as_of_ts >= %s
          AND as_of_ts <= %s
        ORDER BY as_of_ts DESC
        LIMIT 64
        """,
        (symbol, start, end),
    )
    rows = [dict(r) for r in (cur.fetchall() or [])]
    observed = 0
    zeros = 0
    for row in rows:
        vals = row.get("values_col")
        msk = row.get("mask_col")
        if not isinstance(vals, list) or not isinstance(msk, list) or len(vals) != len(msk):
            continue
        for v, m in zip(vals, msk):
            if int(m or 0) != 0:
                continue
            observed += 1
            if abs(float(v or 0.0)) <= 1e-12:
                zeros += 1
    if observed <= 0:
        return {"checked": True, "flagged": False, "zero_ratio": 0.0}
    ratio = float(zeros / observed)
    return {
        "checked": True,
        "flagged": bool(ratio >= 0.98 and observed >= 500),
        "zero_ratio": ratio,
        "observed_entries": observed,
    }


@dataclass
class AuditArgs:
    database_url: str
    track: str
    symbols: List[str]
    start: datetime
    end: datetime
    lookback: int
    bucket: str
    min_market_ratio: float
    min_feature_matrix_ratio: float
    strict_schema_mismatch_zero: bool
    min_text_coverage: float
    enforce_text_coverage: bool
    created_by: str
    task_id: str
    as_of: Optional[datetime] = None
    top_n: int = DEFAULT_TOP_N


def run_audit(args: AuditArgs, progress_cb: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    def _progress(stage: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if progress_cb:
            progress_cb(stage, extra or {})

    _progress("connect", {"database_url": "configured"})
    bucket_sec = _bucket_seconds(args.bucket)
    expected = _expected_buckets(args.start, args.end, bucket_sec)
    horizons = _parse_horizon_steps(bucket_sec)
    max_h = max(horizons.values())

    out: Dict[str, Any] = {
        "task_id": args.task_id,
        "generated_at": _to_iso(_utcnow()),
        "track": args.track,
        "as_of": _to_iso(args.as_of or args.start),
        "window": {
            "start": _to_iso(args.start),
            "end": _to_iso(args.end),
            "lookback": int(args.lookback),
            "bucket": args.bucket,
            "expected_buckets": int(expected),
        },
        "pipeline_requirements": _derive_required_tables_from_code(),
        "thresholds": {
            "market_bars_present_bucket_ratio_min": float(args.min_market_ratio),
            "feature_matrix_present_bucket_ratio_min": float(args.min_feature_matrix_ratio),
            "schema_hash_mismatch_must_be_zero": bool(args.strict_schema_mismatch_zero),
            "text_coverage_min": float(args.min_text_coverage),
            "text_coverage_enforced": bool(args.enforce_text_coverage),
        },
        "table_exists": {},
        "coverage": {
            "market_bars": {"per_symbol": {}, "avg_present_bucket_ratio": 0.0},
            "feature_matrix_main": {},
            "text_source": {},
        },
        "symbol_readiness": {},
        "summary": {"READY": 0, "DEGRADED": 0, "BLOCKED": 0},
        "gates": {"ready": False, "reasons": []},
        "repair_suggestions": [],
    }

    reasons: List[str] = []
    suggestions: List[str] = []

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            _ensure_audit_table(cur)
            _progress("check_tables")

            must_tables = [
                "market_bars",
                "feature_snapshots_main",
                "feature_matrix_main",
                "orderbook_l2",
                "funding_rates",
                "open_interest",
                "events",
                "event_links",
                "entities",
                "social_text_embeddings",
                "social_posts_raw",
                "social_comments_raw",
            ]
            table_flags = {t: _table_exists(cur, t) for t in must_tables}
            out["table_exists"] = table_flags
            out["coverage"]["text_source"] = {
                "social_posts_raw": {"exists": bool(table_flags.get("social_posts_raw"))},
                "social_comments_raw": {"exists": bool(table_flags.get("social_comments_raw"))},
                "events": {"exists": bool(table_flags.get("events"))},
                "social_text_embeddings": {"exists": bool(table_flags.get("social_text_embeddings"))},
            }

            event_cols = _table_columns(cur, "events") if table_flags.get("events") else set()
            has_events_symbol_col = "symbol" in event_cols
            has_event_entity_join = bool(table_flags.get("events") and table_flags.get("event_links") and table_flags.get("entities"))
            fm_cols = _table_columns(cur, "feature_matrix_main") if table_flags.get("feature_matrix_main") else set()
            fm_values_col = _pick_col(fm_cols, ("values", "feature_values"))
            fm_mask_col = _pick_col(fm_cols, ("mask", "feature_mask"))
            fm_features_col = _pick_col(fm_cols, ("features", "feature_payload"))
            fm_feature_dim_col = _pick_col(fm_cols, ("feature_dim",))
            fm_schema_col = _pick_col(fm_cols, ("schema_hash",))

            symbols = list(args.symbols)
            if not symbols:
                symbols = _resolve_symbols_from_snapshot(
                    cur,
                    track=str(args.track),
                    as_of=(args.as_of or args.start),
                    top_n=int(args.top_n),
                )
            if not symbols:
                raise RuntimeError("audit_symbols_empty")

            market_ratios: List[float] = []
            per_symbol: Dict[str, Any] = {}

            for sym in symbols:
                symbol = str(sym).upper()
                blocked_reasons: List[str] = []
                degraded_reasons: List[str] = []

                # A) market bars coverage
                market_present = 0
                if table_flags.get("market_bars"):
                    market_present = _distinct_bucket_count(
                        cur,
                        table_name="market_bars",
                        ts_col="ts",
                        start_dt=args.start,
                        end_dt=args.end,
                        bucket_sec=bucket_sec,
                        symbol_col="symbol",
                        symbol=symbol,
                        timeframe_col="timeframe",
                        timeframe=args.bucket,
                    )
                market_ratio = _coverage(market_present, expected)
                market_ratios.append(market_ratio)
                if market_ratio < float(args.min_market_ratio):
                    blocked_reasons.append(f"market_bars_coverage_low:{market_ratio:.4f}")

                # F) feature matrix coverage + schema + dim
                rows_total = 0
                feature_dim_ok = 0
                vector_dim_ok = 0
                schema_mismatch = 0
                feature_present = 0
                if table_flags.get("feature_matrix_main"):
                    if not fm_feature_dim_col:
                        blocked_reasons.append("feature_matrix_required_columns_missing")
                    elif fm_values_col and fm_mask_col:
                        schema_expr = (
                            f"COALESCE(SUM(CASE WHEN {fm_schema_col} <> %s THEN 1 ELSE 0 END), 0)::bigint"
                            if fm_schema_col
                            else "0::bigint"
                        )
                        cur.execute(
                            f"""
                        SELECT
                            COUNT(*)::bigint AS rows_total,
                            COALESCE(SUM(CASE WHEN {fm_feature_dim_col} = %s THEN 1 ELSE 0 END), 0)::bigint AS feature_dim_ok_rows,
                            COALESCE(SUM(CASE WHEN jsonb_array_length({fm_values_col}) = %s AND jsonb_array_length({fm_mask_col}) = %s THEN 1 ELSE 0 END), 0)::bigint AS vector_dim_ok_rows,
                            {schema_expr} AS schema_hash_mismatch
                        FROM feature_matrix_main
                        WHERE UPPER(symbol) = %s
                          AND as_of_ts >= %s
                          AND as_of_ts <= %s
                            """,
                            (
                                int(FEATURE_DIM),
                                int(FEATURE_DIM),
                                int(FEATURE_DIM),
                                str(SCHEMA_HASH),
                                symbol,
                                args.start,
                                args.end,
                            )
                            if fm_schema_col
                            else (int(FEATURE_DIM), int(FEATURE_DIM), int(FEATURE_DIM), symbol, args.start, args.end),
                        )
                        rr = dict(cur.fetchone() or {})
                        rows_total = int(rr.get("rows_total") or 0)
                        feature_dim_ok = int(rr.get("feature_dim_ok_rows") or 0)
                        vector_dim_ok = int(rr.get("vector_dim_ok_rows") or 0)
                        schema_mismatch = int(rr.get("schema_hash_mismatch") or 0)
                        feature_present = _distinct_bucket_count(
                            cur,
                            table_name="feature_matrix_main",
                            ts_col="as_of_ts",
                            start_dt=args.start,
                            end_dt=args.end,
                            bucket_sec=bucket_sec,
                            symbol_col="symbol",
                            symbol=symbol,
                        )
                    elif fm_features_col:
                        schema_expr = (
                            f"COALESCE(SUM(CASE WHEN {fm_schema_col} <> %s THEN 1 ELSE 0 END), 0)::bigint"
                            if fm_schema_col
                            else "0::bigint"
                        )
                        cur.execute(
                            f"""
                        SELECT
                            COUNT(*)::bigint AS rows_total,
                            COALESCE(SUM(CASE WHEN {fm_feature_dim_col} = %s THEN 1 ELSE 0 END), 0)::bigint AS feature_dim_ok_rows,
                            COALESCE(SUM(CASE WHEN jsonb_typeof({fm_features_col}) = 'object' THEN 1 ELSE 0 END), 0)::bigint AS vector_dim_ok_rows,
                            {schema_expr} AS schema_hash_mismatch
                        FROM feature_matrix_main
                        WHERE UPPER(symbol) = %s
                          AND as_of_ts >= %s
                          AND as_of_ts <= %s
                            """,
                            (int(FEATURE_DIM), str(SCHEMA_HASH), symbol, args.start, args.end)
                            if fm_schema_col
                            else (int(FEATURE_DIM), symbol, args.start, args.end),
                        )
                        rr = dict(cur.fetchone() or {})
                        rows_total = int(rr.get("rows_total") or 0)
                        feature_dim_ok = int(rr.get("feature_dim_ok_rows") or 0)
                        vector_dim_ok = int(rr.get("vector_dim_ok_rows") or 0)
                        schema_mismatch = int(rr.get("schema_hash_mismatch") or 0)
                        feature_present = _distinct_bucket_count(
                            cur,
                            table_name="feature_matrix_main",
                            ts_col="as_of_ts",
                            start_dt=args.start,
                            end_dt=args.end,
                            bucket_sec=bucket_sec,
                            symbol_col="symbol",
                            symbol=symbol,
                        )
                    else:
                        blocked_reasons.append("feature_matrix_required_columns_missing")
                feature_ratio = _coverage(feature_present, expected)
                if rows_total <= 0:
                    blocked_reasons.append("feature_matrix_missing")
                if feature_ratio < float(args.min_feature_matrix_ratio):
                    blocked_reasons.append(f"feature_matrix_coverage_low:{feature_ratio:.4f}")
                if feature_dim_ok != rows_total or vector_dim_ok != rows_total:
                    blocked_reasons.append("feature_matrix_dim_mismatch")
                if bool(args.strict_schema_mismatch_zero) and schema_mismatch > 0:
                    blocked_reasons.append("feature_matrix_schema_mismatch")

                anomaly = _safe_feature_zero_fill_anomaly(
                    cur,
                    symbol,
                    args.start,
                    args.end,
                    values_col=fm_values_col,
                    mask_col=fm_mask_col,
                )
                if bool(anomaly.get("flagged")):
                    blocked_reasons.append("feature_matrix_suspected_zero_fill_without_mask")

                # B) orderbook coverage
                orderbook_ratio = None
                if table_flags.get("orderbook_l2"):
                    present = _distinct_bucket_count(
                        cur,
                        table_name="orderbook_l2",
                        ts_col="ts",
                        start_dt=args.start,
                        end_dt=args.end,
                        bucket_sec=bucket_sec,
                        symbol_col="symbol",
                        symbol=symbol,
                    )
                    orderbook_ratio = _coverage(present, expected)
                    if orderbook_ratio < 0.30:
                        degraded_reasons.append(f"orderbook_coverage_low:{orderbook_ratio:.4f}")

                # C) derivatives coverage
                funding_ratio = None
                if table_flags.get("funding_rates"):
                    present = _distinct_bucket_count(
                        cur,
                        table_name="funding_rates",
                        ts_col="ts",
                        start_dt=args.start,
                        end_dt=args.end,
                        bucket_sec=bucket_sec,
                        symbol_col="symbol",
                        symbol=symbol,
                    )
                    funding_ratio = _coverage(present, expected)
                    if funding_ratio < 0.20:
                        degraded_reasons.append(f"funding_coverage_low:{funding_ratio:.4f}")

                oi_ratio = None
                if table_flags.get("open_interest"):
                    present = _distinct_bucket_count(
                        cur,
                        table_name="open_interest",
                        ts_col="ts",
                        start_dt=args.start,
                        end_dt=args.end,
                        bucket_sec=bucket_sec,
                        symbol_col="symbol",
                        symbol=symbol,
                    )
                    oi_ratio = _coverage(present, expected)
                    if oi_ratio < 0.20:
                        degraded_reasons.append(f"oi_coverage_low:{oi_ratio:.4f}")

                # D) events coverage
                event_count = 0
                if table_flags.get("events"):
                    if has_events_symbol_col:
                        cur.execute(
                            """
                            SELECT COUNT(*)::bigint AS c
                            FROM events
                            WHERE UPPER(symbol) = %s
                              AND occurred_at >= %s
                              AND occurred_at <= %s
                            """,
                            (symbol, args.start, args.end),
                        )
                    elif has_event_entity_join:
                        cur.execute(
                            """
                            SELECT COUNT(*)::bigint AS c
                            FROM events e
                            JOIN event_links el ON el.event_id = e.id
                            JOIN entities en ON en.id = el.entity_id
                            WHERE UPPER(COALESCE(en.symbol,'')) = %s
                              AND e.occurred_at >= %s
                              AND e.occurred_at <= %s
                            """,
                            (symbol, args.start, args.end),
                        )
                    else:
                        cur.execute("SELECT 0::bigint AS c")
                    event_count = int((dict(cur.fetchone() or {})).get("c") or 0)
                    if event_count <= 0:
                        degraded_reasons.append("events_missing")

                # E) text embedding coverage
                text_cov = 0.0
                if table_flags.get("social_text_embeddings"):
                    present = _distinct_bucket_count(
                        cur,
                        table_name="social_text_embeddings",
                        ts_col="as_of_ts",
                        start_dt=args.start,
                        end_dt=args.end,
                        bucket_sec=bucket_sec,
                        symbol_col="symbol",
                        symbol=symbol,
                    )
                    text_cov = _coverage(present, expected)
                if text_cov < float(args.min_text_coverage):
                    tag = f"text_coverage_low:{text_cov:.4f}"
                    if bool(args.enforce_text_coverage):
                        blocked_reasons.append(tag)
                    else:
                        degraded_reasons.append(tag)

                # label window viability
                label_rows = 0
                if table_flags.get("market_bars"):
                    cur.execute(
                        """
                        SELECT COUNT(*)::bigint AS c
                        FROM market_bars
                        WHERE UPPER(symbol)=%s AND timeframe=%s
                          AND ts >= %s AND ts <= %s + (%s * INTERVAL '1 second')
                        """,
                        (symbol, args.bucket, args.start, args.end, int(max_h * bucket_sec)),
                    )
                    label_rows = int((dict(cur.fetchone() or {})).get("c") or 0)
                need = int(args.lookback + max_h + 8)
                if label_rows < need:
                    blocked_reasons.append(f"label_window_insufficient:{label_rows}<{need}")

                if blocked_reasons:
                    status = "BLOCKED"
                elif degraded_reasons:
                    status = "DEGRADED"
                else:
                    status = "READY"

                out["summary"][status] = int(out["summary"].get(status, 0) + 1)
                per_symbol[symbol] = {
                    "status": status,
                    "blocked_reasons": blocked_reasons,
                    "degraded_reasons": degraded_reasons,
                    "coverage": {
                        "market_bars": round(market_ratio, 6),
                        "feature_matrix_main": round(feature_ratio, 6),
                        "orderbook_l2": None if orderbook_ratio is None else round(orderbook_ratio, 6),
                        "funding_rates": None if funding_ratio is None else round(funding_ratio, 6),
                        "open_interest": None if oi_ratio is None else round(oi_ratio, 6),
                        "text_embeddings": round(text_cov, 6),
                    },
                    "feature_matrix": {
                        "rows_total": rows_total,
                        "feature_dim_ok_rows": feature_dim_ok,
                        "vector_dim_ok_rows": vector_dim_ok,
                        "schema_hash_mismatch": schema_mismatch,
                        "zero_fill_anomaly": anomaly,
                    },
                    "event_count": int(event_count),
                    "label_rows": int(label_rows),
                    "label_required_rows": int(need),
                }

            out["symbol_readiness"] = per_symbol
            out["symbols"] = list(per_symbol.keys())
            out["coverage"]["market_bars"]["per_symbol"] = {
                k: {"present_bucket_ratio": v["coverage"]["market_bars"]} for k, v in per_symbol.items()
            }
            out["coverage"]["market_bars"]["avg_present_bucket_ratio"] = float(sum(market_ratios) / max(1, len(market_ratios)))

            blocked_symbols = [s for s, row in per_symbol.items() if row.get("status") == "BLOCKED"]
            degraded_symbols = [s for s, row in per_symbol.items() if row.get("status") == "DEGRADED"]
            if blocked_symbols:
                reasons.append(f"blocked_symbols:{','.join(blocked_symbols)}")
                suggestions.append("fix blocked symbols or exclude them before training")
            if degraded_symbols:
                suggestions.append("review degraded symbols and keep explicit annotation in training report")

            out["blocked_symbols"] = blocked_symbols
            out["degraded_symbols"] = degraded_symbols
            out["ready_symbols"] = [s for s, row in per_symbol.items() if row.get("status") == "READY"]
            out["gates"]["ready"] = len(blocked_symbols) == 0
            out["gates"]["reasons"] = reasons
            out["repair_suggestions"] = list(dict.fromkeys(suggestions))

            cur.execute(
                """
                INSERT INTO offline_data_audits (
                    task_id, status, track, symbols_json, window_start, window_end,
                    lookback, bucket, ready, reasons, payload, created_by, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s::jsonb, %s, %s,
                    %s, %s, %s, %s::jsonb, %s::jsonb, %s, NOW(), NOW()
                )
                ON CONFLICT (task_id)
                DO UPDATE SET
                    status = EXCLUDED.status,
                    track = EXCLUDED.track,
                    symbols_json = EXCLUDED.symbols_json,
                    window_start = EXCLUDED.window_start,
                    window_end = EXCLUDED.window_end,
                    lookback = EXCLUDED.lookback,
                    bucket = EXCLUDED.bucket,
                    ready = EXCLUDED.ready,
                    reasons = EXCLUDED.reasons,
                    payload = EXCLUDED.payload,
                    created_by = EXCLUDED.created_by,
                    updated_at = NOW()
                """,
                (
                    str(args.task_id),
                    "completed",
                    str(args.track),
                    json.dumps(out["symbols"]),
                    args.start,
                    args.end,
                    int(args.lookback),
                    str(args.bucket),
                    bool(out["gates"]["ready"]),
                    json.dumps(out["gates"]["reasons"]),
                    json.dumps(out),
                    str(args.created_by),
                ),
            )
        conn.commit()

    _progress("done", {"ready": bool(out["gates"]["ready"])})
    return out


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Audit offline training data readiness for strict liquid pipeline")
    ap.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    ap.add_argument("--track", default=DEFAULT_TRACK)
    ap.add_argument("--as-of", default="")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    ap.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=420)).isoformat().replace("+00:00", "Z"))
    ap.add_argument("--end", default=_utcnow().isoformat().replace("+00:00", "Z"))
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "96")))
    ap.add_argument("--bucket", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", DEFAULT_BUCKET))
    ap.add_argument("--bucket-minutes", type=int, default=0)
    ap.add_argument("--min-market-ratio", type=float, default=float(os.getenv("AUDIT_MIN_MARKET_RATIO", "0.98")))
    ap.add_argument("--min-feature-matrix-ratio", type=float, default=float(os.getenv("AUDIT_MIN_FEATURE_MATRIX_RATIO", "0.95")))
    ap.add_argument("--strict-schema-mismatch-zero", type=int, default=1)
    ap.add_argument("--min-text-coverage", type=float, default=float(os.getenv("AUDIT_MIN_TEXT_COVERAGE", "0.05")))
    ap.add_argument("--enforce-text-coverage", type=int, default=int(os.getenv("AUDIT_ENFORCE_TEXT_COVERAGE", "0")))
    ap.add_argument("--created-by", default=os.getenv("AUDIT_CREATED_BY", "manual"))
    ap.add_argument("--task-id", default="")
    ap.add_argument("--output", default="artifacts/audit/top50_data_readiness_latest.json")
    return ap


def main() -> int:
    ap = _build_parser()
    ns = ap.parse_args()
    task_id = str(ns.task_id or "").strip() or f"audit-{uuid.uuid4().hex[:16]}"
    as_of = _parse_dt(str(ns.as_of)) if str(ns.as_of).strip() else None
    bucket = str(ns.bucket).strip().lower()
    if int(ns.bucket_minutes) > 0:
        bucket = f"{int(ns.bucket_minutes)}m"
    args = AuditArgs(
        database_url=str(ns.database_url),
        track=str(ns.track).strip().lower() or DEFAULT_TRACK,
        symbols=_parse_symbols(ns.symbols),
        start=_parse_dt(str(ns.start)),
        end=_parse_dt(str(ns.end)),
        lookback=max(1, int(ns.lookback)),
        bucket=bucket,
        min_market_ratio=float(ns.min_market_ratio),
        min_feature_matrix_ratio=float(ns.min_feature_matrix_ratio),
        strict_schema_mismatch_zero=bool(int(ns.strict_schema_mismatch_zero)),
        min_text_coverage=float(ns.min_text_coverage),
        enforce_text_coverage=bool(int(ns.enforce_text_coverage)),
        created_by=str(ns.created_by),
        task_id=task_id,
        as_of=as_of,
        top_n=max(1, int(ns.top_n)),
    )
    out = run_audit(args)

    out_path = Path(str(ns.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = out.get("summary") or {}
    print(
        json.dumps(
            {
                "status": "ok",
                "task_id": task_id,
                "output": str(out_path),
                "ready": int(summary.get("READY") or 0),
                "degraded": int(summary.get("DEGRADED") or 0),
                "blocked": int(summary.get("BLOCKED") or 0),
                "blocked_symbols": list(out.get("blocked_symbols") or []),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
