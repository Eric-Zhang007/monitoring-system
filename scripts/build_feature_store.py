#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from features.align import align_row
from features.feature_contract import FEATURE_DIM, FEATURE_KEYS, SCHEMA_HASH, SCHEMA_VERSION
from features.recipes.base_signals import compute_price_signals
from features.recipes.derivatives import compute_derivatives
from features.recipes.microstructure import compute_microstructure
from features.recipes.onchain import compute_onchain


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _is_synthetic_source(raw: object) -> bool:
    text = str(raw or "").strip().lower()
    if not text:
        return False
    return ("synthetic" in text) or ("fallback" in text and "prices_fallback" not in text)


def _safe_float(v: object) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _latest_before(ts_list: List[datetime], rows: List[Dict[str, Any]], ts: datetime) -> Optional[Dict[str, Any]]:
    if not ts_list:
        return None
    idx = bisect_right(ts_list, ts) - 1
    if idx < 0:
        return None
    return rows[idx]


def _to_feature_payload(values: List[float], mask: List[int]) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for i, k in enumerate(FEATURE_KEYS):
        out[k] = {"value": float(values[i]), "missing": int(mask[i])}
    return out


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Build strict feature snapshots with values+mask+schema_hash")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--max-rows-per-symbol", type=int, default=0)
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--allow-synthetic", action="store_true", default=False)
    args = ap.parse_args()

    start_dt = _parse_dt(args.start)
    end_dt = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]

    rows_created = 0
    synthetic_used_rows = 0

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_snapshots_main (
                    id BIGSERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of_ts TIMESTAMPTZ NOT NULL,
                    feature_payload JSONB NOT NULL,
                    feature_values JSONB NOT NULL,
                    feature_mask JSONB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    schema_hash TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    synthetic_ratio DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_snapshots_main_symbol_ts ON feature_snapshots_main(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE feature_snapshots_main")

            has_orderbook = _table_exists(cur, "orderbook_l2")
            has_funding = _table_exists(cur, "funding_rates")
            has_onchain = _table_exists(cur, "onchain_signals")
            has_events = _table_exists(cur, "events") and _table_exists(cur, "event_links") and _table_exists(cur, "entities")

            for sym in symbols:
                cur.execute(
                    """
                    DELETE FROM feature_snapshots_main
                    WHERE symbol = %s
                      AND as_of_ts >= %s
                      AND as_of_ts <= %s
                      AND feature_version = %s
                    """,
                    (sym, start_dt, end_dt, str(SCHEMA_VERSION)),
                )

                sql = """
                    SELECT ts, open::double precision AS open, close::double precision AS close, volume::double precision AS volume
                    FROM market_bars
                    WHERE symbol = %s
                      AND timeframe = %s
                      AND ts >= %s
                      AND ts <= %s
                    ORDER BY ts ASC
                """
                params: List[Any] = [sym, str(args.timeframe), start_dt, end_dt]
                if int(args.max_rows_per_symbol) > 0:
                    sql += " LIMIT %s"
                    params.append(int(args.max_rows_per_symbol))
                cur.execute(sql, tuple(params))
                bars = [dict(r) for r in cur.fetchall()]
                if len(bars) < 32:
                    continue

                ob_ts: List[datetime] = []
                ob_rows: List[Dict[str, Any]] = []
                if has_orderbook:
                    cur.execute(
                        """
                        SELECT ts, bid_px, ask_px, bid_sz, ask_sz, spread_bps, imbalance, source
                        FROM orderbook_l2
                        WHERE symbol = %s AND ts >= %s - INTERVAL '2 day' AND ts <= %s
                        ORDER BY ts ASC
                        """,
                        (sym, start_dt, end_dt),
                    )
                    for r in cur.fetchall() or []:
                        row = dict(r)
                        row["is_synthetic"] = bool(_is_synthetic_source(row.get("source")))
                        ob_rows.append(row)
                        ob_ts.append(row["ts"])

                fr_ts: List[datetime] = []
                fr_rows: List[Dict[str, Any]] = []
                if has_funding:
                    cur.execute(
                        """
                        SELECT ts, funding_rate, source
                        FROM funding_rates
                        WHERE symbol = %s AND ts >= %s - INTERVAL '7 day' AND ts <= %s
                        ORDER BY ts ASC
                        """,
                        (sym, start_dt, end_dt),
                    )
                    for r in cur.fetchall() or []:
                        row = dict(r)
                        row["is_synthetic"] = bool(_is_synthetic_source(row.get("source")))
                        fr_rows.append(row)
                        fr_ts.append(row["ts"])

                oc_ts: List[datetime] = []
                oc_rows: List[Dict[str, Any]] = []
                if has_onchain:
                    cur.execute(
                        """
                        SELECT ts, metric_name, metric_value, source
                        FROM onchain_signals
                        WHERE asset_symbol = %s AND ts >= %s - INTERVAL '7 day' AND ts <= %s
                          AND metric_name IN ('net_inflow','active_addresses','basis_rate','taker_buy_sell_ratio','open_interest')
                        ORDER BY ts ASC
                        """,
                        (sym, start_dt, end_dt),
                    )
                    for r in cur.fetchall() or []:
                        row = dict(r)
                        row["is_synthetic"] = bool(_is_synthetic_source(row.get("source")))
                        oc_rows.append(row)
                        oc_ts.append(row["ts"])

                event_rows: List[Dict[str, Any]] = []
                if has_events:
                    cur.execute(
                        """
                        SELECT COALESCE(e.available_at, e.occurred_at) AS ts,
                               e.confidence_score,
                               COALESCE((e.payload->>'post_sentiment')::double precision, 0.0) AS post_sentiment,
                               e.payload
                        FROM events e
                        JOIN event_links el ON el.event_id = e.id
                        JOIN entities en ON en.id = el.entity_id
                        WHERE UPPER(COALESCE(en.symbol,'')) = %s
                          AND COALESCE(e.available_at, e.occurred_at) >= %s - INTERVAL '8 hour'
                          AND COALESCE(e.available_at, e.occurred_at) <= %s
                        ORDER BY COALESCE(e.available_at, e.occurred_at) ASC
                        """,
                        (sym, start_dt, end_dt),
                    )
                    event_rows = [dict(r) for r in cur.fetchall()]

                prices = [float(r.get("close") or 0.0) for r in bars]
                volumes = [float(r.get("volume") or 0.0) for r in bars]
                timestamps = [r.get("ts") for r in bars]

                inserts: List[tuple] = []
                for i in range(12, len(bars)):
                    ts = timestamps[i]
                    if not isinstance(ts, datetime):
                        continue
                    px_signals = compute_price_signals(prices, volumes, i)
                    if not px_signals:
                        continue

                    raw: Dict[str, object] = dict(px_signals)
                    source_total = 0
                    source_syn = 0

                    ob = _latest_before(ob_ts, ob_rows, ts)
                    if ob and ((not ob.get("is_synthetic")) or bool(args.allow_synthetic)):
                        micro = compute_microstructure(
                            bid_px=_safe_float(ob.get("bid_px")),
                            ask_px=_safe_float(ob.get("ask_px")),
                            bid_sz=_safe_float(ob.get("bid_sz")),
                            ask_sz=_safe_float(ob.get("ask_sz")),
                            spread_bps=_safe_float(ob.get("spread_bps")),
                            imbalance=_safe_float(ob.get("imbalance")),
                        )
                        raw.update(micro)
                        raw["freshness_orderbook_sec"] = float(max(0.0, (ts - ob["ts"]).total_seconds()))
                        source_total += 1
                        source_syn += int(bool(ob.get("is_synthetic")))

                    fr = _latest_before(fr_ts, fr_rows, ts)
                    if fr and ((not fr.get("is_synthetic")) or bool(args.allow_synthetic)):
                        raw.update(compute_derivatives(
                            funding_rate=_safe_float(fr.get("funding_rate")),
                            basis_rate=raw.get("basis_rate"),
                            taker_buy_sell_ratio=raw.get("taker_buy_sell_ratio"),
                            open_interest=raw.get("open_interest"),
                        ))
                        raw["freshness_funding_sec"] = float(max(0.0, (ts - fr["ts"]).total_seconds()))
                        source_total += 1
                        source_syn += int(bool(fr.get("is_synthetic")))

                    oc_latest = _latest_before(oc_ts, oc_rows, ts)
                    if oc_latest and ((not oc_latest.get("is_synthetic")) or bool(args.allow_synthetic)):
                        source_total += 1
                        source_syn += int(bool(oc_latest.get("is_synthetic")))
                        raw["freshness_onchain_sec"] = float(max(0.0, (ts - oc_latest["ts"]).total_seconds()))

                    net_inflow = None
                    active_addr = None
                    basis_rate = None
                    taker_ratio = None
                    open_interest = None
                    for row in reversed(oc_rows):
                        rts = row.get("ts")
                        if (not isinstance(rts, datetime)) or rts > ts:
                            continue
                        if row.get("is_synthetic") and (not bool(args.allow_synthetic)):
                            continue
                        name = str(row.get("metric_name") or "")
                        val = _safe_float(row.get("metric_value"))
                        if val is None:
                            continue
                        if name == "net_inflow" and net_inflow is None:
                            net_inflow = val
                        elif name == "active_addresses" and active_addr is None:
                            active_addr = val
                        elif name == "basis_rate" and basis_rate is None:
                            basis_rate = val
                        elif name == "taker_buy_sell_ratio" and taker_ratio is None:
                            taker_ratio = val
                        elif name == "open_interest" and open_interest is None:
                            open_interest = val
                        if all(x is not None for x in (net_inflow, active_addr, basis_rate, taker_ratio, open_interest)):
                            break

                    raw.update(compute_onchain(net_inflow=net_inflow, active_addresses=active_addr))
                    raw.update(compute_derivatives(
                        funding_rate=raw.get("funding_rate"),
                        basis_rate=basis_rate,
                        taker_buy_sell_ratio=taker_ratio,
                        open_interest=open_interest,
                    ))

                    ev1 = [e for e in event_rows if isinstance(e.get("ts"), datetime) and (ts - timedelta(hours=1) <= e["ts"] <= ts)]
                    ev6 = [e for e in event_rows if isinstance(e.get("ts"), datetime) and (ts - timedelta(hours=6) <= e["ts"] <= ts)]
                    conf_values = [float(e.get("confidence_score") or 0.0) for e in ev6]
                    sent_values = [float((e.get("payload") or {}).get("post_sentiment") or 0.0) for e in ev6]
                    raw["event_count_1h"] = float(len(ev1))
                    raw["event_count_6h"] = float(len(ev6))
                    raw["event_confidence_mean"] = float(sum(conf_values) / max(1, len(conf_values)))
                    raw["event_sentiment_mean"] = float(sum(sent_values) / max(1, len(sent_values)))

                    raw["freshness_market_sec"] = 0.0
                    raw["synthetic_ratio"] = float(source_syn / max(1, source_total)) if source_total > 0 else 0.0

                    aligned = align_row(raw)
                    values = [float(x) for x in aligned.values.tolist()]
                    mask = [int(x) for x in aligned.mask.tolist()]
                    if raw.get("synthetic_ratio", 0.0) and float(raw.get("synthetic_ratio", 0.0)) > 0.0:
                        synthetic_used_rows += 1

                    payload = _to_feature_payload(values, mask)
                    inserts.append(
                        (
                            sym,
                            ts,
                            json.dumps(payload),
                            json.dumps(values),
                            json.dumps(mask),
                            int(FEATURE_DIM),
                            str(SCHEMA_HASH),
                            str(SCHEMA_VERSION),
                            float(raw.get("synthetic_ratio", 0.0) or 0.0),
                        )
                    )

                if inserts:
                    execute_values(
                        cur,
                        """
                        INSERT INTO feature_snapshots_main(
                            symbol, as_of_ts, feature_payload, feature_values, feature_mask,
                            feature_dim, schema_hash, feature_version, synthetic_ratio
                        ) VALUES %s
                        """,
                        inserts,
                        template="(%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s)",
                    )
                    rows_created += len(inserts)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "feature_snapshots_main",
                "rows_created": int(rows_created),
                "feature_dim": int(FEATURE_DIM),
                "schema_hash": str(SCHEMA_HASH),
                "feature_version": str(SCHEMA_VERSION),
                "synthetic_used_rows": int(synthetic_used_rows),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
