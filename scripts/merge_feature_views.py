#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from features.align import validate_schema_hash
from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, FEATURE_KEYS, GROUP_MAP, SCHEMA_HASH, SCHEMA_VERSION


TEXT_EMB_KEYS = [k for k in FEATURE_KEYS if k.startswith("text_emb_")]


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_feature_payload(values: np.ndarray, mask: np.ndarray) -> Dict[str, Dict[str, float | int]]:
    out: Dict[str, Dict[str, float | int]] = {}
    for i, k in enumerate(FEATURE_KEYS):
        out[k] = {"value": float(values[i]), "missing": int(mask[i])}
    return out


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _latest_before(ts_list: List[datetime], rows: List[Dict[str, Any]], ts: datetime) -> Dict[str, Any] | None:
    if not ts_list:
        return None
    idx = bisect_right(ts_list, ts) - 1
    if idx < 0:
        return None
    return rows[idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge strict numeric snapshots + semantic text embeddings")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--start", default="2018-01-01T00:00:00Z")
    ap.add_argument("--end", default="")
    ap.add_argument("--truncate", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

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
                    values JSONB NOT NULL,
                    mask JSONB NOT NULL,
                    features JSONB NOT NULL,
                    feature_dim INTEGER NOT NULL,
                    schema_hash TEXT NOT NULL,
                    feature_version TEXT NOT NULL,
                    synthetic_ratio DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_feature_matrix_main_symbol_ts ON feature_matrix_main(symbol, as_of_ts DESC)")
            if bool(args.truncate):
                cur.execute("TRUNCATE TABLE feature_matrix_main")

            sql = """
                SELECT symbol, as_of_ts, feature_values, feature_mask, feature_payload, schema_hash, synthetic_ratio
                FROM feature_snapshots_main
                WHERE as_of_ts >= %s AND as_of_ts <= %s
                ORDER BY symbol ASC, as_of_ts ASC
            """
            params: List[Any] = [start_dt, end_dt]
            if int(args.max_rows) > 0:
                sql += " LIMIT %s"
                params.append(int(args.max_rows))
            cur.execute(sql, tuple(params))
            base_rows = [dict(r) for r in cur.fetchall()]

            if not base_rows:
                print(json.dumps({"status": "ok", "rows_inserted": 0, "table": "feature_matrix_main"}, ensure_ascii=False))
                return 0

            symbols = sorted({str(r.get("symbol") or "").upper() for r in base_rows if str(r.get("symbol") or "").strip()})
            cur.execute(
                """
                DELETE FROM feature_matrix_main
                WHERE symbol = ANY(%s)
                  AND as_of_ts >= %s
                  AND as_of_ts <= %s
                  AND feature_version = %s
                """,
                (symbols, start_dt, end_dt, str(SCHEMA_VERSION)),
            )

            text_map: Dict[str, Tuple[List[datetime], List[Dict[str, Any]]]] = {}
            if _table_exists(cur, "social_text_embeddings"):
                cur.execute(
                    """
                    SELECT symbol, as_of_ts, embedding, embedding_dim, text_quality, schema_hash
                    FROM social_text_embeddings
                    WHERE symbol = ANY(%s)
                      AND as_of_ts >= %s - INTERVAL '6 hour'
                      AND as_of_ts <= %s
                    ORDER BY symbol ASC, as_of_ts ASC
                    """,
                    (symbols, start_dt, end_dt),
                )
                for r in cur.fetchall() or []:
                    row = dict(r)
                    sym = str(row.get("symbol") or "").upper()
                    ts = row.get("as_of_ts")
                    if not sym or not isinstance(ts, datetime):
                        continue
                    validate_schema_hash(str(row.get("schema_hash") or ""))
                    bucket = text_map.setdefault(sym, ([], []))
                    bucket[0].append(ts)
                    bucket[1].append(row)

            inserts = []
            for br in base_rows:
                sym = str(br.get("symbol") or "").upper()
                ts = br.get("as_of_ts")
                if not sym or (not isinstance(ts, datetime)):
                    continue

                validate_schema_hash(str(br.get("schema_hash") or ""))
                vals = np.array(br.get("feature_values") or [], dtype=np.float32).reshape(-1)
                msk = np.array(br.get("feature_mask") or [], dtype=np.uint8).reshape(-1)
                if vals.size != FEATURE_DIM or msk.size != FEATURE_DIM:
                    payload = br.get("feature_payload") if isinstance(br.get("feature_payload"), dict) else {}
                    vals = np.zeros((FEATURE_DIM,), dtype=np.float32)
                    msk = np.ones((FEATURE_DIM,), dtype=np.uint8)
                    for i, k in enumerate(FEATURE_KEYS):
                        item = payload.get(k) if isinstance(payload.get(k), dict) else {}
                        vals[i] = float((item or {}).get("value", 0.0) or 0.0)
                        msk[i] = int((item or {}).get("missing", 1) or 0)

                t_meta = text_map.get(sym)
                if t_meta:
                    trow = _latest_before(t_meta[0], t_meta[1], ts)
                else:
                    trow = None

                if isinstance(trow, dict):
                    emb = trow.get("embedding") if isinstance(trow.get("embedding"), list) else []
                    quality = trow.get("text_quality") if isinstance(trow.get("text_quality"), dict) else {}
                    for i, k in enumerate(TEXT_EMB_KEYS):
                        j = FEATURE_INDEX[k]
                        if i < len(emb):
                            vals[j] = float(emb[i])
                            msk[j] = 0
                        else:
                            vals[j] = 0.0
                            msk[j] = 1
                    mapping = {
                        "text_item_count": float(quality.get("num_items", 0.0) or 0.0),
                        "text_unique_authors": float(quality.get("unique_authors", 0.0) or 0.0),
                        "text_dup_ratio": float(quality.get("dup_ratio", 0.0) or 0.0),
                        "text_disagreement": float(quality.get("disagreement", 0.0) or 0.0),
                        "text_avg_lag_sec": float(quality.get("avg_lag_sec", 0.0) or 0.0),
                        "text_coverage": float(quality.get("coverage", 0.0) or 0.0),
                    }
                    for k, v in mapping.items():
                        idx = FEATURE_INDEX[k]
                        vals[idx] = float(v)
                        msk[idx] = 0
                    idx_ft = FEATURE_INDEX.get("freshness_text_sec")
                    if idx_ft is not None:
                        vals[idx_ft] = float(max(0.0, (ts - trow["as_of_ts"]).total_seconds()))
                        msk[idx_ft] = 0
                else:
                    for k in TEXT_EMB_KEYS:
                        idx = FEATURE_INDEX[k]
                        vals[idx] = 0.0
                        msk[idx] = 1
                    for k in ("text_item_count", "text_unique_authors", "text_dup_ratio", "text_disagreement", "text_avg_lag_sec", "text_coverage", "freshness_text_sec"):
                        idx = FEATURE_INDEX.get(k)
                        if idx is not None:
                            vals[idx] = 0.0
                            msk[idx] = 1

                payload = _to_feature_payload(vals, msk)
                inserts.append(
                    (
                        sym,
                        ts,
                        json.dumps([float(x) for x in vals.tolist()]),
                        json.dumps([int(x) for x in msk.tolist()]),
                        json.dumps(payload),
                        int(FEATURE_DIM),
                        str(SCHEMA_HASH),
                        str(SCHEMA_VERSION),
                        float(br.get("synthetic_ratio") or 0.0),
                    )
                )

            if inserts:
                execute_values(
                    cur,
                    """
                    INSERT INTO feature_matrix_main(
                        symbol, as_of_ts, values, mask, features,
                        feature_dim, schema_hash, feature_version, synthetic_ratio
                    ) VALUES %s
                    """,
                    inserts,
                    template="(%s,%s,%s::jsonb,%s::jsonb,%s::jsonb,%s,%s,%s,%s)",
                )
                inserted = len(inserts)

        conn.commit()

    print(
        json.dumps(
            {
                "status": "ok",
                "table": "feature_matrix_main",
                "rows_inserted": int(inserted),
                "feature_dim": int(FEATURE_DIM),
                "schema_hash": str(SCHEMA_HASH),
                "feature_version": str(SCHEMA_VERSION),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
