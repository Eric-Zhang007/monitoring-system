from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

from features.align import FeatureAlignmentError, align_row, build_all_missing_row, validate_schema_hash
from features.feature_contract import BUCKET_INTERVAL, FEATURE_DIM, SCHEMA_HASH
from features.feature_row import SequenceBatch


def _bucket_seconds(bucket_interval: str) -> int:
    tf = str(bucket_interval or BUCKET_INTERVAL).strip().lower()
    if tf.endswith("m"):
        return max(1, int(tf[:-1] or "5")) * 60
    if tf.endswith("h"):
        return max(1, int(tf[:-1] or "1")) * 3600
    if tf.endswith("d"):
        return max(1, int(tf[:-1] or "1")) * 86400
    return max(1, int(tf))


def _parse_ts(raw: object) -> datetime:
    if isinstance(raw, datetime):
        dt = raw
    else:
        text = str(raw or "").strip().replace(" ", "T")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _floor_bucket(ts: datetime, bucket_sec: int) -> datetime:
    epoch = int(ts.timestamp())
    floored = (epoch // bucket_sec) * bucket_sec
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def _coverage(mask: np.ndarray) -> Dict[str, float]:
    total = float(mask.size)
    missing = float(np.sum(mask))
    return {
        "missing_ratio": float(missing / max(1.0, total)),
        "observed_ratio": float(1.0 - (missing / max(1.0, total))),
    }


def _row_from_db_payload(row: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    schema_hash = str(row.get("schema_hash") or "")
    validate_schema_hash(schema_hash)
    vals = row.get("values")
    msk = row.get("mask")
    if isinstance(vals, list) and isinstance(msk, list):
        v = np.array(vals, dtype=np.float32).reshape(-1)
        m = np.array(msk, dtype=np.uint8).reshape(-1)
        if v.size != FEATURE_DIM or m.size != FEATURE_DIM:
            raise FeatureAlignmentError(f"db_feature_dim_mismatch:{v.size}:{m.size}:{FEATURE_DIM}")
        return v, m
    features = row.get("features") if isinstance(row.get("features"), dict) else {}
    aligned = align_row(features)
    return aligned.values, aligned.mask


def build_sequence(
    *,
    db_url: str,
    symbol: str,
    end_ts: datetime | str,
    lookback: int,
    bucket_interval: str = BUCKET_INTERVAL,
) -> SequenceBatch:
    target = str(symbol or "").strip().upper()
    if not target:
        raise FeatureAlignmentError("symbol_required")
    L = max(1, int(lookback))
    end_dt = _parse_ts(end_ts)
    bucket_sec = _bucket_seconds(bucket_interval)
    end_bucket = _floor_bucket(end_dt, bucket_sec)
    start_bucket = end_bucket - timedelta(seconds=bucket_sec * (L - 1))

    expected = [start_bucket + timedelta(seconds=bucket_sec * i) for i in range(L)]

    by_ts: Dict[datetime, Tuple[np.ndarray, np.ndarray]] = {}
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT as_of_ts, values, mask, features, schema_hash
                FROM feature_matrix_main
                WHERE symbol = %s
                  AND as_of_ts >= %s
                  AND as_of_ts <= %s
                ORDER BY as_of_ts ASC
                """,
                (target, start_bucket, end_bucket),
            )
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        ts = _floor_bucket(_parse_ts(r.get("as_of_ts")), bucket_sec)
        by_ts[ts] = _row_from_db_payload(r)

    miss = build_all_missing_row()
    values_rows: List[np.ndarray] = []
    mask_rows: List[np.ndarray] = []
    for ts in expected:
        pair = by_ts.get(ts)
        if pair is None:
            values_rows.append(miss.values.copy())
            mask_rows.append(miss.mask.copy())
        else:
            values_rows.append(pair[0])
            mask_rows.append(pair[1])

    values = np.stack(values_rows, axis=0).astype(np.float32)
    mask = np.stack(mask_rows, axis=0).astype(np.uint8)

    cov = _coverage(mask)
    cov.update(
        {
            "present_bucket_ratio": float(len(by_ts) / max(1, L)),
            "missing_bucket_ratio": float(1.0 - (len(by_ts) / max(1, L))),
        }
    )

    return SequenceBatch(
        values=values,
        mask=mask,
        schema_hash=str(SCHEMA_HASH),
        symbol=target,
        start_ts=start_bucket.isoformat().replace("+00:00", "Z"),
        end_ts=end_bucket.isoformat().replace("+00:00", "Z"),
        bucket_interval=str(bucket_interval),
        coverage_summary=cov,
    )
