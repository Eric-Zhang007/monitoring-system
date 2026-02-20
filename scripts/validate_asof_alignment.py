#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None  # type: ignore[assignment]
    RealDictCursor = None  # type: ignore[assignment]

_ROOT = Path(__file__).resolve().parents[1]
_INFER_DIR = _ROOT / "inference"
if str(_INFER_DIR) not in sys.path:
    sys.path.append(str(_INFER_DIR))
try:
    from liquid_feature_contract import LIQUID_FULL_FEATURE_KEYS, LIQUID_MANUAL_FEATURE_KEYS  # type: ignore
except Exception:
    LIQUID_MANUAL_FEATURE_KEYS = []  # type: ignore[assignment]
    LIQUID_FULL_FEATURE_KEYS = []  # type: ignore[assignment]


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) AS reg", (f"public.{table_name}",))
    row = cur.fetchone() or {}
    return bool(row.get("reg"))


def _columns(cur, table_name: str) -> set[str]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate strict as-of alignment and leakage")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--track", default="liquid")
    parser.add_argument("--out-json", default="artifacts/audit/asof_alignment_latest.json")
    args = parser.parse_args()

    if psycopg2 is None or RealDictCursor is None:
        raise SystemExit("missing psycopg2 dependency; run: PROFILE=runtime bash scripts/bootstrap_env.sh")

    out: Dict[str, Any] = {
        "future_leakage_count": 0,
        "snapshot_sample_time_match_rate": 0.0,
        "snapshot_sample_time_mismatch_count": 0,
        "checked_rows": 0,
        "feature_snapshots_main_present": False,
        "feature_matrix_main_present": False,
        "main_snapshot_feature_dim_mismatch_count": 0,
        "matrix_feature_dim_mismatch_count": 0,
        "matrix_asof_null_count": 0,
        "passed": False,
    }

    with psycopg2.connect(args.database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            if not _table_exists(cur, "feature_snapshots"):
                out["reason"] = "feature_snapshots_table_missing"
                text = json.dumps(out, ensure_ascii=False)
                out_path = str(args.out_json).strip()
                if out_path:
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text + "\n")
                print(text)
                return 2

            cols = _columns(cur, "feature_snapshots")
            as_of_col = "as_of_ts" if "as_of_ts" in cols else ("as_of" if "as_of" in cols else None)
            if not as_of_col:
                out["reason"] = "as_of_column_missing"
                text = json.dumps(out, ensure_ascii=False)
                out_path = str(args.out_json).strip()
                if out_path:
                    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(text + "\n")
                print(text)
                return 2

            event_col = "event_time" if "event_time" in cols else None
            avail_col = "feature_available_at" if "feature_available_at" in cols else None

            leak_checks = []
            if event_col:
                leak_checks.append(f"({event_col} IS NOT NULL AND {event_col} > {as_of_col})")
            if avail_col:
                leak_checks.append(f"({avail_col} IS NOT NULL AND {avail_col} > {as_of_col})")
            leak_pred = " OR ".join(leak_checks) if leak_checks else "FALSE"

            cur.execute(
                f"""
                SELECT COUNT(*)::bigint AS c
                FROM feature_snapshots
                WHERE track = %s
                  AND ({leak_pred})
                """,
                (str(args.track),),
            )
            out["future_leakage_count"] = int((cur.fetchone() or {}).get("c") or 0)

            if event_col:
                cur.execute(
                    f"""
                    SELECT
                      COUNT(*)::bigint AS total_rows,
                      COUNT(*) FILTER (
                        WHERE {event_col} IS NOT NULL
                          AND ABS(EXTRACT(EPOCH FROM ({as_of_col} - {event_col}))) <= 1
                      )::bigint AS matched_rows
                    FROM feature_snapshots
                    WHERE track = %s
                      AND lineage_id LIKE 'train-%%'
                    """,
                    (str(args.track),),
                )
                row = cur.fetchone() or {}
                total_rows = int(row.get("total_rows") or 0)
                matched_rows = int(row.get("matched_rows") or 0)
                mismatch_rows = max(0, total_rows - matched_rows)
                out["checked_rows"] = total_rows
                out["snapshot_sample_time_mismatch_count"] = mismatch_rows
                out["snapshot_sample_time_match_rate"] = round(float(matched_rows / max(1, total_rows)), 8)
            else:
                out["reason"] = "event_time_column_missing"
                out["checked_rows"] = 0
                out["snapshot_sample_time_mismatch_count"] = 0
                out["snapshot_sample_time_match_rate"] = 0.0

            if _table_exists(cur, "feature_snapshots_main"):
                out["feature_snapshots_main_present"] = True
                expected_manual_dim = int(len(LIQUID_MANUAL_FEATURE_KEYS))
                if expected_manual_dim > 0:
                    cur.execute(
                        """
                        SELECT COUNT(*)::bigint AS c
                        FROM feature_snapshots_main
                        WHERE feature_dim IS DISTINCT FROM %s
                        """,
                        (expected_manual_dim,),
                    )
                    out["main_snapshot_feature_dim_mismatch_count"] = int((cur.fetchone() or {}).get("c") or 0)

            if _table_exists(cur, "feature_matrix_main"):
                out["feature_matrix_main_present"] = True
                expected_full_dim = int(len(LIQUID_FULL_FEATURE_KEYS))
                if expected_full_dim > 0:
                    cur.execute(
                        """
                        SELECT COUNT(*)::bigint AS c
                        FROM feature_matrix_main
                        WHERE feature_dim IS DISTINCT FROM %s
                        """,
                        (expected_full_dim,),
                    )
                    out["matrix_feature_dim_mismatch_count"] = int((cur.fetchone() or {}).get("c") or 0)

                cur.execute(
                    """
                    SELECT COUNT(*)::bigint AS c
                    FROM feature_matrix_main
                    WHERE as_of_ts IS NULL
                    """,
                )
                out["matrix_asof_null_count"] = int((cur.fetchone() or {}).get("c") or 0)

    out["passed"] = bool(
        int(out["future_leakage_count"]) == 0
        and float(out["snapshot_sample_time_match_rate"]) >= 1.0
        and bool(out["feature_snapshots_main_present"])
        and bool(out["feature_matrix_main_present"])
        and int(out["main_snapshot_feature_dim_mismatch_count"]) == 0
        and int(out["matrix_feature_dim_mismatch_count"]) == 0
        and int(out["matrix_asof_null_count"]) == 0
    )

    text = json.dumps(out, ensure_ascii=False)
    out_path = str(args.out_json).strip()
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
