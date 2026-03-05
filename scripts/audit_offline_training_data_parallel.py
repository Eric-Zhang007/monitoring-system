#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.audit_offline_training_data as audit_mod


def _resolve_symbols(database_url: str, track: str, as_of: datetime, top_n: int) -> List[str]:
    with psycopg2.connect(database_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            symbols = audit_mod._resolve_symbols_from_snapshot(cur, track=track, as_of=as_of, top_n=top_n)
    return [str(s).upper() for s in symbols]


def _run_one(base: audit_mod.AuditArgs, symbol: str, idx: int) -> Dict[str, Any]:
    args = audit_mod.AuditArgs(
        **{
            **asdict(base),
            "symbols": [str(symbol).upper()],
            "task_id": f"{base.task_id}-s{idx:03d}-{str(symbol).upper()}",
        }
    )
    out = audit_mod.run_audit(args)
    row = (out.get("symbol_readiness") or {}).get(str(symbol).upper())
    if not isinstance(row, dict):
        raise RuntimeError(f"missing_symbol_readiness:{symbol}")
    return {
        "symbol": str(symbol).upper(),
        "task_id": str(out.get("task_id") or ""),
        "row": row,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Parallel wrapper for offline training data audit")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--as-of", default="")
    ap.add_argument("--top-n", type=int, default=int(os.getenv("LIQUID_UNIVERSE_TOP_N", "50")))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", ""))
    ap.add_argument("--start", default=(datetime.now(timezone.utc) - timedelta(days=420)).isoformat().replace("+00:00", "Z"))
    ap.add_argument("--end", default=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "96")))
    ap.add_argument("--bucket", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    ap.add_argument("--bucket-minutes", type=int, default=0)
    ap.add_argument("--min-market-ratio", type=float, default=float(os.getenv("AUDIT_MIN_MARKET_RATIO", "0.98")))
    ap.add_argument("--min-feature-matrix-ratio", type=float, default=float(os.getenv("AUDIT_MIN_FEATURE_MATRIX_RATIO", "0.95")))
    ap.add_argument("--strict-schema-mismatch-zero", type=int, default=1)
    ap.add_argument("--min-text-coverage", type=float, default=float(os.getenv("AUDIT_MIN_TEXT_COVERAGE", "0.05")))
    ap.add_argument("--enforce-text-coverage", type=int, default=int(os.getenv("AUDIT_ENFORCE_TEXT_COVERAGE", "0")))
    ap.add_argument("--created-by", default=os.getenv("AUDIT_CREATED_BY", "manual"))
    ap.add_argument("--task-id", default="")
    ap.add_argument("--parallel-workers", type=int, default=int(os.getenv("AUDIT_PARALLEL_WORKERS", "4")))
    ap.add_argument("--output", default="artifacts/audit/top50_data_readiness_latest.json")
    ns = ap.parse_args()

    task_id = str(ns.task_id or "").strip() or f"audit-par-{uuid.uuid4().hex[:16]}"
    as_of = audit_mod._parse_dt(str(ns.as_of)) if str(ns.as_of).strip() else audit_mod._parse_dt(str(ns.start))
    bucket = str(ns.bucket).strip().lower()
    if int(ns.bucket_minutes) > 0:
        bucket = f"{int(ns.bucket_minutes)}m"

    symbols = audit_mod._parse_symbols(ns.symbols)
    if not symbols:
        symbols = _resolve_symbols(str(ns.database_url), str(ns.track).strip().lower() or "liquid", as_of, max(1, int(ns.top_n)))
    if not symbols:
        raise RuntimeError("audit_symbols_empty")

    base = audit_mod.AuditArgs(
        database_url=str(ns.database_url),
        track=str(ns.track).strip().lower() or "liquid",
        symbols=[],
        start=audit_mod._parse_dt(str(ns.start)),
        end=audit_mod._parse_dt(str(ns.end)),
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

    rows: Dict[str, Dict[str, Any]] = {}
    task_ids: List[str] = []
    workers = max(1, int(ns.parallel_workers))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_one, base, sym, i): sym for i, sym in enumerate(symbols, start=1)}
        for fut in as_completed(futs):
            sym = futs[fut]
            rec = fut.result()
            rows[str(sym).upper()] = dict(rec.get("row") or {})
            task_ids.append(str(rec.get("task_id") or ""))

    ready = [s for s, r in rows.items() if str((r or {}).get("status") or "").upper() == "READY"]
    degraded = [s for s, r in rows.items() if str((r or {}).get("status") or "").upper() == "DEGRADED"]
    blocked = [s for s, r in rows.items() if str((r or {}).get("status") or "").upper() == "BLOCKED"]
    summary = {"READY": len(ready), "DEGRADED": len(degraded), "BLOCKED": len(blocked)}

    out: Dict[str, Any] = {
        "task_id": task_id,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parallel_workers": workers,
        "child_task_ids": [x for x in task_ids if x],
        "track": str(base.track),
        "as_of": audit_mod._to_iso(as_of),
        "window": {
            "start": audit_mod._to_iso(base.start),
            "end": audit_mod._to_iso(base.end),
            "lookback": int(base.lookback),
            "bucket": str(base.bucket),
        },
        "symbols": sorted(rows.keys()),
        "symbol_readiness": rows,
        "summary": summary,
        "ready_symbols": sorted(ready),
        "degraded_symbols": sorted(degraded),
        "blocked_symbols": sorted(blocked),
        "gates": {
            "ready": len(blocked) == 0,
            "reasons": [f"blocked_symbols:{','.join(sorted(blocked))}"] if blocked else [],
        },
    }

    out_path = Path(str(ns.output))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": "ok",
                "task_id": task_id,
                "output": str(out_path),
                "symbol_count": len(rows),
                "ready": summary["READY"],
                "degraded": summary["DEGRADED"],
                "blocked": summary["BLOCKED"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
