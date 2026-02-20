#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import List

from _psql import run_psql


def _run_psql(sql: str) -> str:
    return run_psql(sql)


def _completed_ids(track: str) -> List[int]:
    sql = (
        "SELECT id::text FROM backtest_runs "
        f"WHERE track='{track}' "
        "AND COALESCE(metrics->>'status','')='completed' "
        "AND superseded_by_run_id IS NULL "
        "ORDER BY created_at DESC;"
    )
    rows = [r.strip() for r in _run_psql(sql).splitlines() if r.strip()]
    out: List[int] = []
    for r in rows:
        try:
            out.append(int(r))
        except ValueError:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Supersede stale completed backtest runs while keeping recent baselines")
    ap.add_argument("--track", default="liquid")
    ap.add_argument("--keep-latest", type=int, default=30)
    ap.add_argument("--reason", default="historical_baseline_superseded")
    args = ap.parse_args()

    ids = _completed_ids(args.track)
    if len(ids) <= args.keep_latest:
        print(json.dumps({"track": args.track, "kept": len(ids), "superseded": 0, "anchor_run_id": ids[0] if ids else None}, ensure_ascii=False))
        return 0

    keep_ids = ids[: args.keep_latest]
    stale_ids = ids[args.keep_latest :]
    anchor = keep_ids[0]
    sql = (
        "UPDATE backtest_runs "
        f"SET superseded_by_run_id={anchor}, supersede_reason='{args.reason}', superseded_at=NOW() "
        f"WHERE id IN ({','.join(str(i) for i in stale_ids)});"
    )
    _run_psql(sql)
    print(
        json.dumps(
            {
                "track": args.track,
                "kept": len(keep_ids),
                "superseded": len(stale_ids),
                "anchor_run_id": anchor,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
