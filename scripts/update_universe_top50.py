#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.v2_repository import V2Repository
from training.universe.top50 import UniverseBuildConfig, build_top_universe_snapshot, write_universe_snapshot


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build and persist a stable top50 liquid universe snapshot")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--track", default=os.getenv("LIQUID_UNIVERSE_TRACK", "liquid"))
    ap.add_argument("--as-of", default="")
    ap.add_argument("--top-n", type=int, default=int(os.getenv("LIQUID_UNIVERSE_TOP_N", "50")))
    ap.add_argument(
        "--rank-by",
        choices=["volume_usd_30d", "adv_usd_30d", "notional_usd_30d"],
        default=os.getenv("LIQUID_UNIVERSE_RANK_BY", "volume_usd_30d"),
    )
    ap.add_argument("--source", default=os.getenv("LIQUID_UNIVERSE_SOURCE", "db"))
    ap.add_argument("--lookback-days", type=int, default=int(os.getenv("LIQUID_UNIVERSE_LOOKBACK_DAYS", "30")))
    ap.add_argument("--timeframe", default=os.getenv("LIQUID_UNIVERSE_TIMEFRAME", "1h"))
    ap.add_argument("--min-notional-usd", type=float, default=float(os.getenv("LIQUID_UNIVERSE_MIN_NOTIONAL_USD", "1000000")))
    ap.add_argument("--exclude-stable", type=int, default=int(os.getenv("LIQUID_UNIVERSE_EXCLUDE_STABLE", "1")))
    ap.add_argument("--exclude-leveraged", type=int, default=int(os.getenv("LIQUID_UNIVERSE_EXCLUDE_LEVERAGED", "1")))
    ap.add_argument("--hysteresis-keep-rank", type=int, default=int(os.getenv("LIQUID_UNIVERSE_HYSTERESIS_KEEP_RANK", "60")))
    ap.add_argument("--snapshot-file", default=os.getenv("LIQUID_UNIVERSE_SNAPSHOT_FILE", "artifacts/universe/liquid_top50_snapshot.json"))
    ap.add_argument("--persist-db", type=int, default=int(os.getenv("LIQUID_UNIVERSE_PERSIST_DB", "1")))
    args = ap.parse_args()

    as_of = _parse_ts(str(args.as_of))
    snapshot_file = Path(str(args.snapshot_file))
    cfg = UniverseBuildConfig(
        as_of=as_of,
        top_n=max(1, int(args.top_n)),
        lookback_days=max(1, int(args.lookback_days)),
        timeframe=str(args.timeframe).strip() or "1h",
        min_notional_usd=max(0.0, float(args.min_notional_usd)),
        exclude_stable=bool(int(args.exclude_stable)),
        exclude_leveraged=bool(int(args.exclude_leveraged)),
        hysteresis_keep_rank=max(1, int(args.hysteresis_keep_rank)),
        track=str(args.track).strip().lower() or "liquid",
        rank_by=str(args.rank_by).strip().lower(),
        source=str(args.source).strip().lower() or "db",
    )

    snapshot = build_top_universe_snapshot(
        database_url=str(args.database_url),
        cfg=cfg,
        previous_snapshot_file=snapshot_file,
    )
    out = write_universe_snapshot(snapshot, snapshot_file)

    db_written = False
    if bool(int(args.persist_db)):
        repo = V2Repository(str(args.database_url))
        repo.upsert_asset_universe_snapshot(
            track=str(cfg.track),
            as_of=as_of,
            symbols=list(snapshot["symbols"]),
            universe_version=str(snapshot["snapshot_hash"]),
            source=str(snapshot["source"]),
        )
        db_written = True

    print(
        json.dumps(
            {
                "status": "ok",
                "snapshot_file": str(out),
                "snapshot_hash": snapshot["snapshot_hash"],
                "as_of": snapshot["as_of"],
                "symbols": snapshot["symbols"],
                "db_written": db_written,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
