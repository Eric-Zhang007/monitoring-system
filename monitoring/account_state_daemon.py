#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "backend"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from account_state.aggregator import AccountStateAggregator
from execution_engine import ExecutionEngine
from v2_repository import V2Repository


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def run_once(
    *,
    repo: V2Repository,
    aggregator: AccountStateAggregator,
    venue: str,
    adapter: str,
) -> Dict[str, Any]:
    refreshed = aggregator.refresh_full_state()
    open_orders = repo.list_open_orders_live(venue=venue, adapter=adapter, limit=200)
    for row in open_orders:
        aggregator.apply_order_event(
            {
                "client_order_id": row.get("client_order_id"),
                "venue_order_id": row.get("venue_order_id"),
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "qty": row.get("qty"),
                "filled_qty": row.get("filled_qty"),
                "status": row.get("status"),
            }
        )
    state = aggregator.get_state(require_fresh=False)
    return {
        "checked_at": _now_iso(),
        "venue": venue,
        "adapter": adapter,
        "equity": float(state.balances.equity),
        "free_margin": float(state.balances.free_margin),
        "margin_ratio": float(state.balances.margin_ratio),
        "positions": int(len(state.positions)),
        "open_orders": int(len(state.open_orders)),
        "is_fresh": bool(state.health.is_fresh),
        "refreshed_ts": refreshed.ts.isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Account state aggregation daemon")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--adapter", default=os.getenv("ACCOUNT_STATE_ADAPTER", "paper"))
    parser.add_argument("--venue", default=os.getenv("ACCOUNT_STATE_VENUE", "coinbase"))
    parser.add_argument("--ttl-sec", type=int, default=int(float(os.getenv("ACCOUNT_STATE_TTL_SEC", "10") or 10)))
    parser.add_argument("--refresh-sec", type=float, default=float(os.getenv("ACCOUNT_REFRESH_SEC", "30") or 30))
    parser.add_argument("--fast-sec", type=float, default=float(os.getenv("ACCOUNT_FAST_SEC", "2") or 2))
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    repo = V2Repository(args.database_url)
    engine = ExecutionEngine()
    adapter = engine.adapters.get(str(args.adapter))
    if adapter is None:
        raise RuntimeError(f"unsupported_adapter:{args.adapter}")

    agg = AccountStateAggregator(adapter=adapter, venue=str(args.venue), store=repo, cache_ttl_s=max(1, int(args.ttl_sec)))
    next_refresh = 0.0
    while True:
        now = time.monotonic()
        if now >= next_refresh:
            out = run_once(repo=repo, aggregator=agg, venue=str(args.venue), adapter=str(args.adapter))
            print(out, flush=True)
            next_refresh = now + max(1.0, float(args.refresh_sec))
        else:
            # fast path keeps order statuses fresh between full refreshes.
            for row in repo.list_open_orders_live(venue=str(args.venue), adapter=str(args.adapter), limit=200):
                agg.apply_order_event(dict(row))
        if not bool(args.loop):
            break
        time.sleep(max(0.2, float(args.fast_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
