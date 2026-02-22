#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from execution_engine import ExecutionEngine
from v2_repository import V2Repository
from metrics import EXEC_RECON_DRIFT_EVENTS_TOTAL


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _position_diff_usd(repo: V2Repository, venue: str, remote: List[Dict[str, Any]], local: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
    remote_map: Dict[str, float] = {}
    for r in remote:
        symbol = str(r.get("symbol") or "").upper()
        if not symbol:
            continue
        remote_map[symbol] = remote_map.get(symbol, 0.0) + _safe_float(r.get("position_qty"), 0.0)
    local_map: Dict[str, float] = {}
    for l in local:
        symbol = str(l.get("symbol") or "").upper()
        if not symbol:
            continue
        local_map[symbol] = local_map.get(symbol, 0.0) + _safe_float(l.get("position_qty"), 0.0)

    diff_total = 0.0
    by_symbol: Dict[str, float] = {}
    for symbol in sorted(set(remote_map.keys()) | set(local_map.keys())):
        dq = abs(remote_map.get(symbol, 0.0) - local_map.get(symbol, 0.0))
        if dq <= 1e-12:
            continue
        px = repo.latest_price_snapshot(symbol) or {}
        price = _safe_float(px.get("price"), 0.0)
        diff_usd = dq * max(0.0, price)
        by_symbol[symbol] = diff_usd
        diff_total += diff_usd
    return float(diff_total), {"venue": venue, "by_symbol_usd": by_symbol, "checked_at": _now_iso()}


def run_once(repo: V2Repository, engine: ExecutionEngine, max_age_sec: int, pos_diff_usd_max: float, fail_triggers_kill: bool) -> Dict[str, Any]:
    pending = repo.list_open_child_orders_live(max_age_sec=max_age_sec)
    by_adapter: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in pending:
        by_adapter[str(row.get("adapter") or "")].append(row)

    actions: List[Dict[str, Any]] = []
    fail_count = 0
    for adapter_name, rows in by_adapter.items():
        adapter = engine.adapters.get(adapter_name)
        if adapter is None:
            continue
        for row in rows:
            child_id = int(row["id"])
            venue_order_id = str(row.get("venue_order_id") or "")
            if not venue_order_id:
                continue
            try:
                polled = adapter.poll_order(venue_order_id, timeout=1.5)
                status = str(polled.get("status") or "submitted")
                repo.update_child_order_status(
                    child_id,
                    status=status,
                    venue_order_id=venue_order_id,
                    lifecycle_append={"event": "recon_poll", "status": status, "time": _now_iso(), "metrics": {"filled_qty": _safe_float(polled.get("filled_qty"), 0.0)}},
                )
                fills = adapter.fetch_fills(venue_order_id)
                repo.insert_execution_fills(child_id, fills)
                repo.update_parent_from_fills(int(row.get("parent_order_id")))
                actions.append(
                    {
                        "action": "poll_and_sync",
                        "adapter": adapter_name,
                        "child_order_id": child_id,
                        "venue_order_id": venue_order_id,
                        "status": status,
                        "fills": len(fills),
                    }
                )
            except Exception as exc:
                fail_count += 1
                actions.append(
                    {
                        "action": "poll_failed",
                        "adapter": adapter_name,
                        "child_order_id": child_id,
                        "venue_order_id": venue_order_id,
                        "error": f"{type(exc).__name__}:{exc}",
                    }
                )

    drift_events = 0
    for adapter_name, rows in by_adapter.items():
        if not rows:
            continue
        venue = str(rows[0].get("venue") or "unknown")
        adapter = engine.adapters.get(adapter_name)
        if adapter is None:
            continue
        try:
            remote_pos = adapter.fetch_positions()
        except Exception:
            remote_pos = []
        local_pos = repo.list_positions_live(venue=venue)
        diff_usd, diff_payload = _position_diff_usd(repo, venue, remote_pos, local_pos)
        status = "ok"
        if diff_usd > pos_diff_usd_max:
            status = "drift_exceeded"
            drift_events += 1
            EXEC_RECON_DRIFT_EVENTS_TOTAL.labels(venue=venue).inc()
            if fail_triggers_kill:
                repo.upsert_kill_switch_state(
                    track="liquid",
                    strategy_id="global",
                    state="triggered",
                    reason="reconciliation_drift_exceeded",
                    duration_minutes=30,
                    metadata={"venue": venue, "diff_usd": diff_usd},
                )
        repo.append_reconciliation_log(
            venue=venue,
            adapter=adapter_name,
            decision_id=None,
            open_orders_diff={"pending_count": len(rows), "max_age_sec": max_age_sec},
            positions_diff={**diff_payload, "diff_usd_total": diff_usd},
            actions_taken=list(actions),
            status=status,
            error=None,
        )

    if fail_count > 0 and fail_triggers_kill:
        repo.upsert_kill_switch_state(
            track="liquid",
            strategy_id="global",
            state="triggered",
            reason="reconciliation_poll_failures",
            duration_minutes=30,
            metadata={"failures": fail_count},
        )

    return {
        "checked_at": _now_iso(),
        "pending": len(pending),
        "actions": len(actions),
        "poll_failures": fail_count,
        "drift_events": drift_events,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="OMS reconciliation daemon")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("RECON_INTERVAL_SEC", "30")))
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    repo = V2Repository(args.database_url)
    engine = ExecutionEngine()
    max_age_sec = int(float(os.getenv("RECON_OPEN_ORDERS_MAX_AGE_SEC", "300") or 300))
    pos_diff_usd_max = float(os.getenv("RECON_POS_DIFF_USD_MAX", "100.0") or 100.0)
    fail_triggers_kill = _env_flag("RECON_FAILS_TRIGGER_KILL_SWITCH", "1")

    if args.loop:
        while True:
            out = run_once(repo, engine, max_age_sec=max_age_sec, pos_diff_usd_max=pos_diff_usd_max, fail_triggers_kill=fail_triggers_kill)
            print(out, flush=True)
            time.sleep(max(1.0, float(args.interval_sec)))
    else:
        out = run_once(repo, engine, max_age_sec=max_age_sec, pos_diff_usd_max=pos_diff_usd_max, fail_triggers_kill=fail_triggers_kill)
        print(out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
