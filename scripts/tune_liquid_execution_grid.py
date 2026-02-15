#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
from typing import Any, Dict, List

import requests


def _mk_orders_payload(decision_id: str) -> Dict[str, Any]:
    return {
        "decision_id": decision_id,
        "adapter": "paper",
        "time_in_force": "IOC",
        "venue": "coinbase",
        "max_slippage_bps": 20.0,
        "market_type": "spot",
        "product_type": "USDT-FUTURES",
        "reduce_only": False,
        "position_mode": "one_way",
        "margin_mode": "cross",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid tune execution retry/timeout parameters for liquid")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--decision-id", required=True)
    ap.add_argument("--timeout-grid", default="2,3,4")
    ap.add_argument("--retry-grid", default="1,2,3")
    ap.add_argument("--slippage-grid", default="15,20,25")
    args = ap.parse_args()

    timeouts = [float(x) for x in args.timeout_grid.split(",") if x.strip()]
    retries = [int(float(x)) for x in args.retry_grid.split(",") if x.strip()]
    slippages = [float(x) for x in args.slippage_grid.split(",") if x.strip()]

    results: List[Dict[str, Any]] = []
    for timeout_sec, max_retries, slippage_bps in itertools.product(timeouts, retries, slippages):
        payload = _mk_orders_payload(args.decision_id)
        payload["limit_timeout_sec"] = timeout_sec
        payload["max_retries"] = max_retries
        payload["max_slippage_bps"] = slippage_bps
        try:
            resp = requests.post(f"{args.api_base}/api/v2/execution/run", json=payload, timeout=30)
            if resp.status_code >= 300:
                results.append(
                    {
                        "timeout_sec": timeout_sec,
                        "max_retries": max_retries,
                        "slippage_bps": slippage_bps,
                        "status": "http_error",
                        "http_status": resp.status_code,
                    }
                )
                continue
            body = resp.json()
            total = int(body.get("total") or 0)
            rejected = int(body.get("rejected") or 0)
            reject_rate = (rejected / total) if total else 1.0
            results.append(
                {
                    "timeout_sec": timeout_sec,
                    "max_retries": max_retries,
                    "slippage_bps": slippage_bps,
                    "status": "ok",
                    "total": total,
                    "rejected": rejected,
                    "reject_rate": round(reject_rate, 6),
                    "reject_breakdown": body.get("reject_breakdown") or {},
                }
            )
        except Exception as exc:
            results.append(
                {
                    "timeout_sec": timeout_sec,
                    "max_retries": max_retries,
                    "slippage_bps": slippage_bps,
                    "status": "error",
                    "error": type(exc).__name__,
                }
            )

    ok_rows = [r for r in results if r.get("status") == "ok"]
    ok_rows.sort(key=lambda x: (x.get("reject_rate", 1.0), x.get("rejected", 1_000_000)))

    out = {
        "trials": len(results),
        "best": ok_rows[:5],
        "results": results,
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ok_rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
