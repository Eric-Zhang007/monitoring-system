from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import requests


class PaperExecutionAdapter:
    def __init__(self, reject_rate: float = 0.05):
        self.reject_rate = reject_rate

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)
        if random.random() < self.reject_rate:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "paper_reject_simulated",
            }
        qty = float(order.get("quantity") or 0.0)
        px = order.get("est_price")
        if px is None:
            px = 0.0
        slip_bps = min(max_slippage_bps, 5.0)
        slip = 1.0 + ((slip_bps / 10000.0) if order.get("side") == "buy" else -(slip_bps / 10000.0))
        return {
            "status": "filled",
            "filled_qty": qty,
            "avg_fill_price": float(px) * slip,
            "reject_reason": None,
            "venue_order_id": None,
        }


class CoinbaseLiveAdapter:
    def __init__(self):
        self.endpoint = os.getenv("COINBASE_EXECUTION_ENDPOINT", "").strip()
        self.api_key = os.getenv("COINBASE_API_KEY", "").strip()
        self.timeout_sec = float(os.getenv("COINBASE_EXEC_TIMEOUT_SEC", "5"))

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        tif = context.get("time_in_force", "IOC")
        venue = context.get("venue", "coinbase")
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)

        if venue != "coinbase":
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"unsupported_venue:{venue}",
                "venue_order_id": None,
            }

        if not self.endpoint:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "coinbase_endpoint_not_configured",
                "venue_order_id": None,
            }

        payload = {
            "symbol": str(order.get("target") or "").upper(),
            "side": order.get("side"),
            "quantity": float(order.get("quantity") or 0.0),
            "est_price": order.get("est_price"),
            "time_in_force": tif,
            "max_slippage_bps": max_slippage_bps,
            "client_order_id": f"ms-{order.get('id', 'na')}",
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_sec,
            )
            if resp.status_code >= 400:
                return {
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_fill_price": None,
                    "reject_reason": f"venue_http_{resp.status_code}",
                    "venue_order_id": None,
                }
            data = resp.json() if resp.content else {}
            status = str(data.get("status", "submitted")).lower()
            filled_qty = float(data.get("filled_qty", 0.0) or 0.0)
            avg_fill_price = data.get("avg_fill_price")
            if avg_fill_price is not None:
                avg_fill_price = float(avg_fill_price)
            mapped = "filled" if status in {"filled", "done"} else "rejected" if status in {"rejected", "failed"} else "submitted"
            return {
                "status": mapped,
                "filled_qty": filled_qty,
                "avg_fill_price": avg_fill_price,
                "reject_reason": None if mapped != "rejected" else str(data.get("reject_reason") or "venue_rejected"),
                "venue_order_id": data.get("venue_order_id") or data.get("order_id"),
            }
        except Exception as exc:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"venue_error:{type(exc).__name__}",
                "venue_order_id": None,
            }


class ExecutionEngine:
    def __init__(self):
        self.adapters = {
            "paper": PaperExecutionAdapter(),
            "coinbase_live": CoinbaseLiveAdapter(),
        }

    def run(self, adapter: str, orders: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        exec_adapter = self.adapters.get(adapter)
        if not exec_adapter:
            raise ValueError(f"unsupported adapter: {adapter}")
        return [exec_adapter.execute(o, context=context) for o in orders]
