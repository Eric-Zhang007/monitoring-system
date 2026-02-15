from __future__ import annotations

import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
try:
    import jwt
except Exception:
    jwt = None


class PaperExecutionAdapter:
    def __init__(self, reject_rate: float = 0.05, partial_fill_rate: float = 0.35):
        self.reject_rate = reject_rate
        self.partial_fill_rate = partial_fill_rate

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)
        limit_timeout_sec = float(context.get("limit_timeout_sec", 2.0) or 2.0)
        max_retries = int(context.get("max_retries", 1) or 1)
        fee_bps = float(context.get("fee_bps", 5.0) or 5.0)
        lifecycle: List[Dict] = []
        if random.random() < self.reject_rate:
            lifecycle.append({"event": "limit_submit", "status": "rejected", "reason": "paper_reject_simulated"})
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "paper_reject_simulated",
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        qty = float(order.get("quantity") or 0.0)
        if qty <= 0:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "invalid_quantity",
                "fees_paid": 0.0,
                "lifecycle": [{"event": "validation", "status": "rejected", "reason": "invalid_quantity"}],
            }
        px = float(order.get("est_price") or 0.0)
        side = order.get("side")
        limit_slip_bps = min(max_slippage_bps, 3.0)
        market_slip_bps = min(max_slippage_bps * 1.2, 8.0)

        fill_ratio = 1.0
        if random.random() < self.partial_fill_rate:
            fill_ratio = float(np.clip(random.uniform(0.3, 0.9), 0.0, 1.0))
        limit_filled = qty * fill_ratio
        remaining = max(0.0, qty - limit_filled)
        limit_px = px * (1.0 + ((limit_slip_bps / 10000.0) if side == "buy" else -(limit_slip_bps / 10000.0)))
        lifecycle.append(
            {
                "event": "limit_submit",
                "status": "partially_filled" if remaining > 0 else "filled",
                "filled_qty": round(limit_filled, 8),
                "remaining_qty": round(remaining, 8),
                "avg_fill_price": round(limit_px, 8),
                "timeout_sec": limit_timeout_sec,
            }
        )

        retries = 0
        market_filled = 0.0
        market_px = 0.0
        while remaining > 1e-10 and retries <= max_retries:
            retries += 1
            lifecycle.append({"event": "cancel_limit", "status": "ok", "retry": retries})
            if random.random() < 0.1 and retries <= max_retries:
                lifecycle.append({"event": "market_submit", "status": "retry", "retry": retries, "reason": "venue_timeout_sim"})
                continue
            market_filled = remaining
            market_px = px * (1.0 + ((market_slip_bps / 10000.0) if side == "buy" else -(market_slip_bps / 10000.0)))
            remaining = 0.0
            lifecycle.append(
                {
                    "event": "market_submit",
                    "status": "filled",
                    "filled_qty": round(market_filled, 8),
                    "avg_fill_price": round(market_px, 8),
                    "retry": retries,
                }
            )
            break

        total_filled = limit_filled + market_filled
        if total_filled <= 1e-10:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "no_fill_after_retries",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        avg_fill_price = ((limit_filled * limit_px) + (market_filled * market_px)) / max(total_filled, 1e-12)
        fees_paid = total_filled * avg_fill_price * (fee_bps / 10000.0)
        final_status = "filled" if remaining <= 1e-10 else "partially_filled"
        return {
            "status": final_status,
            "filled_qty": round(total_filled, 8),
            "avg_fill_price": round(avg_fill_price, 8),
            "reject_reason": None if final_status != "partially_filled" else "residual_unfilled",
            "venue_order_id": None,
            "remaining_qty": round(remaining, 8),
            "fees_paid": round(fees_paid, 8),
            "lifecycle": lifecycle,
        }


class CoinbaseLiveAdapter:
    def __init__(self):
        self.base_url = os.getenv("COINBASE_ADVANCED_TRADE_API", "https://api.coinbase.com").strip().rstrip("/")
        self.api_key_name = os.getenv("COINBASE_API_KEY", "").strip()
        self.api_private_key = os.getenv("COINBASE_API_SECRET", "").replace("\\n", "\n").strip()
        self.timeout_sec = float(os.getenv("COINBASE_EXEC_TIMEOUT_SEC", "5"))
        self.quote_currency = os.getenv("COINBASE_QUOTE_CURRENCY", "USD").strip().upper() or "USD"

    def _build_jwt(self, method: str, path: str) -> str:
        if jwt is None:
            raise RuntimeError("pyjwt_not_installed")
        now = int(time.time())
        host = urlparse(self.base_url).netloc or "api.coinbase.com"
        claims = {
            "iss": "cdp",
            "nbf": now,
            "exp": now + 120,
            "sub": self.api_key_name,
            "uri": f"{method.upper()} {host}{path}",
        }
        headers = {
            "kid": self.api_key_name,
            "nonce": uuid.uuid4().hex,
        }
        return str(jwt.encode(claims, self.api_private_key, algorithm="ES256", headers=headers))

    def _request(self, method: str, path: str, *, payload: Optional[Dict] = None, params: Optional[Dict] = None) -> tuple[int, Dict[str, Any]]:
        token = self._build_jwt(method, path)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        resp = requests.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            params=params,
            timeout=self.timeout_sec,
        )
        body: Dict[str, Any] = {}
        if resp.content:
            try:
                parsed = resp.json()
                if isinstance(parsed, dict):
                    body = parsed
            except Exception:
                body = {}
        return int(resp.status_code), body

    @staticmethod
    def _to_product_id(target: str, quote_currency: str) -> str:
        t = (target or "").strip().upper()
        if "-" in t:
            return t
        return f"{t}-{quote_currency}"

    @staticmethod
    def _extract_order_snapshot(body: Dict[str, Any]) -> Dict[str, Any]:
        success_resp = body.get("success_response") if isinstance(body.get("success_response"), dict) else {}
        error_resp = body.get("error_response") if isinstance(body.get("error_response"), dict) else {}
        order = body.get("order") if isinstance(body.get("order"), dict) else {}
        if not order and isinstance(body.get("order_configuration"), dict):
            order = body
        if not order:
            order = body
        status = str(order.get("status") or "").upper()
        filled_qty = float(order.get("filled_size") or order.get("filled_quantity") or 0.0)
        avg_fill_price = order.get("average_filled_price") or order.get("avg_fill_price") or None
        if avg_fill_price is not None:
            try:
                avg_fill_price = float(avg_fill_price)
            except Exception:
                avg_fill_price = None
        fees_paid = float(order.get("total_fees") or order.get("filled_fees") or 0.0)
        return {
            "order_id": success_resp.get("order_id") or order.get("order_id") or order.get("id") or body.get("order_id"),
            "status": status,
            "filled_qty": filled_qty,
            "avg_fill_price": avg_fill_price,
            "fees_paid": fees_paid,
            "reject_reason": str(
                order.get("reject_reason")
                or error_resp.get("message")
                or error_resp.get("error")
                or body.get("error")
                or ""
            ),
        }

    @staticmethod
    def _map_status(status: str, filled_qty: float, requested_qty: float) -> str:
        s = status.upper()
        if s in {"FILLED", "DONE"}:
            return "filled"
        if s in {"CANCELLED", "CANCELED"}:
            return "canceled" if filled_qty <= 1e-10 else "partially_filled"
        if s in {"EXPIRED", "FAILED", "REJECTED"}:
            return "rejected" if filled_qty <= 1e-10 else "partially_filled"
        if filled_qty > 1e-10 and filled_qty + 1e-10 < requested_qty:
            return "partially_filled"
        if filled_qty >= requested_qty - 1e-10:
            return "filled"
        return "submitted"

    def _place_limit_order(self, order: Dict, context: Dict, product_id: str) -> tuple[int, Dict[str, Any], Dict[str, Any]]:
        qty = float(order.get("quantity") or 0.0)
        side = "BUY" if str(order.get("side") or "").lower() == "buy" else "SELL"
        tif = str(context.get("time_in_force") or "GTC").upper()
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)
        est_price = float(order.get("est_price") or 0.0)
        slip = min(max_slippage_bps, 2.0) / 10000.0
        if est_price <= 0:
            raise ValueError("invalid_est_price")
        limit_price = est_price * (1.0 + slip if side == "BUY" else 1.0 - slip)
        cfg_key = "limit_limit_gtc"
        if tif == "IOC":
            cfg_key = "limit_limit_ioc"
        elif tif == "FOK":
            cfg_key = "limit_limit_fok"
        order_key = str(order.get("id") or order.get("order_id") or order.get("decision_id") or uuid.uuid4().hex[:12])
        body = {
            "client_order_id": f"ms-{order_key}-l",
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                cfg_key: {
                    "base_size": f"{qty:.8f}",
                    "limit_price": f"{limit_price:.8f}",
                    "post_only": False,
                }
            },
        }
        code, resp = self._request("POST", "/api/v3/brokerage/orders", payload=body)
        return code, resp, body

    def _place_market_order(self, order: Dict, qty: float, product_id: str, retry_tag: int) -> tuple[int, Dict[str, Any], Dict[str, Any]]:
        side = "BUY" if str(order.get("side") or "").lower() == "buy" else "SELL"
        order_key = str(order.get("id") or order.get("order_id") or order.get("decision_id") or uuid.uuid4().hex[:12])
        body = {
            "client_order_id": f"ms-{order_key}-m{retry_tag}",
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": f"{qty:.8f}",
                }
            },
        }
        code, resp = self._request("POST", "/api/v3/brokerage/orders", payload=body)
        return code, resp, body

    def _cancel_order(self, order_id: str) -> tuple[int, Dict[str, Any]]:
        body = {"order_ids": [order_id]}
        return self._request("POST", "/api/v3/brokerage/orders/batch_cancel", payload=body)

    def _poll_order(self, order_id: str, timeout_sec: float, poll_interval: float) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.2, timeout_sec)
        latest: Dict[str, Any] = {"order_id": order_id, "status": "UNKNOWN", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0, "reject_reason": ""}
        while time.monotonic() <= deadline:
            code, body = self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")
            if code >= 400:
                time.sleep(max(0.1, poll_interval))
                continue
            latest = self._extract_order_snapshot(body)
            mapped = self._map_status(str(latest.get("status") or ""), float(latest.get("filled_qty") or 0.0), 1.0)
            if mapped in {"filled", "rejected", "partially_filled"}:
                return latest
            time.sleep(max(0.1, poll_interval))
        return latest

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        venue = str(context.get("venue", "coinbase") or "coinbase").lower()
        limit_timeout_sec = float(context.get("limit_timeout_sec", 2.0) or 2.0)
        max_retries = int(context.get("max_retries", 1) or 1)
        poll_interval = float(context.get("poll_interval_sec", 0.4) or 0.4)
        lifecycle: List[Dict[str, Any]] = []

        if venue != "coinbase":
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"unsupported_venue:{venue}",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }

        qty = float(order.get("quantity") or 0.0)
        if qty <= 0:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "invalid_quantity",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": [{"event": "validation", "status": "rejected", "reason": "invalid_quantity"}],
            }

        if not self.api_key_name or not self.api_private_key:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "coinbase_credentials_not_configured",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }

        product_id = self._to_product_id(str(order.get("target") or ""), self.quote_currency)
        total_filled = 0.0
        avg_fill_price: Optional[float] = None
        fees_paid = 0.0
        last_order_id: Optional[str] = None
        remaining = qty

        try:
            # Step 1: place limit first when est_price is present.
            placed_limit = False
            if float(order.get("est_price") or 0.0) > 0:
                placed_limit = True
                code, resp, req_body = self._place_limit_order(order, context, product_id)
                snap = self._extract_order_snapshot(resp)
                last_order_id = str(snap.get("order_id") or "") or None
                lifecycle.append(
                    {
                        "event": "limit_submit",
                        "status": "accepted" if code < 400 else "rejected",
                        "http_code": code,
                        "venue_order_id": last_order_id,
                        "request": req_body,
                    }
                )
                if code >= 400 or not last_order_id:
                    return {
                        "status": "rejected",
                        "filled_qty": 0.0,
                        "avg_fill_price": None,
                        "reject_reason": f"venue_http_{code}",
                        "venue_order_id": last_order_id,
                        "fees_paid": 0.0,
                        "lifecycle": lifecycle,
                    }

                polled = self._poll_order(last_order_id, timeout_sec=limit_timeout_sec, poll_interval=poll_interval)
                limit_fill = float(polled.get("filled_qty") or 0.0)
                limit_px = polled.get("avg_fill_price")
                if limit_px is not None:
                    limit_px = float(limit_px)
                total_filled += limit_fill
                remaining = max(0.0, qty - total_filled)
                fees_paid += float(polled.get("fees_paid") or 0.0)
                if limit_px is not None and limit_fill > 0:
                    avg_fill_price = limit_px
                lifecycle.append(
                    {
                        "event": "limit_poll",
                        "status": str(polled.get("status") or "UNKNOWN").lower(),
                        "filled_qty": round(limit_fill, 8),
                        "remaining_qty": round(remaining, 8),
                        "venue_order_id": last_order_id,
                    }
                )

                if remaining > 1e-10:
                    c_code, c_resp = self._cancel_order(last_order_id)
                    lifecycle.append(
                        {
                            "event": "cancel_limit",
                            "status": "ok" if c_code < 400 else "failed",
                            "http_code": c_code,
                            "response": c_resp,
                            "venue_order_id": last_order_id,
                        }
                    )

            # Step 2: route residual to market IOC with retries.
            retries = 0
            while remaining > 1e-10 and retries <= max_retries:
                retries += 1
                m_code, m_resp, m_req = self._place_market_order(order, remaining, product_id, retries)
                m_snap = self._extract_order_snapshot(m_resp)
                m_order_id = str(m_snap.get("order_id") or "") or None
                if m_order_id:
                    last_order_id = m_order_id
                m_fill = float(m_snap.get("filled_qty") or 0.0)
                m_px = m_snap.get("avg_fill_price")
                if m_px is not None:
                    m_px = float(m_px)

                lifecycle.append(
                    {
                        "event": "market_submit",
                        "status": "accepted" if m_code < 400 else "retry",
                        "retry": retries,
                        "http_code": m_code,
                        "venue_order_id": m_order_id,
                        "request": m_req,
                    }
                )

                if m_code >= 400:
                    continue

                if m_order_id:
                    polled_market = self._poll_order(m_order_id, timeout_sec=max(0.5, limit_timeout_sec), poll_interval=poll_interval)
                    m_fill = float(polled_market.get("filled_qty") or m_fill)
                    m_px = polled_market.get("avg_fill_price") if polled_market.get("avg_fill_price") is not None else m_px
                    lifecycle.append(
                        {
                            "event": "market_poll",
                            "status": str(polled_market.get("status") or "UNKNOWN").lower(),
                            "retry": retries,
                            "filled_qty": round(m_fill, 8),
                            "venue_order_id": m_order_id,
                        }
                    )

                old_filled = total_filled
                total_filled += m_fill
                fees_paid += float(m_snap.get("fees_paid") or 0.0)
                remaining = max(0.0, qty - total_filled)
                if m_px is not None and m_fill > 0:
                    if avg_fill_price is None or old_filled <= 1e-10:
                        avg_fill_price = float(m_px)
                    else:
                        avg_fill_price = ((avg_fill_price * old_filled) + (float(m_px) * m_fill)) / max(total_filled, 1e-12)

            if total_filled <= 1e-10:
                return {
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_fill_price": None,
                    "reject_reason": "no_fill_after_retries",
                    "venue_order_id": last_order_id,
                    "remaining_qty": round(qty, 8),
                    "fees_paid": round(fees_paid, 8),
                    "lifecycle": lifecycle,
                }

            status = "filled" if remaining <= 1e-10 else "partially_filled"
            reject_reason = None if status == "filled" else "residual_unfilled"
            if not placed_limit and status == "partially_filled":
                reject_reason = "market_partial_fill"
            return {
                "status": status,
                "filled_qty": round(total_filled, 8),
                "avg_fill_price": round(float(avg_fill_price), 8) if avg_fill_price is not None else None,
                "reject_reason": reject_reason,
                "venue_order_id": last_order_id,
                "remaining_qty": round(remaining, 8),
                "fees_paid": round(fees_paid, 8),
                "lifecycle": lifecycle,
            }
        except Exception as exc:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"venue_error:{type(exc).__name__}",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
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
