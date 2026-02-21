from __future__ import annotations

import base64
import hashlib
import hmac
import json
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

from execution_policy import resolve_order_execution_context


class PaperExecutionAdapter:
    def __init__(self, reject_rate: float = 0.0, partial_fill_rate: float = 0.35):
        self.enable_random_reject = os.getenv("PAPER_ENABLE_RANDOM_REJECT", "0").strip().lower() in {"1", "true", "yes", "y"}
        self.reject_rate = reject_rate
        self.partial_fill_rate = partial_fill_rate
        self.timeout_reject_guard = float(os.getenv("PAPER_MAX_TIMEOUT_REJECT_RATE_GUARD", "0.0") or 0.0)
        self.timeout_by_symbol = self._parse_timeout_by_symbol(os.getenv("PAPER_TIMEOUT_BY_SYMBOL", "BTC=0.07,ETH=0.08,SOL=0.10"))

    @staticmethod
    def _parse_timeout_by_symbol(raw: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for token in (raw or "").split(","):
            t = token.strip()
            if not t or "=" not in t:
                continue
            k, v = t.split("=", 1)
            try:
                out[k.strip().upper()] = max(0.0, min(1.0, float(v.strip())))
            except Exception:
                continue
        return out

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)
        limit_timeout_sec = float(context.get("limit_timeout_sec", 2.0) or 2.0)
        max_retries = int(context.get("max_retries", 1) or 1)
        fee_bps = float(context.get("fee_bps", 5.0) or 5.0)
        lifecycle: List[Dict] = []
        if max_slippage_bps > 150.0:
            lifecycle.append({"event": "validation", "status": "rejected", "reason": "slippage_too_wide"})
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "slippage_too_wide",
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        if self.enable_random_reject and random.random() < self.reject_rate:
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
            target = str(order.get("target") or "").upper()
            base_timeout_prob = self.timeout_by_symbol.get(target, 0.1)
            timeout_prob = base_timeout_prob + max(0.0, self.timeout_reject_guard)
            if random.random() < timeout_prob and retries <= max_retries:
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


class BitgetLiveAdapter:
    def __init__(self):
        self.base_url = os.getenv("BITGET_BASE_URL", "https://api.bitget.com").strip().rstrip("/")
        self.api_key = os.getenv("BITGET_API_KEY", "").strip()
        self.api_secret = os.getenv("BITGET_API_SECRET", "").strip()
        self.api_passphrase = os.getenv("BITGET_API_PASSPHRASE", "").strip()
        self.timeout_sec = float(os.getenv("BITGET_TIMEOUT_SEC", "6") or 6.0)
        self.recv_window_ms = str(int(float(os.getenv("BITGET_RECV_WINDOW_MS", "5000") or 5000)))
        self.default_margin_coin = os.getenv("BITGET_DEFAULT_MARGIN_COIN", "USDT").strip().upper() or "USDT"

    def _sign(self, ts_ms: str, method: str, path: str, body: str) -> str:
        prehash = f"{ts_ms}{method.upper()}{path}{body}"
        digest = hmac.new(self.api_secret.encode("utf-8"), prehash.encode("utf-8"), hashlib.sha256).digest()
        return base64.b64encode(digest).decode("utf-8")

    def _request(self, method: str, path: str, *, payload: Optional[Dict[str, Any]] = None) -> tuple[int, Dict[str, Any]]:
        body_str = json.dumps(payload or {}, separators=(",", ":")) if payload is not None else ""
        ts_ms = str(int(time.time() * 1000))
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_secret and self.api_passphrase:
            headers.update(
                {
                    "ACCESS-KEY": self.api_key,
                    "ACCESS-SIGN": self._sign(ts_ms, method, path, body_str),
                    "ACCESS-TIMESTAMP": ts_ms,
                    "ACCESS-PASSPHRASE": self.api_passphrase,
                    "ACCESS-RECV-WINDOW": self.recv_window_ms,
                    "locale": "en-US",
                }
            )
        resp = requests.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            headers=headers,
            data=body_str if payload is not None else None,
            timeout=self.timeout_sec,
        )
        parsed: Dict[str, Any] = {}
        if resp.content:
            try:
                j = resp.json()
                if isinstance(j, dict):
                    parsed = j
            except Exception as exc:
                raise ValueError("bitget_response_parse_error") from exc
        return int(resp.status_code), parsed

    @staticmethod
    def _symbol(target: str, market_type: str) -> str:
        t = (target or "").strip().upper().replace("/", "").replace("-", "")
        if not t:
            return ""
        if market_type == "perp_usdt":
            return t if t.endswith("USDT") else f"{t}USDT"
        return t if t.endswith("USDT") else f"{t}USDT"

    @staticmethod
    def _classify_error(http_code: int, body: Dict[str, Any], fallback: str = "venue_error") -> str:
        if http_code == 429:
            return "bitget_rate_limited"
        code = str(body.get("code") or "")
        msg = str(body.get("msg") or body.get("message") or "").lower()
        if code in {"40009", "40011", "40012", "00171"} or "signature" in msg:
            return "bitget_signature_error"
        if "symbol" in msg and ("not exist" in msg or "invalid" in msg):
            return "bitget_symbol_not_supported"
        if "precision" in msg or "size too" in msg:
            return "bitget_precision_invalid"
        if "position" in msg or "reduceonly" in msg or "reduce only" in msg:
            return "bitget_position_rule_violation"
        if "rate limit" in msg or code in {"429", "13006"}:
            return "bitget_rate_limited"
        return fallback

    @staticmethod
    def _map_status(status: str, filled_qty: float, requested_qty: float) -> str:
        s = (status or "").strip().lower()
        if s in {"full-fill", "filled", "done"}:
            return "filled"
        if s in {"cancelled", "canceled"}:
            return "canceled" if filled_qty <= 1e-10 else "partially_filled"
        if s in {"reject", "rejected", "failed"}:
            return "rejected" if filled_qty <= 1e-10 else "partially_filled"
        if filled_qty >= requested_qty - 1e-10:
            return "filled"
        if filled_qty > 1e-10:
            return "partially_filled"
        return "submitted"

    def execute(self, order: Dict, context: Optional[Dict] = None) -> Dict:
        context = context or {}
        venue = str(context.get("venue") or "bitget").strip().lower()
        market_type = str(context.get("market_type") or "spot").strip().lower()
        product_type = str(context.get("product_type") or "USDT-FUTURES").strip()
        margin_mode = str(context.get("margin_mode") or "cross").strip().lower()
        position_mode = str(context.get("position_mode") or "one_way").strip().lower()
        reduce_only = bool(context.get("reduce_only") or False)
        leverage = context.get("leverage")
        lifecycle: List[Dict[str, Any]] = []
        if venue != "bitget":
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
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "bitget_credentials_not_configured",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }

        symbol = self._symbol(str(order.get("target") or ""), market_type)
        if not symbol:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "bitget_symbol_not_supported",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        side = "buy" if str(order.get("side") or "").lower() == "buy" else "sell"
        order_type = "limit" if float(order.get("est_price") or 0.0) > 0 else "market"
        tif = str(context.get("time_in_force") or "IOC").upper()
        client_oid = f"ms-{order.get('id') or order.get('decision_id') or uuid.uuid4().hex[:12]}"
        req: Dict[str, Any]
        path: str
        if market_type == "perp_usdt":
            path = "/api/v2/mix/order/place-order"
            req = {
                "symbol": symbol,
                "productType": product_type,
                "marginMode": margin_mode,
                "marginCoin": self.default_margin_coin,
                "side": side,
                "orderType": order_type,
                "size": f"{qty:.8f}",
                "force": tif,
                "clientOid": client_oid,
                "reduceOnly": "YES" if reduce_only else "NO",
                "tradeSide": "open" if not reduce_only else "close",
            }
            if order_type == "limit":
                req["price"] = f"{float(order.get('est_price')):.8f}"
            if leverage is not None:
                req["leverage"] = f"{float(leverage):.2f}"
            req["posMode"] = "hedge_mode" if position_mode == "hedge" else "one_way_mode"
        else:
            path = "/api/v2/spot/trade/place-order"
            req = {
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "size": f"{qty:.8f}",
                "force": tif,
                "clientOid": client_oid,
            }
            if order_type == "limit":
                req["price"] = f"{float(order.get('est_price')):.8f}"

        try:
            code, body = self._request("POST", path, payload=req)
            data = body.get("data") if isinstance(body.get("data"), dict) else {}
            order_id = str(data.get("orderId") or data.get("ordId") or "")
            lifecycle.append(
                {
                    "event": "submit",
                    "status": "accepted" if code < 400 and str(body.get("code") or "00000") == "00000" else "rejected",
                    "http_code": code,
                    "venue_order_id": order_id or None,
                }
            )
            if code >= 400 or str(body.get("code") or "00000") != "00000":
                return {
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_fill_price": None,
                    "reject_reason": self._classify_error(code, body),
                    "venue_order_id": order_id or None,
                    "fees_paid": 0.0,
                    "lifecycle": lifecycle,
                }

            # Bitget order-query endpoints differ by market; here we provide minimal polling and fallback.
            poll_path = "/api/v2/mix/order/detail" if market_type == "perp_usdt" else "/api/v2/spot/trade/orderInfo"
            poll_payload = {"symbol": symbol, "orderId": order_id}
            if market_type == "perp_usdt":
                poll_payload["productType"] = product_type
            pcode, pbody = self._request("POST", poll_path, payload=poll_payload)
            pdata = pbody.get("data") if isinstance(pbody.get("data"), dict) else {}
            if pcode >= 400 or str(pbody.get("code") or "00000") != "00000":
                lifecycle.append({"event": "poll", "status": "failed", "http_code": pcode})
                return {
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_fill_price": None,
                    "reject_reason": self._classify_error(pcode, pbody, fallback="no_fill_after_retries"),
                    "venue_order_id": order_id or None,
                    "fees_paid": 0.0,
                    "lifecycle": lifecycle,
                }
            filled_qty = float(pdata.get("baseVolume") or pdata.get("filledQty") or pdata.get("filledSize") or 0.0)
            avg_fill_price = pdata.get("priceAvg") or pdata.get("fillPrice") or pdata.get("avgPrice")
            avg_fill_price_val = float(avg_fill_price) if avg_fill_price not in (None, "") else float(order.get("est_price") or 0.0)
            status = self._map_status(str(pdata.get("status") or pdata.get("state") or "submitted"), filled_qty, qty)
            lifecycle.append(
                {
                    "event": "poll",
                    "status": status,
                    "filled_qty": round(filled_qty, 8),
                    "remaining_qty": round(max(0.0, qty - filled_qty), 8),
                    "venue_order_id": order_id or None,
                }
            )
            return {
                "status": status,
                "filled_qty": round(filled_qty, 8),
                "avg_fill_price": round(avg_fill_price_val, 8) if avg_fill_price_val > 0 else None,
                "reject_reason": None if status in {"filled", "partially_filled"} else "no_fill_after_retries",
                "venue_order_id": order_id or None,
                "remaining_qty": round(max(0.0, qty - filled_qty), 8),
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        except requests.Timeout:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": "bitget_transport_error:timeout",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        except requests.RequestException as exc:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"bitget_transport_error:{type(exc).__name__}",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }
        except Exception as exc:
            return {
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "reject_reason": f"bitget_unknown_error:{type(exc).__name__}",
                "venue_order_id": None,
                "fees_paid": 0.0,
                "lifecycle": lifecycle,
            }


class ExecutionEngine:
    def __init__(self):
        self.adapters = {
            "paper": PaperExecutionAdapter(),
            "coinbase_live": CoinbaseLiveAdapter(),
            "bitget_live": BitgetLiveAdapter(),
        }

    def run(self, adapter: str, orders: List[Dict], context: Optional[Dict] = None) -> List[Dict]:
        exec_adapter = self.adapters.get(adapter)
        if not exec_adapter:
            raise ValueError(f"unsupported adapter: {adapter}")
        out: List[Dict] = []
        for o in orders:
            merged_context = resolve_order_execution_context(o, context or {})
            res = exec_adapter.execute(o, context=merged_context)
            if isinstance(res, dict):
                res = {**res, "execution_policy": str(merged_context.get("execution_policy") or "")}
            out.append(res)
        return out
