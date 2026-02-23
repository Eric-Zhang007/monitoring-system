from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
import random
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import jwt
except Exception:
    jwt = None

from execution_adapters.base import ExecutionAdapterBase
from execution_fsm import transition_child, transition_decision, transition_parent
from execution_planner import build_execution_plan
from execution_policy import build_execution_trace, resolve_order_execution_context
from position_accounting import apply_fills_to_position, compute_realized_pnl, compute_unrealized_pnl

logger = logging.getLogger(__name__)


class PaperExecutionAdapter(ExecutionAdapterBase):
    name = "paper"

    def __init__(self, reject_rate: float = 0.0, partial_fill_rate: float = 0.35):
        self.enable_random_reject = os.getenv("PAPER_ENABLE_RANDOM_REJECT", "0").strip().lower() in {"1", "true", "yes", "y"}
        self.reject_rate = reject_rate
        self.partial_fill_rate = partial_fill_rate
        self.timeout_reject_guard = float(os.getenv("PAPER_MAX_TIMEOUT_REJECT_RATE_GUARD", "0.0") or 0.0)
        self.timeout_by_symbol = self._parse_timeout_by_symbol(os.getenv("PAPER_TIMEOUT_BY_SYMBOL", "BTC=0.07,ETH=0.08,SOL=0.10"))
        self.seed = int(os.getenv("PAPER_SEED", "20260222") or 20260222)
        self._orders: Dict[str, Dict[str, Any]] = {}
        self._fills_by_order: Dict[str, List[Dict[str, Any]]] = {}
        self._positions: Dict[str, Dict[str, float]] = {}

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

    @staticmethod
    def _utcnow() -> datetime:
        return datetime.now(timezone.utc)

    def _rng(self, key: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{key}".encode("utf-8")).hexdigest()[:16]
        return random.Random(int(digest, 16))

    def prepare(self, symbol: str) -> Dict[str, Any]:
        return {
            "symbol": str(symbol or "").upper(),
            "min_qty": float(os.getenv("PAPER_MIN_QTY", "0.000001") or 0.000001),
            "qty_step": float(os.getenv("PAPER_QTY_STEP", "0.000001") or 0.000001),
        }

    def submit_order(self, child_order: Dict[str, Any], context: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
        symbol = str(child_order.get("symbol") or "").upper()
        side = str(child_order.get("side") or "buy").lower()
        qty = float(child_order.get("qty") or 0.0)
        client_order_id = str(child_order.get("client_order_id") or "").strip()
        if not client_order_id:
            client_order_id = f"paper:{uuid.uuid4().hex[:20]}"
        if client_order_id in self._orders:
            old = self._orders[client_order_id]
            return str(old.get("venue_order_id") or ""), {**old, "idempotency_hit": True}

        if qty <= 0:
            snap = {
                "client_order_id": client_order_id,
                "venue_order_id": None,
                "symbol": symbol,
                "side": side,
                "status": "rejected",
                "requested_qty": qty,
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "fees_paid": 0.0,
                "reject_reason": "invalid_quantity",
                "submitted_at": self._utcnow().isoformat(),
            }
            self._orders[client_order_id] = snap
            return None, snap

        rng = self._rng(client_order_id)
        if self.enable_random_reject and rng.random() < self.reject_rate:
            snap = {
                "client_order_id": client_order_id,
                "venue_order_id": None,
                "symbol": symbol,
                "side": side,
                "status": "rejected",
                "requested_qty": qty,
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "fees_paid": 0.0,
                "reject_reason": "paper_reject_simulated",
                "submitted_at": self._utcnow().isoformat(),
            }
            self._orders[client_order_id] = snap
            return None, snap

        market_state = context.get("market_state") if isinstance(context.get("market_state"), dict) else {}
        limit_price = child_order.get("limit_price")
        est_price = float(context.get("est_price") or 0.0)
        if est_price <= 0:
            est_price = float(limit_price or 0.0)
        bid = float(market_state.get("bid_px") or 0.0)
        ask = float(market_state.get("ask_px") or 0.0)
        spread_bps = float(market_state.get("spread_bps") or 0.0)
        if spread_bps <= 0.0 and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
            if mid > 0:
                spread_bps = (ask - bid) / mid * 10000.0
        if est_price <= 0 and bid > 0 and ask > 0:
            est_price = (bid + ask) / 2.0
        if est_price <= 0:
            est_price = 1.0
        imbalance = float(market_state.get("imbalance") or 0.0)
        max_slippage_bps = float(context.get("max_slippage_bps", 20.0) or 20.0)
        fill_ratio = 1.0
        if rng.random() < self.partial_fill_rate:
            fill_ratio = float(np.clip(rng.uniform(0.35, 0.95), 0.0, 1.0))
        liq_factor = max(0.4, min(1.0, 1.0 - min(0.6, abs(imbalance) * 0.5)))
        fill_ratio = max(0.0, min(1.0, fill_ratio * liq_factor))
        fill_qty = float(round(qty * fill_ratio, 12))
        tif = str(child_order.get("tif") or context.get("time_in_force") or "IOC").upper()
        if tif == "FOK" and fill_qty + 1e-12 < qty:
            fill_qty = 0.0
        if fill_qty <= 1e-12:
            venue_order_id = f"paper-{hashlib.sha1(client_order_id.encode('utf-8')).hexdigest()[:18]}"
            snap = {
                "client_order_id": client_order_id,
                "venue_order_id": venue_order_id,
                "symbol": symbol,
                "side": side,
                "status": "rejected",
                "requested_qty": qty,
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "fees_paid": 0.0,
                "reject_reason": "no_fill_after_retries",
                "submitted_at": self._utcnow().isoformat(),
            }
            self._orders[client_order_id] = snap
            self._fills_by_order[venue_order_id] = []
            return venue_order_id, snap

        base_slip = max(0.2, min(max_slippage_bps, max(0.2, spread_bps * 0.8 + abs(imbalance) * 4.0 + rng.uniform(0.0, 1.0))))
        side_sign = 1.0 if side == "buy" else -1.0
        fill_price = est_price * (1.0 + side_sign * (base_slip / 10000.0))
        fee_bps = float(context.get("fee_bps", 5.0) or 5.0)
        fee = float(fill_qty * fill_price * fee_bps / 10000.0)
        status = "filled" if fill_qty + 1e-12 >= qty else "partially_filled"
        venue_order_id = f"paper-{hashlib.sha1(client_order_id.encode('utf-8')).hexdigest()[:18]}"
        fill_row = {
            "fill_ts": self._utcnow(),
            "qty": fill_qty,
            "price": float(round(fill_price, 8)),
            "fee": float(round(fee, 10)),
            "fee_currency": str(context.get("fee_currency") or "USD"),
            "liquidity_flag": "maker" if str(context.get("execution_style") or "") == "passive_twap" else "taker",
            "raw": {
                "adapter": "paper",
                "spread_bps": spread_bps,
                "imbalance": imbalance,
                "seed": self.seed,
            },
        }
        self._fills_by_order[venue_order_id] = [fill_row]
        snap = {
            "client_order_id": client_order_id,
            "venue_order_id": venue_order_id,
            "symbol": symbol,
            "side": side,
            "status": status,
            "requested_qty": qty,
            "filled_qty": fill_qty,
            "avg_fill_price": float(round(fill_price, 8)),
            "fees_paid": float(round(fee, 10)),
            "reject_reason": None if status != "partially_filled" else "residual_unfilled",
            "submitted_at": self._utcnow().isoformat(),
        }
        self._orders[client_order_id] = snap
        pos = self._positions.get(symbol, {"position_qty": 0.0, "avg_cost": 0.0})
        signed = fill_qty if side == "buy" else -fill_qty
        new_qty = float(pos["position_qty"]) + signed
        if abs(new_qty) <= 1e-12:
            self._positions[symbol] = {"position_qty": 0.0, "avg_cost": 0.0}
        elif abs(float(pos["position_qty"])) <= 1e-12 or (float(pos["position_qty"]) > 0 and signed > 0) or (
            float(pos["position_qty"]) < 0 and signed < 0
        ):
            total_notional = float(pos["position_qty"]) * float(pos["avg_cost"]) + signed * fill_price
            self._positions[symbol] = {"position_qty": float(new_qty), "avg_cost": float(total_notional / new_qty)}
        else:
            self._positions[symbol] = {"position_qty": float(new_qty), "avg_cost": float(fill_price)}
        return venue_order_id, snap

    def poll_order(self, venue_order_id: str, timeout: float) -> Dict[str, Any]:
        _ = timeout
        if not venue_order_id:
            return {"venue_order_id": None, "status": "rejected", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0}
        for row in self._orders.values():
            if str(row.get("venue_order_id") or "") == str(venue_order_id):
                return dict(row)
        return {"venue_order_id": venue_order_id, "status": "rejected", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0}

    def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        for key, row in self._orders.items():
            if str(row.get("venue_order_id") or "") != str(venue_order_id):
                continue
            if str(row.get("status") or "") in {"filled", "rejected", "canceled"}:
                return {"venue_order_id": venue_order_id, "status": str(row.get("status") or "unknown")}
            row["status"] = "canceled"
            self._orders[key] = row
            return {"venue_order_id": venue_order_id, "status": "canceled"}
        return {"venue_order_id": venue_order_id, "status": "unknown"}

    def fetch_fills(self, venue_order_id: str) -> List[Dict[str, Any]]:
        return list(self._fills_by_order.get(str(venue_order_id), []))

    def fetch_balances(self) -> Dict[str, Any]:
        cash = float(os.getenv("PAPER_CASH_USD", "100000") or 100000.0)
        used_margin = 0.0
        unrealized = 0.0
        for row in self._positions.values():
            qty = float(row.get("position_qty") or 0.0)
            avg = float(row.get("avg_cost") or 0.0)
            notional = abs(qty * avg)
            used_margin += notional * float(os.getenv("PAPER_MARGIN_RATE", "0.1") or 0.1)
        equity = cash + unrealized
        free_margin = max(0.0, equity - used_margin)
        margin_ratio = (equity / used_margin) if used_margin > 1e-9 else 999.0
        return {
            "cash": float(cash),
            "equity": float(equity),
            "free_margin": float(free_margin),
            "used_margin": float(used_margin),
            "margin_ratio": float(margin_ratio),
            "account_currency": str(os.getenv("PAPER_ACCOUNT_CCY", "USD")).upper(),
            "raw": {"adapter": "paper"},
        }

    def fetch_open_orders(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for row in self._orders.values():
            st = str(row.get("status") or "").lower()
            if st not in {"submitted", "partially_filled"}:
                continue
            out.append(
                {
                    "client_order_id": str(row.get("client_order_id") or ""),
                    "venue_order_id": str(row.get("venue_order_id") or ""),
                    "symbol": str(row.get("symbol") or ""),
                    "side": str(row.get("side") or "buy"),
                    "qty": float(row.get("requested_qty") or 0.0),
                    "filled_qty": float(row.get("filled_qty") or 0.0),
                    "status": st,
                    "created_at": row.get("submitted_at"),
                    "updated_at": row.get("submitted_at"),
                }
            )
        return out

    def fetch_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for symbol, row in self._positions.items():
            out.append({"symbol": symbol, "position_qty": float(row.get("position_qty") or 0.0), "avg_cost": float(row.get("avg_cost") or 0.0)})
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


class CoinbaseLiveAdapter(ExecutionAdapterBase):
    name = "coinbase_live"

    def __init__(self):
        self.base_url = os.getenv("COINBASE_ADVANCED_TRADE_API", "https://api.coinbase.com").strip().rstrip("/")
        self.api_key_name = os.getenv("COINBASE_API_KEY", "").strip()
        self.api_private_key = os.getenv("COINBASE_API_SECRET", "").replace("\\n", "\n").strip()
        self.timeout_sec = float(os.getenv("COINBASE_EXEC_TIMEOUT_SEC", "5"))
        self.quote_currency = os.getenv("COINBASE_QUOTE_CURRENCY", "USD").strip().upper() or "USD"
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.35,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._idempotency: Dict[str, Dict[str, Any]] = {}

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
        resp = None
        max_attempts = int(os.getenv("COINBASE_HTTP_MAX_RETRIES", "3") or 3)
        for i in range(max(1, max_attempts)):
            resp = requests.request(
                method=method.upper(),
                url=f"{self.base_url}{path}",
                headers=headers,
                json=payload,
                params=params,
                timeout=self.timeout_sec,
            )
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
            if i + 1 < max_attempts:
                time.sleep(0.2 * (2**i))
        if resp is None:
            raise RuntimeError("coinbase_request_failed")
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

    def prepare(self, symbol: str) -> Dict[str, Any]:
        product_id = self._to_product_id(str(symbol or ""), self.quote_currency)
        return {"symbol": str(symbol or "").upper(), "product_id": product_id}

    def submit_order(self, child_order: Dict[str, Any], context: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
        symbol = str(child_order.get("symbol") or "")
        product_id = self._to_product_id(symbol, self.quote_currency)
        qty = float(child_order.get("qty") or 0.0)
        side = "BUY" if str(child_order.get("side") or "").lower() == "buy" else "SELL"
        limit_price = child_order.get("limit_price")
        tif = str(child_order.get("tif") or context.get("time_in_force") or "IOC").upper()
        client_order_id = str(child_order.get("client_order_id") or uuid.uuid4().hex)
        if client_order_id in self._idempotency:
            hit = dict(self._idempotency[client_order_id])
            return hit.get("venue_order_id"), {**hit, "idempotency_hit": True}
        if qty <= 0:
            return None, {"status": "rejected", "reject_reason": "invalid_quantity", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0}
        cfg_key = "limit_limit_gtc"
        if tif == "IOC":
            cfg_key = "limit_limit_ioc"
        elif tif == "FOK":
            cfg_key = "limit_limit_fok"
        body: Dict[str, Any] = {"client_order_id": client_order_id, "product_id": product_id, "side": side}
        if limit_price is not None:
            body["order_configuration"] = {
                cfg_key: {
                    "base_size": f"{qty:.8f}",
                    "limit_price": f"{float(limit_price):.8f}",
                    "post_only": False,
                }
            }
        else:
            body["order_configuration"] = {"market_market_ioc": {"base_size": f"{qty:.8f}"}}
        code, resp = self._request("POST", "/api/v3/brokerage/orders", payload=body)
        snap = self._extract_order_snapshot(resp)
        order_id = str(snap.get("order_id") or "") or None
        status = self._map_status(str(snap.get("status") or ""), float(snap.get("filled_qty") or 0.0), qty)
        out = {
            "status": status if code < 400 else "rejected",
            "filled_qty": float(snap.get("filled_qty") or 0.0),
            "avg_fill_price": snap.get("avg_fill_price"),
            "fees_paid": float(snap.get("fees_paid") or 0.0),
            "reject_reason": str(snap.get("reject_reason") or (f"venue_http_{code}" if code >= 400 else "")) or None,
            "http_code": code,
            "client_order_id": client_order_id,
            "venue_order_id": order_id,
        }
        self._idempotency[client_order_id] = dict(out)
        return order_id, out

    def poll_order(self, venue_order_id: str, timeout: float) -> Dict[str, Any]:
        polled = self._poll_order(str(venue_order_id), timeout_sec=float(timeout or 2.0), poll_interval=0.4)
        return {
            "venue_order_id": str(venue_order_id),
            "status": self._map_status(str(polled.get("status") or ""), float(polled.get("filled_qty") or 0.0), max(1e-12, float(polled.get("filled_qty") or 0.0))),
            "filled_qty": float(polled.get("filled_qty") or 0.0),
            "avg_fill_price": polled.get("avg_fill_price"),
            "fees_paid": float(polled.get("fees_paid") or 0.0),
            "reject_reason": str(polled.get("reject_reason") or "") or None,
        }

    def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        code, body = self._cancel_order(str(venue_order_id))
        snap = self.poll_order(str(venue_order_id), timeout=1.2)
        return {
            "venue_order_id": str(venue_order_id),
            "status": "canceled" if code < 400 else "cancel_failed",
            "http_code": code,
            "poll_status": str(snap.get("status") or ""),
            "raw": body,
        }

    def fetch_fills(self, venue_order_id: str) -> List[Dict[str, Any]]:
        code, body = self._request("GET", "/api/v3/brokerage/orders/historical/fills", params={"order_id": str(venue_order_id)})
        if code >= 400:
            return []
        rows = body.get("fills")
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                qty = float(r.get("size") or r.get("filled_size") or 0.0)
                price = float(r.get("price") or 0.0)
            except Exception:
                continue
            if qty <= 0 or price <= 0:
                continue
            out.append(
                {
                    "fill_ts": datetime.now(timezone.utc),
                    "qty": qty,
                    "price": price,
                    "fee": float(r.get("commission") or r.get("fee") or 0.0),
                    "fee_currency": r.get("fee_currency"),
                    "liquidity_flag": r.get("liquidity_indicator"),
                    "raw": r,
                }
            )
        return out

    def fetch_balances(self) -> Dict[str, Any]:
        code, body = self._request("GET", "/api/v3/brokerage/accounts")
        if code >= 400:
            return {"cash": 0.0, "equity": 0.0, "free_margin": 0.0, "used_margin": 0.0, "margin_ratio": 0.0, "account_currency": self.quote_currency, "raw": body}
        accounts = body.get("accounts") if isinstance(body.get("accounts"), list) else []
        equity = 0.0
        cash = 0.0
        for a in accounts:
            if not isinstance(a, dict):
                continue
            ccy = str(a.get("currency") or "").upper()
            avail = a.get("available_balance") if isinstance(a.get("available_balance"), dict) else {}
            hold = a.get("hold") if isinstance(a.get("hold"), dict) else {}
            qty = float(avail.get("value") or 0.0) + float(hold.get("value") or 0.0)
            if ccy == self.quote_currency:
                cash += qty
                equity += qty
        return {
            "cash": float(cash),
            "equity": float(equity),
            "free_margin": float(cash),
            "used_margin": 0.0,
            "margin_ratio": 999.0 if cash > 0 else 0.0,
            "account_currency": self.quote_currency,
            "raw": {"accounts": accounts},
        }

    def fetch_open_orders(self) -> List[Dict[str, Any]]:
        code, body = self._request("GET", "/api/v3/brokerage/orders/historical/batch", params={"limit": 100})
        if code >= 400:
            return []
        rows = body.get("orders")
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            status = self._map_status(
                str(r.get("status") or ""),
                float(r.get("filled_size") or r.get("filled_quantity") or 0.0),
                max(1e-12, float(r.get("base_size") or r.get("size") or 0.0)),
            )
            if status not in {"submitted", "partially_filled"}:
                continue
            out.append(
                {
                    "client_order_id": str(r.get("client_order_id") or ""),
                    "venue_order_id": str(r.get("order_id") or r.get("id") or ""),
                    "symbol": str(r.get("product_id") or ""),
                    "side": str(r.get("side") or "").lower(),
                    "qty": float(r.get("base_size") or r.get("size") or 0.0),
                    "filled_qty": float(r.get("filled_size") or r.get("filled_quantity") or 0.0),
                    "status": status,
                    "created_at": r.get("created_time"),
                    "updated_at": r.get("last_fill_time") or r.get("created_time"),
                }
            )
        return out

    def fetch_positions(self) -> List[Dict[str, Any]]:
        code, body = self._request("GET", "/api/v3/brokerage/accounts")
        if code >= 400:
            return []
        accounts = body.get("accounts")
        if not isinstance(accounts, list):
            return []
        out: List[Dict[str, Any]] = []
        for a in accounts:
            if not isinstance(a, dict):
                continue
            currency = str(a.get("currency") or "").upper()
            bal = a.get("available_balance") if isinstance(a.get("available_balance"), dict) else {}
            qty = float(bal.get("value") or 0.0) if isinstance(bal, dict) else 0.0
            if currency and abs(qty) > 0:
                out.append({"symbol": currency, "position_qty": qty, "avg_cost": 0.0, "raw": a})
        return out

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


class BitgetLiveAdapter(ExecutionAdapterBase):
    name = "bitget_live"

    def __init__(self):
        self.base_url = os.getenv("BITGET_BASE_URL", "https://api.bitget.com").strip().rstrip("/")
        self.api_key = os.getenv("BITGET_API_KEY", "").strip()
        self.api_secret = os.getenv("BITGET_API_SECRET", "").strip()
        self.api_passphrase = os.getenv("BITGET_API_PASSPHRASE", "").strip()
        self.timeout_sec = float(os.getenv("BITGET_TIMEOUT_SEC", "6") or 6.0)
        self.recv_window_ms = str(int(float(os.getenv("BITGET_RECV_WINDOW_MS", "5000") or 5000)))
        self.default_margin_coin = os.getenv("BITGET_DEFAULT_MARGIN_COIN", "USDT").strip().upper() or "USDT"
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            raise_on_status=False,
        )
        self.session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._rules_cache: Dict[str, Dict[str, Any]] = {}
        self._order_lookup: Dict[str, Dict[str, Any]] = {}
        self._idempotency: Dict[str, Dict[str, Any]] = {}

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
        resp = None
        max_attempts = int(os.getenv("BITGET_HTTP_MAX_RETRIES", "3") or 3)
        for i in range(max(1, max_attempts)):
            resp = requests.request(
                method=method.upper(),
                url=f"{self.base_url}{path}",
                headers=headers,
                data=body_str if payload is not None else None,
                timeout=self.timeout_sec,
            )
            if resp.status_code not in {429, 500, 502, 503, 504}:
                break
            if i + 1 < max_attempts:
                time.sleep(0.2 * (2**i))
        if resp is None:
            raise RuntimeError("bitget_request_failed")
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

    def prepare(self, symbol: str) -> Dict[str, Any]:
        sym = self._symbol(str(symbol or ""), "spot")
        if sym in self._rules_cache:
            return dict(self._rules_cache[sym])
        rules = {"symbol": sym, "min_qty": 0.000001, "qty_step": 0.000001, "price_step": 0.000001}
        if not sym:
            return rules
        code, body = self._request("GET", f"/api/v2/spot/public/symbols?symbol={sym}")
        if code < 400 and str(body.get("code") or "00000") == "00000":
            data = body.get("data")
            row = data[0] if isinstance(data, list) and data else (data if isinstance(data, dict) else {})
            try:
                rules["min_qty"] = float(row.get("minTradeAmount") or row.get("minTradeNum") or rules["min_qty"])
                rules["qty_step"] = float(row.get("quantityScale") or row.get("sizeMultiplier") or rules["qty_step"])
                rules["price_step"] = float(row.get("priceScale") or row.get("priceMultiplier") or rules["price_step"])
            except Exception:
                pass
        self._rules_cache[sym] = dict(rules)
        return rules

    def submit_order(self, child_order: Dict[str, Any], context: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
        market_type = str(context.get("market_type") or "spot").strip().lower()
        product_type = str(context.get("product_type") or "USDT-FUTURES").strip()
        margin_mode = str(context.get("margin_mode") or "cross").strip().lower()
        position_mode = str(context.get("position_mode") or "one_way").strip().lower()
        reduce_only = bool(context.get("reduce_only") or False)
        leverage = context.get("leverage")
        symbol = self._symbol(str(child_order.get("symbol") or ""), market_type)
        side = "buy" if str(child_order.get("side") or "").lower() == "buy" else "sell"
        qty = float(child_order.get("qty") or 0.0)
        tif = str(child_order.get("tif") or context.get("time_in_force") or "IOC").upper()
        client_oid = str(child_order.get("client_order_id") or f"ms-{uuid.uuid4().hex[:16]}")
        if client_oid in self._idempotency:
            hit = dict(self._idempotency[client_oid])
            return hit.get("venue_order_id"), {**hit, "idempotency_hit": True}
        limit_price = child_order.get("limit_price")
        if not self.api_key or not self.api_secret or not self.api_passphrase:
            return None, {"status": "rejected", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0, "reject_reason": "bitget_credentials_not_configured"}
        if qty <= 0:
            return None, {"status": "rejected", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0, "reject_reason": "invalid_quantity"}
        rules = self.prepare(symbol)
        min_qty = float(rules.get("min_qty") or 0.0)
        if qty + 1e-12 < min_qty:
            return None, {"status": "rejected", "filled_qty": 0.0, "avg_fill_price": None, "fees_paid": 0.0, "reject_reason": "bitget_precision_invalid:min_qty"}
        order_type = "limit" if limit_price is not None else "market"
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
                "posMode": "hedge_mode" if position_mode == "hedge" else "one_way_mode",
            }
            if order_type == "limit":
                req["price"] = f"{float(limit_price):.8f}"
            if leverage is not None:
                req["leverage"] = f"{float(leverage):.2f}"
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
                req["price"] = f"{float(limit_price):.8f}"
        code, body = self._request("POST", path, payload=req)
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        order_id = str(data.get("orderId") or data.get("ordId") or "")
        if code >= 400 or str(body.get("code") or "00000") != "00000" or not order_id:
            return (
                None,
                {
                    "status": "rejected",
                    "filled_qty": 0.0,
                    "avg_fill_price": None,
                    "fees_paid": 0.0,
                    "reject_reason": self._classify_error(code, body),
                    "http_code": code,
                },
            )
        self._order_lookup[order_id] = {
            "symbol": symbol,
            "market_type": market_type,
            "product_type": product_type,
            "requested_qty": qty,
        }
        out = {
            "status": "submitted",
            "filled_qty": 0.0,
            "avg_fill_price": None,
            "fees_paid": 0.0,
            "reject_reason": None,
            "http_code": code,
            "venue_order_id": order_id,
            "client_order_id": client_oid,
        }
        self._idempotency[client_oid] = dict(out)
        return order_id, out

    def poll_order(self, venue_order_id: str, timeout: float) -> Dict[str, Any]:
        _ = timeout
        ctx = self._order_lookup.get(str(venue_order_id), {})
        symbol = str(ctx.get("symbol") or "")
        market_type = str(ctx.get("market_type") or "spot")
        product_type = str(ctx.get("product_type") or "USDT-FUTURES")
        requested_qty = float(ctx.get("requested_qty") or 0.0)
        path = "/api/v2/mix/order/detail" if market_type == "perp_usdt" else "/api/v2/spot/trade/orderInfo"
        payload = {"symbol": symbol, "orderId": str(venue_order_id)}
        if market_type == "perp_usdt":
            payload["productType"] = product_type
        code, body = self._request("POST", path, payload=payload)
        if code >= 400 or str(body.get("code") or "00000") != "00000":
            return {
                "venue_order_id": str(venue_order_id),
                "status": "rejected",
                "filled_qty": 0.0,
                "avg_fill_price": None,
                "fees_paid": 0.0,
                "reject_reason": self._classify_error(code, body, fallback="no_fill_after_retries"),
            }
        data = body.get("data") if isinstance(body.get("data"), dict) else {}
        filled_qty = float(data.get("baseVolume") or data.get("filledQty") or data.get("filledSize") or 0.0)
        avg_fill_price = data.get("priceAvg") or data.get("fillPrice") or data.get("avgPrice")
        avg_fill_price_f = float(avg_fill_price) if avg_fill_price not in (None, "") else None
        status = self._map_status(str(data.get("status") or data.get("state") or "submitted"), filled_qty, max(requested_qty, filled_qty))
        return {
            "venue_order_id": str(venue_order_id),
            "status": status,
            "filled_qty": filled_qty,
            "avg_fill_price": avg_fill_price_f,
            "fees_paid": 0.0,
            "reject_reason": None if status in {"filled", "partially_filled", "submitted"} else "no_fill_after_retries",
        }

    def cancel_order(self, venue_order_id: str) -> Dict[str, Any]:
        ctx = self._order_lookup.get(str(venue_order_id), {})
        symbol = str(ctx.get("symbol") or "")
        market_type = str(ctx.get("market_type") or "spot")
        product_type = str(ctx.get("product_type") or "USDT-FUTURES")
        path = "/api/v2/mix/order/cancel-order" if market_type == "perp_usdt" else "/api/v2/spot/trade/cancel-order"
        payload = {"symbol": symbol, "orderId": str(venue_order_id)}
        if market_type == "perp_usdt":
            payload["productType"] = product_type
        code, body = self._request("POST", path, payload=payload)
        polled = self.poll_order(str(venue_order_id), timeout=1.0)
        return {
            "venue_order_id": str(venue_order_id),
            "status": "canceled" if code < 400 and str(body.get("code") or "00000") == "00000" else "cancel_failed",
            "poll_status": str(polled.get("status") or ""),
            "http_code": code,
            "raw": body,
        }

    def fetch_fills(self, venue_order_id: str) -> List[Dict[str, Any]]:
        ctx = self._order_lookup.get(str(venue_order_id), {})
        symbol = str(ctx.get("symbol") or "")
        market_type = str(ctx.get("market_type") or "spot")
        product_type = str(ctx.get("product_type") or "USDT-FUTURES")
        if not symbol:
            return []
        path = "/api/v2/mix/order/fills" if market_type == "perp_usdt" else "/api/v2/spot/trade/fills"
        payload = {"symbol": symbol, "orderId": str(venue_order_id)}
        if market_type == "perp_usdt":
            payload["productType"] = product_type
        code, body = self._request("POST", path, payload=payload)
        if code >= 400 or str(body.get("code") or "00000") != "00000":
            return []
        data = body.get("data")
        rows = data if isinstance(data, list) else (data.get("fills") if isinstance(data, dict) else [])
        if not isinstance(rows, list):
            return []
        out: List[Dict[str, Any]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            try:
                qty = float(r.get("sizeQty") or r.get("fillQty") or r.get("baseVolume") or 0.0)
                price = float(r.get("priceAvg") or r.get("fillPrice") or r.get("price") or 0.0)
            except Exception:
                continue
            if qty <= 0 or price <= 0:
                continue
            out.append(
                {
                    "fill_ts": datetime.now(timezone.utc),
                    "qty": qty,
                    "price": price,
                    "fee": float(r.get("fee") or 0.0),
                    "fee_currency": r.get("feeCoin"),
                    "liquidity_flag": r.get("execType"),
                    "raw": r,
                }
            )
        return out

    def fetch_balances(self) -> Dict[str, Any]:
        cash = 0.0
        equity = 0.0
        raw: Dict[str, Any] = {}
        code, body = self._request("GET", "/api/v2/spot/account/assets")
        if code < 400 and str(body.get("code") or "00000") == "00000":
            data = body.get("data")
            rows = data if isinstance(data, list) else (data.get("assets") if isinstance(data, dict) else [])
            if isinstance(rows, list):
                raw["spot_assets"] = rows
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    coin = str(r.get("coin") or "").upper()
                    avail = float(r.get("available") or r.get("availableAmount") or 0.0)
                    frozen = float(r.get("frozen") or r.get("lock") or 0.0)
                    qty = avail + frozen
                    if coin == self.default_margin_coin:
                        cash += qty
                        equity += qty
        free_margin = cash
        used_margin = max(0.0, equity - free_margin)
        margin_ratio = (equity / used_margin) if used_margin > 1e-9 else 999.0
        return {
            "cash": float(cash),
            "equity": float(equity),
            "free_margin": float(free_margin),
            "used_margin": float(used_margin),
            "margin_ratio": float(margin_ratio),
            "account_currency": self.default_margin_coin,
            "raw": raw,
        }

    def fetch_open_orders(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        code, body = self._request("GET", "/api/v2/spot/trade/unfilled-orders")
        if code < 400 and str(body.get("code") or "00000") == "00000":
            data = body.get("data")
            rows = data if isinstance(data, list) else (data.get("orderList") if isinstance(data, dict) else [])
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    out.append(
                        {
                            "client_order_id": str(r.get("clientOid") or ""),
                            "venue_order_id": str(r.get("orderId") or ""),
                            "symbol": str(r.get("symbol") or ""),
                            "side": str(r.get("side") or "").lower(),
                            "qty": float(r.get("size") or 0.0),
                            "filled_qty": float(r.get("baseVolume") or 0.0),
                            "status": "submitted",
                            "created_at": r.get("cTime"),
                            "updated_at": r.get("uTime") or r.get("cTime"),
                        }
                    )
        return out

    def fetch_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        code, body = self._request("GET", "/api/v2/spot/account/assets")
        if code < 400 and str(body.get("code") or "00000") == "00000":
            data = body.get("data")
            rows = data if isinstance(data, list) else (data.get("assets") if isinstance(data, dict) else [])
            if isinstance(rows, list):
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    coin = str(r.get("coin") or "").upper()
                    qty = float(r.get("available") or r.get("availableAmount") or 0.0)
                    if coin and abs(qty) > 0:
                        out.append({"symbol": coin, "position_qty": qty, "avg_cost": 0.0, "raw": r})
        code2, body2 = self._request("POST", "/api/v2/mix/position/all-position", payload={"productType": "USDT-FUTURES"})
        if code2 < 400 and str(body2.get("code") or "00000") == "00000":
            data2 = body2.get("data")
            rows2 = data2 if isinstance(data2, list) else (data2.get("positions") if isinstance(data2, dict) else [])
            if isinstance(rows2, list):
                for r in rows2:
                    if not isinstance(r, dict):
                        continue
                    symbol = str(r.get("symbol") or "").upper()
                    qty = float(r.get("total") or r.get("holdVol") or r.get("available") or 0.0)
                    avg = float(r.get("openPriceAvg") or r.get("averageOpenPrice") or 0.0)
                    if symbol and abs(qty) > 0:
                        out.append({"symbol": symbol, "position_qty": qty, "avg_cost": avg, "raw": r})
        return out

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

    @staticmethod
    def _status_event(status: str) -> str:
        s = str(status or "").strip().lower()
        if s == "filled":
            return "fill"
        if s == "partially_filled":
            return "partial_fill"
        if s == "canceled":
            return "cancel"
        if s == "expired":
            return "expire"
        if s == "rejected":
            return "reject"
        return "submit"

    @staticmethod
    def _reject_category(reason: str) -> str:
        r = str(reason or "").strip().lower()
        if not r:
            return "none"
        if "invalid_quantity" in r:
            return "invalid_quantity"
        if "slippage" in r:
            return "slippage"
        if "precision" in r:
            return "precision"
        if "rate" in r:
            return "rate_limited"
        if "timeout" in r or "no_fill" in r:
            return "timeout_or_no_fill"
        if "risk" in r:
            return "risk_blocked"
        if "credential" in r or "signature" in r:
            return "auth_error"
        return "other"

    @staticmethod
    def _ts_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def run_decision(
        self,
        repo: Any,
        decision_id: str,
        *,
        adapter: str,
        venue: str,
        market_type: str,
        max_orders: int,
        base_context: Optional[Dict[str, Any]] = None,
        orders_override: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        exec_adapter = self.adapters.get(str(adapter))
        if not exec_adapter:
            raise ValueError(f"unsupported adapter: {adapter}")

        decision = repo.get_execution_decision(decision_id)
        if not decision:
            repo.create_execution_decision(
                {
                    "decision_id": decision_id,
                    "adapter": adapter,
                    "venue": venue,
                    "market_type": market_type,
                    "requested_by": "api",
                    "status": "created",
                    "policy_snapshot": dict(base_context or {}),
                }
            )
            decision = repo.get_execution_decision(decision_id) or {"status": "created"}

        d_state = str(decision.get("status") or "created")
        if d_state == "created":
            d_state = transition_decision(d_state, "start")
            repo.start_execution_decision(decision_id)
        elif d_state != "running":
            raise ValueError(f"decision_not_runnable:{decision_id}:{d_state}")

        orders = list(orders_override or [])
        if not orders:
            orders = repo.fetch_orders_for_decision(decision_id, limit=max(1, int(max_orders)))
        if not orders:
            d_state = transition_decision("running", "fail")
            repo.finish_execution_decision(decision_id, d_state, summary={"total": 0, "filled": 0, "rejected": 0}, error="no_orders_for_decision")
            raise ValueError("no_orders_for_decision")

        merged_rows: List[Dict[str, Any]] = []
        reject_breakdown: Dict[str, int] = {}
        filled_parents = 0
        rejected_parents = 0
        market_missing = 0
        t_decision_start = time.perf_counter()

        for order in orders:
            parent_id = int(order["id"])
            symbol = str(order.get("target") or "").upper()
            side = str(order.get("side") or "buy").lower()
            parent_state = transition_parent("submitted", "start_exec")
            market_state = repo.latest_orderbook_l2(symbol) or {}
            context = resolve_order_execution_context(order, base_context or {})
            if isinstance(base_context, dict) and base_context.get("max_slippage_bps") is not None:
                try:
                    context["max_slippage_bps"] = min(float(context.get("max_slippage_bps") or 0.0), float(base_context.get("max_slippage_bps") or 0.0))
                except Exception:
                    pass
            context = {
                **context,
                "market_state": market_state,
                "venue": venue,
                "market_type": market_type,
                "est_price": float(order.get("est_price") or 0.0),
            }
            if "allow_market_fallback" not in context:
                context["allow_market_fallback"] = True
            if "twap_interval_sec" not in context:
                context["twap_interval_sec"] = float(os.getenv("EXEC_TWAP_INTERVAL_SEC", "0.5") or 0.5)
            plan = build_execution_plan(order=order, context=context, market_state=market_state)
            market_missing += int(plan.market_state_missing)
            child_template_rows = [
                {
                    "decision_id": decision_id,
                    "client_order_id": c.client_order_id,
                    "venue_order_id": None,
                    "symbol": c.symbol,
                    "side": c.side,
                    "qty": c.qty,
                    "limit_price": c.limit_price,
                    "tif": c.tif,
                    "status": "new",
                    "slice_index": c.slice_index,
                    "lifecycle": [],
                }
                for c in plan.child_orders
            ]
            child_queue: List[Dict[str, Any]] = [dict(r) for r in child_template_rows]
            child_out: List[Dict[str, Any]] = []
            all_fills: List[Dict[str, Any]] = []
            saw_reject = False
            market_fallback_count = 0
            max_market_fallback = max(0, int(context.get("max_retries") or 0)) + 1
            twap_interval_sec = float(context.get("twap_interval_sec") or 0.0)
            requested_qty = float(order.get("quantity") or 0.0)
            queue_idx = 0
            while queue_idx < len(child_queue):
                child_payload = dict(child_queue[queue_idx])
                queue_idx += 1
                created = repo.create_child_orders(parent_id, [child_payload])
                if not created:
                    saw_reject = True
                    continue
                child_id = int(created[0])
                c_state = transition_child("new", "submit")
                repo.update_child_order_status(
                    int(child_id),
                    status=c_state,
                    lifecycle_append={"event": "submit", "status": "submitted", "time": self._ts_iso(), "metrics": {}},
                )
                venue_order_id, ack = exec_adapter.submit_order(child_payload, context)
                logger.info(
                    "exec_child_submit decision_id=%s parent_order_id=%s child_order_id=%s symbol=%s side=%s qty=%.12f venue=%s adapter=%s client_order_id=%s venue_order_id=%s",
                    decision_id,
                    parent_id,
                    int(child_id),
                    symbol,
                    side,
                    float(child_payload.get("qty") or 0.0),
                    venue,
                    adapter,
                    str(child_payload.get("client_order_id") or ""),
                    str(venue_order_id or ""),
                )
                if not venue_order_id:
                    c_state = transition_child(c_state, "reject")
                    repo.update_child_order_status(
                        int(child_id),
                        status=c_state,
                        lifecycle_append={"event": "submit_ack", "status": "rejected", "time": self._ts_iso(), "metrics": {}},
                    )
                    saw_reject = True
                    child_out.append(
                        {
                            "id": int(child_id),
                            **child_payload,
                            "status": c_state,
                            "venue_order_id": None,
                            "ack": ack,
                        }
                    )
                    continue

                repo.update_child_order_status(
                    int(child_id),
                    status="submitted",
                    venue_order_id=str(venue_order_id),
                    lifecycle_append={"event": "submit_ack", "status": "accepted", "time": self._ts_iso(), "metrics": {}},
                )
                polled = exec_adapter.poll_order(str(venue_order_id), timeout=float(context.get("limit_timeout_sec") or 2.0))
                logger.info(
                    "exec_child_poll decision_id=%s parent_order_id=%s child_order_id=%s symbol=%s side=%s qty=%.12f venue=%s adapter=%s client_order_id=%s venue_order_id=%s status=%s",
                    decision_id,
                    parent_id,
                    int(child_id),
                    symbol,
                    side,
                    float(child_payload.get("qty") or 0.0),
                    venue,
                    adapter,
                    str(child_payload.get("client_order_id") or ""),
                    str(venue_order_id),
                    str(polled.get("status") or ""),
                )
                child_requested_qty = float(child_payload.get("qty") or 0.0)
                child_polled_qty = float(polled.get("filled_qty") or 0.0)
                pevent = self._status_event(str(polled.get("status") or "submitted"))
                if pevent != "submit":
                    c_state = transition_child("submitted", pevent)
                else:
                    c_state = "submitted"
                repo.update_child_order_status(
                    int(child_id),
                    status=c_state,
                    venue_order_id=str(venue_order_id),
                    lifecycle_append={
                        "event": "poll",
                        "status": str(polled.get("status") or "submitted"),
                        "time": self._ts_iso(),
                        "metrics": {
                            "filled_qty": float(polled.get("filled_qty") or 0.0),
                            "avg_fill_price": float(polled.get("avg_fill_price") or 0.0) if polled.get("avg_fill_price") is not None else 0.0,
                        },
                    },
                )

                # Non-terminal and under-filled limit child should be explicitly canceled before replacement.
                if (
                    str(c_state) in {"submitted", "partially_filled"}
                    and child_payload.get("limit_price") is not None
                    and child_polled_qty + 1e-12 < child_requested_qty
                ):
                    cancel_snapshot = exec_adapter.cancel_order(str(venue_order_id))
                    cancel_status = str(cancel_snapshot.get("status") or "").lower()
                    if cancel_status == "canceled":
                        try:
                            c_state = transition_child(str(c_state), "cancel")
                        except Exception:
                            c_state = "canceled"
                        repo.update_child_order_status(
                            int(child_id),
                            status=str(c_state),
                            venue_order_id=str(venue_order_id),
                            lifecycle_append={
                                "event": "cancel",
                                "status": "canceled",
                                "time": self._ts_iso(),
                                "metrics": {},
                            },
                        )

                fills = exec_adapter.fetch_fills(str(venue_order_id))
                if not fills and float(polled.get("filled_qty") or 0.0) > 0 and float(polled.get("avg_fill_price") or 0.0) > 0:
                    fills = [
                        {
                            "fill_ts": datetime.now(timezone.utc),
                            "qty": float(polled.get("filled_qty") or 0.0),
                            "price": float(polled.get("avg_fill_price") or 0.0),
                            "fee": float(polled.get("fees_paid") or 0.0),
                            "fee_currency": "USD",
                            "liquidity_flag": None,
                            "raw": {"synthetic_from_poll": True},
                        }
                    ]
                repo.insert_execution_fills(int(child_id), fills)
                for f in fills:
                    all_fills.append({**f, "child_order_id": int(child_id)})

                if str(c_state) in {"rejected", "expired"}:
                    saw_reject = True
                child_out.append(
                    {
                        "id": int(child_id),
                        **child_payload,
                        "status": str(c_state),
                        "venue_order_id": str(venue_order_id),
                        "ack": ack,
                        "poll": polled,
                    }
                )
                parent_agg_now = repo.update_parent_from_fills(parent_id)

                # marketable_limit residual routing: promote remaining qty to market IOC children.
                if plan.style == "marketable_limit":
                    remaining_qty = max(0.0, requested_qty - float(parent_agg_now.get("filled_qty") or 0.0))
                    is_underfilled = remaining_qty > 1e-12
                    if (
                        bool(context.get("allow_market_fallback", True))
                        and
                        is_underfilled
                        and market_fallback_count < max_market_fallback
                        and str(c_state) in {"submitted", "partially_filled", "rejected", "canceled", "expired"}
                    ):
                        retry_idx = market_fallback_count + 1
                        market_fallback_count += 1
                        new_slice = int(child_payload.get("slice_index") or 0) + 1000 + retry_idx
                        child_queue.append(
                            {
                                "decision_id": decision_id,
                                "client_order_id": f"{decision_id}:{parent_id}:{new_slice}:{retry_idx}",
                                "venue_order_id": None,
                                "symbol": symbol,
                                "side": side,
                                "qty": float(round(remaining_qty, 12)),
                                "limit_price": None,
                                "tif": "IOC",
                                "status": "new",
                                "slice_index": new_slice,
                                "lifecycle": [],
                            }
                        )

                if (
                    plan.style == "passive_twap"
                    and adapter != "paper"
                    and twap_interval_sec > 0.0
                    and queue_idx < len(child_queue)
                ):
                    time.sleep(min(5.0, max(0.0, twap_interval_sec)))

            agg = repo.update_parent_from_fills(parent_id)
            filled_qty_now = float(agg.get("filled_qty") or 0.0)
            if filled_qty_now >= max(0.0, requested_qty) - 1e-12 and requested_qty > 0:
                agg_status = "filled"
            elif filled_qty_now > 1e-12:
                agg_status = "partially_filled"
            elif saw_reject:
                agg_status = "rejected"
            else:
                agg_status = str(agg.get("status") or "submitted")
            pevent = self._status_event(agg_status)
            if pevent != "submit":
                try:
                    parent_state = transition_parent(parent_state, pevent)
                except Exception:
                    if agg_status in {"filled", "partially_filled", "rejected", "canceled"}:
                        parent_state = agg_status
            else:
                parent_state = "executing"

            execution = {
                "status": parent_state if parent_state != "executing" else agg_status,
                "filled_qty": float(agg.get("filled_qty") or 0.0),
                "avg_fill_price": agg.get("avg_fill_price"),
                "fees_paid": float(agg.get("fees_paid") or 0.0),
                "reject_reason": "no_fill_after_retries" if float(agg.get("filled_qty") or 0.0) <= 1e-12 and saw_reject else None,
                "venue_order_id": agg.get("last_venue_order_id"),
                "execution_policy": str(context.get("execution_policy") or ""),
                "market_state_missing": int(plan.market_state_missing),
            }
            execution_trace = build_execution_trace(order, execution)
            execution_trace["market_state_missing"] = int(plan.market_state_missing)
            fills_for_meta: List[Dict[str, Any]] = []
            for f in all_fills:
                ff = dict(f)
                ts = ff.get("fill_ts")
                if isinstance(ts, datetime):
                    ff["fill_ts"] = ts.isoformat()
                fills_for_meta.append(ff)
            repo.update_order_execution(
                parent_id,
                status=str(execution.get("status") or "rejected"),
                metadata={
                    "execution": execution,
                    "execution_trace": execution_trace,
                    "execution_policy": str(context.get("execution_policy") or ""),
                    "child_orders": child_out,
                    "fills": fills_for_meta,
                },
            )

            prev_pos = repo.get_position_live(venue=venue, symbol=symbol, account_id="")
            next_pos = apply_fills_to_position(prev_pos or {}, all_fills, side=side)
            mark_price = float(order.get("est_price") or execution.get("avg_fill_price") or 0.0)
            realized = compute_realized_pnl(prev_pos or {}, all_fills, side=side)
            unrealized = compute_unrealized_pnl(next_pos, mark_price)
            repo.upsert_position_live(
                venue=venue,
                symbol=symbol,
                account_id="",
                position_qty=float(next_pos.get("position_qty") or 0.0),
                avg_cost=float(next_pos.get("avg_cost") or 0.0),
                unrealized_pnl=float(unrealized),
                raw={
                    "decision_id": decision_id,
                    "parent_order_id": parent_id,
                    "realized_pnl": float(realized),
                    "last_mark_price": mark_price,
                },
            )

            final_status = str(execution.get("status") or "rejected")
            if final_status in {"filled", "partially_filled"}:
                filled_parents += 1
            else:
                rejected_parents += 1
                reason = str(execution.get("reject_reason") or "other")
                reason_cat = self._reject_category(reason)
                reject_breakdown[reason_cat] = reject_breakdown.get(reason_cat, 0) + 1

            merged_rows.append(
                {
                    **order,
                    "execution": execution,
                    "execution_trace": execution_trace,
                    "execution_policy": str(context.get("execution_policy") or ""),
                    "child_orders": child_out,
                    "fills": fills_for_meta,
                }
            )

        summary = {
            "total": len(orders),
            "filled": filled_parents,
            "rejected": rejected_parents,
            "reject_breakdown": reject_breakdown,
            "market_state_missing": market_missing,
            "latency_sec": float(max(0.0, time.perf_counter() - t_decision_start)),
        }
        d_state = transition_decision("running", "complete")
        repo.finish_execution_decision(decision_id, d_state, summary=summary)
        return {
            "decision_id": decision_id,
            "adapter": adapter,
            "total": len(orders),
            "filled": filled_parents,
            "rejected": rejected_parents,
            "reject_breakdown": reject_breakdown,
            "orders": merged_rows,
        }
