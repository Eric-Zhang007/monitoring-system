from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from account_state.models import AccountHealth, AccountState, BalanceState, ExecutionStats, OrderState, PositionState


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AccountStateAggregator:
    def __init__(self, adapter, venue: str, store=None, cache_ttl_s: int = 10):
        self.adapter = adapter
        self.venue = str(venue or "unknown")
        self.store = store
        self.cache_ttl_s = max(1, int(cache_ttl_s))
        self._state: Optional[AccountState] = None

    def _is_fresh(self, ts: datetime) -> bool:
        return (_utcnow() - ts).total_seconds() <= float(self.cache_ttl_s)

    @staticmethod
    def _parse_balances(row: Dict[str, Any]) -> BalanceState:
        return BalanceState(
            cash=float(row.get("cash") or 0.0),
            equity=float(row.get("equity") or 0.0),
            free_margin=float(row.get("free_margin") or 0.0),
            used_margin=float(row.get("used_margin") or 0.0),
            margin_ratio=float(row.get("margin_ratio") or 0.0),
            account_currency=str(row.get("account_currency") or "USD"),
        )

    @staticmethod
    def _parse_positions(rows) -> Dict[str, PositionState]:
        out: Dict[str, PositionState] = {}
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            symbol = str(r.get("symbol") or "").upper()
            if not symbol:
                continue
            out[symbol] = PositionState(
                symbol=symbol,
                qty=float(r.get("position_qty") if r.get("position_qty") is not None else r.get("qty") or 0.0),
                avg_cost=float(r.get("avg_cost") or 0.0),
                unrealized_pnl=float(r.get("unrealized_pnl") or 0.0),
                realized_pnl=float(r.get("realized_pnl") or 0.0),
                leverage=float(r.get("leverage") or 1.0),
                liq_price=(float(r["liq_price"]) if r.get("liq_price") is not None else None),
                margin_mode=str(r.get("margin_mode") or "cross"),
                position_mode=str(r.get("position_mode") or "one_way"),
            )
        return out

    @staticmethod
    def _parse_open_orders(rows) -> list[OrderState]:
        out: list[OrderState] = []
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            out.append(
                OrderState(
                    client_order_id=str(r.get("client_order_id") or ""),
                    venue_order_id=str(r.get("venue_order_id") or ""),
                    symbol=str(r.get("symbol") or "").upper(),
                    side=str(r.get("side") or "buy").lower(),
                    qty=float(r.get("qty") or 0.0),
                    filled_qty=float(r.get("filled_qty") or 0.0),
                    status=str(r.get("status") or "submitted"),
                    created_at=r.get("created_at") if isinstance(r.get("created_at"), datetime) else _utcnow(),
                    updated_at=r.get("updated_at") if isinstance(r.get("updated_at"), datetime) else _utcnow(),
                )
            )
        return out

    def refresh_full_state(self) -> AccountState:
        ts = _utcnow()
        balances_raw = self.adapter.fetch_balances()
        positions_raw = self.adapter.fetch_positions()
        open_orders_raw = self.adapter.fetch_open_orders()

        if self.store is not None and hasattr(self.store, "recent_execution_stats"):
            stats_raw = self.store.recent_execution_stats(self.venue, lookback_minutes=5) or {}
        else:
            stats_raw = {}
        stats = ExecutionStats(
            slippage_bps_p50=float(stats_raw.get("slippage_bps_p50") or 0.0),
            slippage_bps_p90=float(stats_raw.get("slippage_bps_p90") or 0.0),
            reject_rate_5m=float(stats_raw.get("reject_rate_5m") or 0.0),
            fill_latency_p50=float(stats_raw.get("fill_latency_p50") or 0.0),
            last_fill_ts=stats_raw.get("last_fill_ts") if isinstance(stats_raw.get("last_fill_ts"), datetime) else None,
        )

        state = AccountState(
            ts=ts,
            venue=self.venue,
            adapter=str(getattr(self.adapter, "name", "unknown")),
            balances=self._parse_balances(balances_raw or {}),
            positions=self._parse_positions(positions_raw),
            open_orders=self._parse_open_orders(open_orders_raw),
            execution_stats=stats,
            health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
            raw={
                "balances": balances_raw or {},
                "positions": positions_raw or [],
                "open_orders": open_orders_raw or [],
            },
        )
        self._state = state

        if self.store is not None:
            try:
                self.store.upsert_balance_state(
                    ts=ts,
                    venue=self.venue,
                    cash=state.balances.cash,
                    equity=state.balances.equity,
                    free_margin=state.balances.free_margin,
                    used_margin=state.balances.used_margin,
                    margin_ratio=state.balances.margin_ratio,
                    account_currency=state.balances.account_currency,
                    raw=balances_raw or {},
                )
                for pos in state.positions.values():
                    self.store.upsert_position_state(
                        ts=ts,
                        venue=self.venue,
                        symbol=pos.symbol,
                        qty=pos.qty,
                        avg_cost=pos.avg_cost,
                        liq_price=pos.liq_price,
                        unrealized_pnl=pos.unrealized_pnl,
                        realized_pnl=pos.realized_pnl,
                        leverage=pos.leverage,
                        margin_mode=pos.margin_mode,
                        position_mode=pos.position_mode,
                        raw=pos.model_dump(),
                    )
                self.store.insert_account_state_snapshot(
                    ts=ts,
                    venue=self.venue,
                    adapter=state.adapter,
                    equity=state.balances.equity,
                    free_margin=state.balances.free_margin,
                    margin_ratio=state.balances.margin_ratio,
                    raw_state=state.model_dump(mode="json"),
                )
            except Exception:
                pass
        return state

    def apply_order_event(self, evt: Dict[str, Any]) -> None:
        if not isinstance(evt, dict):
            return
        if self._state is None:
            return
        venue_order_id = str(evt.get("venue_order_id") or "")
        status = str(evt.get("status") or "").lower()
        if venue_order_id:
            updated = False
            for i, row in enumerate(self._state.open_orders):
                if row.venue_order_id != venue_order_id:
                    continue
                row.status = status or row.status
                row.filled_qty = float(evt.get("filled_qty") or row.filled_qty)
                row.updated_at = _utcnow()
                self._state.open_orders[i] = row
                updated = True
                break
            if not updated and status in {"submitted", "partially_filled"}:
                self._state.open_orders.append(
                    OrderState(
                        client_order_id=str(evt.get("client_order_id") or ""),
                        venue_order_id=venue_order_id,
                        symbol=str(evt.get("symbol") or "").upper(),
                        side=str(evt.get("side") or "buy").lower(),
                        qty=float(evt.get("qty") or 0.0),
                        filled_qty=float(evt.get("filled_qty") or 0.0),
                        status=status,
                    )
                )
        self._state.open_orders = [o for o in self._state.open_orders if str(o.status).lower() in {"submitted", "partially_filled"}]
        self._state.ts = _utcnow()
        self._state.health.is_fresh = True

    def get_state(self, require_fresh: bool = True) -> AccountState:
        if self._state is None:
            if require_fresh:
                raise RuntimeError("account_state_missing")
            return AccountState(health=AccountHealth(is_fresh=False, recon_ok=False, ws_ok=False, last_error="missing"))
        fresh = self._is_fresh(self._state.ts)
        self._state.health.is_fresh = bool(fresh)
        if require_fresh and not fresh:
            raise RuntimeError("account_state_stale")
        return self._state
