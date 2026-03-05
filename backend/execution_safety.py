from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional


class KillSwitchAction:
    NONE = "NONE"
    SAFE_MODE = "SAFE_MODE"
    PANIC_CLOSE = "PANIC_CLOSE"


@dataclass
class LocalGuardStop:
    symbol: str
    side: str
    trigger_price: float
    size: float
    armed_at: str
    reason: str
    active: bool = True


class StopLossManager:
    """
    Minimal stop-loss reliability layer:
    - prefer exchange trigger plan order if supported
    - otherwise arm local_guard
    - local_guard trigger calls protective_close callback and enters safe_mode
    """

    def __init__(
        self,
        *,
        supports_exchange_trigger: Callable[[], bool],
        place_exchange_stop: Callable[[str, str, float, float], Dict[str, Any]],
        protective_close: Callable[[str, str, float, str], None],
    ) -> None:
        self._supports_exchange_trigger = supports_exchange_trigger
        self._place_exchange_stop = place_exchange_stop
        self._protective_close = protective_close
        self.local_guards: Dict[str, LocalGuardStop] = {}
        self.safe_mode: bool = False
        self.safe_mode_reason: str = ""

    @staticmethod
    def _key(symbol: str, side: str) -> str:
        return f"{str(symbol).upper()}:{str(side).lower()}"

    @staticmethod
    def _utc_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def ensure_stop_loss(
        self,
        *,
        symbol: str,
        side: str,
        desired_sl_price: float,
        desired_size: float,
        mode: str,
    ) -> Dict[str, Any]:
        if desired_size <= 0:
            raise RuntimeError("stoploss_size_non_positive")
        if desired_sl_price <= 0:
            raise RuntimeError("stoploss_trigger_non_positive")
        mode_norm = str(mode or "trigger").strip().lower()
        if mode_norm in {"trigger", "plan"} and bool(self._supports_exchange_trigger()):
            placed = dict(self._place_exchange_stop(symbol, side, desired_sl_price, desired_size) or {})
            if not bool(placed.get("ok")):
                raise RuntimeError(f"exchange_stoploss_place_failed:{placed.get('reason')}")
            # exchange stop loss in place -> local guard not required
            self.local_guards.pop(self._key(symbol, side), None)
            return {
                "ok": True,
                "mode": "exchange_trigger",
                "symbol": str(symbol).upper(),
                "side": str(side).lower(),
                "order_id": placed.get("order_id"),
            }

        guard = LocalGuardStop(
            symbol=str(symbol).upper(),
            side=str(side).lower(),
            trigger_price=float(desired_sl_price),
            size=float(desired_size),
            armed_at=self._utc_iso(),
            reason="plan_unsupported_or_local_guard",
            active=True,
        )
        self.local_guards[self._key(guard.symbol, guard.side)] = guard
        return {
            "ok": True,
            "mode": "local_guard",
            "symbol": guard.symbol,
            "side": guard.side,
            "trigger_price": guard.trigger_price,
            "size": guard.size,
        }

    def process_local_guards(self, *, mark_prices: Mapping[str, float]) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []
        for key, guard in list(self.local_guards.items()):
            if not guard.active:
                continue
            mark = float(mark_prices.get(guard.symbol, 0.0) or 0.0)
            if mark <= 0:
                continue
            should_trigger = False
            if guard.side == "long":
                should_trigger = mark <= guard.trigger_price
            elif guard.side == "short":
                should_trigger = mark >= guard.trigger_price
            if not should_trigger:
                continue
            self._protective_close(guard.symbol, guard.side, guard.size, "local_guard_triggered")
            guard.active = False
            self.safe_mode = True
            self.safe_mode_reason = "local_guard_triggered"
            actions.append(
                {
                    "symbol": guard.symbol,
                    "side": guard.side,
                    "size": guard.size,
                    "trigger_price": guard.trigger_price,
                    "mark_price": mark,
                    "action": "protective_close",
                }
            )
            self.local_guards[key] = guard
        return actions


def read_kill_switch_action(
    *,
    file_path: str = "./KILL_SWITCH",
    env_key: str = "TRADER_KILL_SWITCH",
    sqlite_value: Optional[str] = None,
) -> str:
    p = Path(str(file_path))
    if p.exists():
        content = str(p.read_text(encoding="utf-8")).strip().lower()
        if content in {"", "safe", "safe_mode", "1", "true", "on"}:
            return KillSwitchAction.SAFE_MODE
        if content in {"panic", "panic_close", "2"}:
            return KillSwitchAction.PANIC_CLOSE
        return KillSwitchAction.SAFE_MODE

    env_val = str(os.getenv(env_key, "")).strip().lower()
    if env_val in {"1", "true", "safe", "safe_mode", "on"}:
        return KillSwitchAction.SAFE_MODE
    if env_val in {"panic", "panic_close", "2"}:
        return KillSwitchAction.PANIC_CLOSE

    if sqlite_value is not None:
        sv = str(sqlite_value).strip().lower()
        if sv in {"1", "true", "safe", "safe_mode", "on"}:
            return KillSwitchAction.SAFE_MODE
        if sv in {"panic", "panic_close", "2"}:
            return KillSwitchAction.PANIC_CLOSE

    return KillSwitchAction.NONE


class ExecutionSafetyController:
    def __init__(self) -> None:
        self.safe_mode: bool = False
        self.safe_mode_reason: str = ""
        self.stoploss_mode: str = str(os.getenv("STOPLOSS_MODE", "trigger")).strip().lower() or "trigger"
        self.plan_order_supported: Optional[bool] = None

    def apply_startup_probe(
        self,
        *,
        plan_order_supported: bool,
        safe_mode_on_failure: bool = True,
    ) -> Dict[str, Any]:
        self.plan_order_supported = bool(plan_order_supported)
        if bool(plan_order_supported):
            return {
                "supported": True,
                "stoploss_mode": self.stoploss_mode,
                "safe_mode": self.safe_mode,
            }
        if self.stoploss_mode in {"trigger", "plan"}:
            self.stoploss_mode = "local_guard"
            if bool(safe_mode_on_failure):
                self.safe_mode = True
                self.safe_mode_reason = "plan_order_capability_unsupported"
        return {
            "supported": False,
            "stoploss_mode": self.stoploss_mode,
            "safe_mode": self.safe_mode,
            "reason": self.safe_mode_reason or "plan_order_capability_unsupported",
        }

    def apply_price_feed_status(
        self,
        *,
        ws_ok: bool,
        rest_ok: bool,
        rest_fallback_action_when_local_guard: str = "safe_mode",
    ) -> Dict[str, Any]:
        if bool(ws_ok):
            return {
                "price_feed_mode": "ws",
                "safe_mode": self.safe_mode,
            }
        if not bool(rest_ok):
            self.safe_mode = True
            self.safe_mode_reason = "price_feed_unavailable"
            return {
                "price_feed_mode": "down",
                "safe_mode": self.safe_mode,
                "reason": self.safe_mode_reason,
            }
        mode = "rest"
        action = str(rest_fallback_action_when_local_guard or "safe_mode").strip().lower()
        if self.stoploss_mode == "local_guard" and action == "safe_mode":
            self.safe_mode = True
            self.safe_mode_reason = "ws_to_rest_with_local_guard"
        return {
            "price_feed_mode": mode,
            "safe_mode": self.safe_mode,
            "reason": self.safe_mode_reason,
        }

    def apply_unknown_positions(self, *, known_symbols: Iterable[str], exchange_positions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        known = {str(s).upper() for s in known_symbols}
        unknown: List[str] = []
        for row in exchange_positions:
            symbol = str((row or {}).get("symbol") or "").upper()
            if not symbol:
                continue
            size = abs(float((row or {}).get("size", (row or {}).get("qty", 0.0)) or 0.0))
            if size <= 0:
                continue
            if symbol not in known:
                unknown.append(symbol)
        if unknown:
            self.safe_mode = True
            self.safe_mode_reason = f"unknown_position_detected:{','.join(sorted(set(unknown)))}"
        return {
            "safe_mode": self.safe_mode,
            "reason": self.safe_mode_reason,
            "unknown_symbols": sorted(set(unknown)),
        }

    def preflight(
        self,
        *,
        sqlite_value: Optional[str] = None,
        file_path: str = "./KILL_SWITCH",
        env_key: str = "TRADER_KILL_SWITCH",
    ) -> Dict[str, Any]:
        action = read_kill_switch_action(file_path=file_path, env_key=env_key, sqlite_value=sqlite_value)
        if action == KillSwitchAction.SAFE_MODE:
            self.safe_mode = True
            if not self.safe_mode_reason:
                self.safe_mode_reason = "kill_switch_safe_mode"
            return {
                "blocked": True,
                "reason": self.safe_mode_reason,
                "kill_switch": action,
            }
        if action == KillSwitchAction.PANIC_CLOSE:
            self.safe_mode = True
            self.safe_mode_reason = "kill_switch_panic_close"
            return {
                "blocked": True,
                "reason": self.safe_mode_reason,
                "kill_switch": action,
            }
        if self.safe_mode:
            return {
                "blocked": True,
                "reason": self.safe_mode_reason or "safe_mode",
                "kill_switch": action,
            }
        return {
            "blocked": False,
            "reason": "",
            "kill_switch": action,
        }
