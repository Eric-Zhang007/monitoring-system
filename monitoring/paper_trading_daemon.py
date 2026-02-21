#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


DEFAULT_SYMBOLS = "BTC,ETH,SOL"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw) if raw.strip() else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _target_position(pred_payload: Dict[str, Any]) -> float:
    signal = float(pred_payload.get("expected_return") or 0.0)
    confidence = float(pred_payload.get("signal_confidence") or 0.0)
    neutral_eps = float(os.getenv("PAPER_NEUTRAL_EPS", "0.000001") or 0.000001)
    if abs(signal) <= neutral_eps:
        return 0.0
    signed = 1.0 if signal > 0 else -1.0
    conf = max(0.0, min(1.0, confidence))
    mag = min(1.0, abs(signal) * 80.0 * conf)
    return float(signed * mag)


def _http_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def _call_json(
    sess: requests.Session,
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 12.0,
) -> Dict[str, Any]:
    resp = sess.request(
        method=method.upper(),
        url=url,
        json=payload,
        timeout=max(0.5, float(timeout_sec)),
    )
    body: Dict[str, Any] = {}
    if resp.content:
        try:
            parsed = resp.json()
            if isinstance(parsed, dict):
                body = parsed
        except Exception:
            body = {"raw": resp.text[:500]}
    if resp.status_code >= 300:
        raise RuntimeError(f"http_{resp.status_code}:{body}")
    return body


def _run_cycle(
    *,
    api_base: str,
    symbols: List[str],
    horizon: str,
    min_confidence: float,
    strategy_id: str,
    capital_per_order_usd: float,
    max_orders: int,
    timeout_sec: float,
    state_path: Path,
    history_path: Path,
    execution_event_path: Path,
    control_path: Path,
) -> Dict[str, Any]:
    state = _load_json(state_path)
    control = _load_json(control_path)
    live_enabled = bool(control.get("live_enabled", False))
    paper_enabled = bool(control.get("paper_enabled", True))

    base_cycle: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "paper_enabled": paper_enabled,
        "live_enabled": live_enabled,
        "symbols": symbols,
        "horizon": horizon,
        "min_confidence": float(min_confidence),
    }

    if not paper_enabled:
        cycle = {
            **base_cycle,
            "status": "paper_disabled",
            "signals": 0,
            "orders": 0,
            "filled": 0,
            "rejected": 0,
            "reject_breakdown": {},
            "decision_id": None,
            "order_ids": [],
            "failures": [],
        }
        _write_json(state_path, cycle)
        _append_jsonl(history_path, cycle)
        return cycle

    sess = _http_session()
    signals: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for sym in symbols:
        try:
            sig = _call_json(
                sess,
                "POST",
                f"{api_base}/api/v2/signals/generate",
                payload={
                    "track": "liquid",
                    "target": sym,
                    "horizon": horizon,
                    "policy": "baseline-v2",
                    "min_confidence": float(min_confidence),
                    "strategy_id": strategy_id,
                    "cost_profile": "standard",
                    "risk_profile": "balanced",
                },
                timeout_sec=timeout_sec,
            )
            signals.append(sig)
        except Exception as exc:
            failures.append({"symbol": sym, "stage": "signal", "error": str(exc)})
            continue

        action = str(sig.get("action") or "hold").lower()
        confidence = float(sig.get("confidence") or 0.0)
        if action not in {"buy", "sell"} or confidence < float(min_confidence):
            continue

        try:
            pred = _call_json(
                sess,
                "POST",
                f"{api_base}/api/v2/predict/liquid",
                payload={"symbol": sym, "horizon": horizon},
                timeout_sec=timeout_sec,
            )
            px = float(pred.get("outputs", {}).get("current_price") or 0.0)
        except Exception as exc:
            failures.append({"symbol": sym, "stage": "price", "error": str(exc)})
            continue

        if px <= 0.0:
            failures.append({"symbol": sym, "stage": "price", "error": "invalid_price"})
            continue

        qty = max(0.000001, (float(capital_per_order_usd) / px) * max(0.2, min(1.0, confidence)))
        orders.append(
            {
                "target": sym,
                "track": "liquid",
                "side": action,
                "quantity": round(float(qty), 8),
                "est_price": float(px),
                "strategy_id": strategy_id,
                "metadata": {
                    "source": "paper_trading_daemon",
                    "signal_confidence": confidence,
                    "signal_id": sig.get("signal_id"),
                    "horizon": horizon,
                    "selected_horizon": horizon,
                    "strategy_bucket": str(sig.get("strategy_bucket") or "event"),
                },
            }
        )

    if len(orders) > max(1, int(max_orders)):
        orders = orders[: max(1, int(max_orders))]

    if not orders:
        cycle = {
            **base_cycle,
            "status": "no_orders",
            "signals": len(signals),
            "orders": 0,
            "filled": 0,
            "rejected": 0,
            "reject_breakdown": {},
            "decision_id": None,
            "order_ids": [],
            "failures": failures[:20],
            "prev_state_status": str(state.get("status") or ""),
        }
        _write_json(state_path, cycle)
        _append_jsonl(history_path, cycle)
        return cycle

    submit = _call_json(
        sess,
        "POST",
        f"{api_base}/api/v2/execution/orders",
        payload={
            "adapter": "paper",
            "venue": "coinbase",
            "time_in_force": "IOC",
            "max_slippage_bps": 12.0,
            "market_type": "spot",
            "orders": orders,
        },
        timeout_sec=timeout_sec,
    )
    decision_id = str(submit.get("decision_id") or "")
    if not decision_id:
        raise RuntimeError("missing_decision_id_from_submit")

    run_out = _call_json(
        sess,
        "POST",
        f"{api_base}/api/v2/execution/run",
        payload={
            "decision_id": decision_id,
            "adapter": "paper",
            "time_in_force": "IOC",
            "max_slippage_bps": 12.0,
            "venue": "coinbase",
            "market_type": "spot",
            "max_orders": len(orders),
            "limit_timeout_sec": 2.0,
            "max_retries": 1,
            "fee_bps": 5.0,
        },
        timeout_sec=timeout_sec,
    )
    for ord_row in list(run_out.get("orders") or []):
        if not isinstance(ord_row, dict):
            continue
        event_payload = {
            "timestamp": _now_iso(),
            "decision_id": decision_id,
            "symbol": str(ord_row.get("target") or ""),
            "horizon": str(((ord_row.get("metadata") or {}) if isinstance(ord_row.get("metadata"), dict) else {}).get("horizon") or horizon),
            "strategy_bucket": str(((ord_row.get("metadata") or {}) if isinstance(ord_row.get("metadata"), dict) else {}).get("strategy_bucket") or ""),
            "execution_policy": str(ord_row.get("execution_policy") or ""),
            "execution_trace": dict(ord_row.get("execution_trace") or {}),
            "execution": dict(ord_row.get("execution") or {}),
            "status": str(((ord_row.get("execution") or {}) if isinstance(ord_row.get("execution"), dict) else {}).get("status") or ""),
        }
        _append_jsonl(execution_event_path, event_payload)

    cycle = {
        **base_cycle,
        "status": "ok",
        "signals": len(signals),
        "orders": len(orders),
        "filled": int(run_out.get("filled") or 0),
        "rejected": int(run_out.get("rejected") or 0),
        "reject_breakdown": dict(run_out.get("reject_breakdown") or {}),
        "decision_id": decision_id,
        "order_ids": list(submit.get("order_ids") or []),
        "failures": failures[:20],
        "prev_state_status": str(state.get("status") or ""),
    }
    _write_json(state_path, cycle)
    _append_jsonl(history_path, cycle)
    return cycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous paper-trading daemon through unified v2 execution loop")
    parser.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", DEFAULT_SYMBOLS))
    parser.add_argument("--horizon", default=os.getenv("PAPER_HORIZON", "1d"), choices=["1h", "4h", "1d", "7d"])
    parser.add_argument("--min-confidence", type=float, default=float(os.getenv("PAPER_MIN_CONFIDENCE", "0.45")))
    parser.add_argument("--strategy-id", default=os.getenv("PAPER_STRATEGY_ID", "continuous-paper-v1"))
    parser.add_argument("--capital-per-order-usd", type=float, default=float(os.getenv("PAPER_CAPITAL_PER_ORDER_USD", "300.0")))
    parser.add_argument("--max-orders", type=int, default=int(os.getenv("PAPER_MAX_ORDERS_PER_CYCLE", "8")))
    parser.add_argument("--request-timeout-sec", type=float, default=float(os.getenv("PAPER_REQUEST_TIMEOUT_SEC", "12")))
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("PAPER_DAEMON_INTERVAL_SEC", "60")))
    parser.add_argument("--state-file", default=os.getenv("PAPER_STATE_FILE", "artifacts/paper/paper_state.json"))
    parser.add_argument("--history-file", default=os.getenv("PAPER_HISTORY_FILE", "artifacts/paper/paper_history.jsonl"))
    parser.add_argument(
        "--execution-events-file",
        default=os.getenv("PAPER_EXECUTION_EVENTS_FILE", "artifacts/paper/paper_execution_events.jsonl"),
    )
    parser.add_argument("--control-file", default=os.getenv("LIVE_CONTROL_FILE", "artifacts/ops/live_control_state.json"))
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        symbols = [s.strip().upper() for s in DEFAULT_SYMBOLS.split(",") if s.strip()]

    state_path = Path(args.state_file)
    history_path = Path(args.history_file)
    execution_event_path = Path(args.execution_events_file)
    control_path = Path(args.control_file)

    if bool(args.loop):
        while True:
            try:
                _run_cycle(
                    api_base=str(args.api_base).rstrip("/"),
                    symbols=symbols,
                    horizon=str(args.horizon),
                    min_confidence=float(args.min_confidence),
                    strategy_id=str(args.strategy_id),
                    capital_per_order_usd=float(args.capital_per_order_usd),
                    max_orders=int(args.max_orders),
                    timeout_sec=float(args.request_timeout_sec),
                    state_path=state_path,
                    history_path=history_path,
                    execution_event_path=execution_event_path,
                    control_path=control_path,
                )
            except Exception as exc:
                err = {
                    "timestamp": _now_iso(),
                    "status": "error",
                    "error": str(exc),
                    "paper_enabled": bool(_load_json(control_path).get("paper_enabled", True)),
                    "live_enabled": bool(_load_json(control_path).get("live_enabled", False)),
                    "symbols": symbols,
                }
                _write_json(state_path, err)
                _append_jsonl(history_path, err)
            time.sleep(max(1.0, float(args.interval_sec)))
    else:
        cycle = _run_cycle(
            api_base=str(args.api_base).rstrip("/"),
            symbols=symbols,
            horizon=str(args.horizon),
            min_confidence=float(args.min_confidence),
            strategy_id=str(args.strategy_id),
            capital_per_order_usd=float(args.capital_per_order_usd),
            max_orders=int(args.max_orders),
            timeout_sec=float(args.request_timeout_sec),
            state_path=state_path,
                history_path=history_path,
                execution_event_path=execution_event_path,
                control_path=control_path,
            )
        print(json.dumps(cycle, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
