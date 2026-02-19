#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_symbols(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in str(raw or "").split(","):
        sym = item.strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


def _http_session() -> requests.Session:
    sess = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess


def _try_parse_last_json(text: str) -> Optional[Dict[str, Any]]:
    rows = [r.strip() for r in str(text or "").splitlines() if r.strip()]
    if not rows:
        return None
    for row in reversed(rows):
        if not (row.startswith("{") and row.endswith("}")):
            continue
        try:
            parsed = json.loads(row)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def _run_cmd(cmd_text: str, timeout_sec: int) -> Dict[str, Any]:
    cmd_text = str(cmd_text or "").strip()
    if not cmd_text:
        return {"status": "skipped", "reason": "empty_command"}
    cmd = shlex.split(cmd_text)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(10, int(timeout_sec)))
        parsed = _try_parse_last_json(p.stdout)
        return {
            "status": "ok" if p.returncode == 0 else "failed",
            "returncode": int(p.returncode),
            "elapsed_sec": round(time.time() - t0, 3),
            "cmd": cmd,
            "stdout_tail": (p.stdout or "")[-1500:],
            "stderr_tail": (p.stderr or "")[-1000:],
            "parsed_json": parsed,
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "returncode": 124,
            "elapsed_sec": round(time.time() - t0, 3),
            "cmd": cmd,
        }
    except Exception as exc:
        return {
            "status": "error",
            "returncode": 2,
            "elapsed_sec": round(time.time() - t0, 3),
            "cmd": cmd,
            "error": str(exc),
        }


def _call_json(
    sess: requests.Session,
    method: str,
    url: str,
    *,
    payload: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 12.0,
) -> Dict[str, Any]:
    method = method.upper().strip()
    resp = sess.request(
        method=method,
        url=url,
        json=payload,
        params=params,
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


def _run_paper_cycle(
    sess: requests.Session,
    *,
    api_base: str,
    symbols: List[str],
    horizon: str,
    min_confidence: float,
    strategy_id: str,
    capital_per_order_usd: float,
    max_orders: int,
) -> Dict[str, Any]:
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
            )
        except Exception as exc:
            failures.append({"symbol": sym, "stage": "signal", "error": str(exc)})
            continue
        signals.append(sig)
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
            )
            outputs = pred.get("outputs") if isinstance(pred.get("outputs"), dict) else {}
            px = float(pred.get("current_price") or outputs.get("current_price") or 0.0)
        except Exception as exc:
            failures.append({"symbol": sym, "stage": "price", "error": str(exc)})
            continue

        if px <= 0:
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
                "metadata": {"source": "continuous_ops_loop", "signal_confidence": confidence},
            }
        )

    if len(orders) > max(1, int(max_orders)):
        orders = orders[: max(1, int(max_orders))]

    if not orders:
        return {
            "status": "no_orders",
            "signals": len(signals),
            "orders": 0,
            "failures": failures[:20],
        }

    submit = _call_json(
        sess,
        "POST",
        f"{api_base}/api/v2/execution/orders",
        payload={
            "adapter": "paper",
            "venue": "coinbase",
            "time_in_force": "IOC",
            "max_slippage_bps": 12.0,
            "orders": orders,
        },
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
            "max_orders": len(orders),
            "limit_timeout_sec": 2.0,
            "max_retries": 1,
            "fee_bps": 5.0,
        },
    )

    return {
        "status": "ok",
        "decision_id": decision_id,
        "signals": len(signals),
        "orders": len(orders),
        "filled": int(run_out.get("filled") or 0),
        "rejected": int(run_out.get("rejected") or 0),
        "reject_breakdown": dict(run_out.get("reject_breakdown") or {}),
        "failures": failures[:20],
    }


def _write_state(latest_file: Path, history_file: Path, payload: Dict[str, Any]) -> None:
    latest_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    latest_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with history_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
        obj = json.loads(raw) if raw.strip() else {}
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Continuous paper-trading + retraining ops loop (monitoring only)")
    ap.add_argument("--api-base", default=os.getenv("API_BASE", "http://localhost:8000"))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--horizon", default="1d", choices=["1h", "1d", "7d"])
    ap.add_argument("--min-confidence", type=float, default=float(os.getenv("PAPER_MIN_CONFIDENCE", "0.45")))
    ap.add_argument("--strategy-id", default=os.getenv("PAPER_STRATEGY_ID", "continuous-paper-v1"))
    ap.add_argument("--capital-per-order-usd", type=float, default=float(os.getenv("PAPER_CAPITAL_PER_ORDER_USD", "300.0")))
    ap.add_argument("--max-orders", type=int, default=int(os.getenv("PAPER_MAX_ORDERS_PER_CYCLE", "8")))
    ap.add_argument("--cycles", type=int, default=int(os.getenv("OPS_CYCLES", "1")))
    ap.add_argument("--loop", action="store_true", help="run forever")
    ap.add_argument("--interval-sec", type=float, default=float(os.getenv("OPS_INTERVAL_SEC", "900")))
    ap.add_argument("--cmd-timeout-sec", type=int, default=int(os.getenv("OPS_CMD_TIMEOUT_SEC", "7200")))
    ap.add_argument("--ingest-cmd", default=os.getenv("OPS_INGEST_CMD", ""))
    ap.add_argument(
        "--train-cmd",
        default=os.getenv(
            "OPS_TRAIN_CMD",
            "python3 scripts/train_gpu_stage2.py --enable-liquid --epochs 8 --batch-size 128 --out artifacts/gpu_stage2/train_gpu_stage2_latest.json",
        ),
    )
    ap.add_argument(
        "--backtest-cmd",
        default=os.getenv(
            "OPS_BACKTEST_CMD",
            "python3 scripts/run_prod_live_backtest_batch.py --n-runs 6 --lookback-days 180 --train-days 56 --test-days 14",
        ),
    )
    ap.add_argument(
        "--parity-cmd",
        default=os.getenv(
            "OPS_PARITY_CMD",
            "python3 scripts/check_backtest_paper_parity.py --track liquid --max-deviation 0.10 --min-completed-runs 5 --data-regimes prod_live",
        ),
    )
    ap.add_argument(
        "--audit-cmd",
        default=os.getenv(
            "OPS_AUDIT_CMD",
            "python3 scripts/audit_full_history_completeness.py --start 2018-01-01T00:00:00Z --timeframe 5m",
        ),
    )
    ap.add_argument(
        "--profit-gate-cmd",
        default=os.getenv(
            "OPS_PROFIT_GATE_CMD",
            "python3 scripts/gate_training_profitability.py --track liquid --lookback-hours 168",
        ),
    )
    ap.add_argument(
        "--finetune-cmd",
        default=os.getenv(
            "OPS_FINETUNE_CMD",
            "python3 scripts/optuna_liquid_hpo.py --compute-tier local --n-trials 20",
        ),
    )
    ap.add_argument(
        "--paper-after-first-train",
        action="store_true",
        default=os.getenv("OPS_PAPER_AFTER_FIRST_TRAIN", "1").strip().lower() in {"1", "true", "yes", "y"},
    )
    ap.add_argument(
        "--run-finetune-on-gate-fail",
        action="store_true",
        default=os.getenv("OPS_RUN_FINETUNE_ON_GATE_FAIL", "1").strip().lower() in {"1", "true", "yes", "y"},
    )
    ap.add_argument("--state-file", default=os.getenv("OPS_STATE_FILE", "artifacts/ops/continuous_runtime_state.json"))
    ap.add_argument("--history-file", default=os.getenv("OPS_HISTORY_FILE", "artifacts/ops/continuous_runtime_history.jsonl"))
    args = ap.parse_args()

    symbols = _parse_symbols(args.symbols)
    if not symbols:
        raise SystemExit("empty symbols")

    api_base = str(args.api_base).rstrip("/")
    latest_file = Path(str(args.state_file))
    history_file = Path(str(args.history_file))
    sess = _http_session()
    first_train_ok = bool(_load_state(latest_file).get("first_train_ok", False))

    cycle_id = 0
    target_cycles = max(1, int(args.cycles))
    while True:
        cycle_id += 1
        state: Dict[str, Any] = {
            "status": "ok",
            "operator_controlled_live_switch": True,
            "cycle": cycle_id,
            "started_at": _now_iso(),
            "symbols": symbols,
            "ingest": {},
            "train": {},
            "backtest": {},
            "parity": {},
            "audit": {},
            "profit_gate": {},
            "finetune": {},
            "paper": {},
            "first_train_ok_before": bool(first_train_ok),
            "first_train_ok": bool(first_train_ok),
            "errors": [],
        }

        state["ingest"] = _run_cmd(str(args.ingest_cmd), int(args.cmd_timeout_sec))
        if state["ingest"].get("status") in {"failed", "error", "timeout"}:
            state["status"] = "degraded"
            state["errors"].append("ingest_failed")

        state["train"] = _run_cmd(str(args.train_cmd), int(args.cmd_timeout_sec))
        train_ok = str(state["train"].get("status") or "").strip().lower() == "ok"
        if not train_ok:
            state["status"] = "degraded"
            state["errors"].append("train_failed")
        if train_ok:
            first_train_ok = True
        state["first_train_ok"] = bool(first_train_ok)

        state["backtest"] = _run_cmd(str(args.backtest_cmd), int(args.cmd_timeout_sec))
        if state["backtest"].get("status") in {"failed", "error", "timeout"}:
            state["status"] = "degraded"
            state["errors"].append("backtest_failed")

        state["parity"] = _run_cmd(str(args.parity_cmd), int(args.cmd_timeout_sec))
        if state["parity"].get("status") in {"failed", "error", "timeout"}:
            state["status"] = "degraded"
            state["errors"].append("parity_failed")

        state["audit"] = _run_cmd(str(args.audit_cmd), int(args.cmd_timeout_sec))
        if state["audit"].get("status") in {"failed", "error", "timeout"}:
            state["status"] = "degraded"
            state["errors"].append("audit_failed")

        state["profit_gate"] = _run_cmd(str(args.profit_gate_cmd), int(args.cmd_timeout_sec))
        profit_gate_status = str(state["profit_gate"].get("status") or "unknown").strip().lower()
        profit_gate_passed = True
        if profit_gate_status in {"failed", "error", "timeout"}:
            profit_gate_passed = False
        elif profit_gate_status == "ok":
            parsed = state["profit_gate"].get("parsed_json")
            if isinstance(parsed, dict) and ("passed" in parsed):
                profit_gate_passed = bool(parsed.get("passed"))
        state["profit_gate"]["passed"] = bool(profit_gate_passed)
        if not profit_gate_passed:
            state["status"] = "degraded"
            state["errors"].append("profit_gate_failed")

        run_finetune = bool(args.run_finetune_on_gate_fail and (not profit_gate_passed))
        if run_finetune:
            state["finetune"] = _run_cmd(str(args.finetune_cmd), int(args.cmd_timeout_sec))
            if state["finetune"].get("status") in {"failed", "error", "timeout"}:
                state["status"] = "degraded"
                state["errors"].append("finetune_failed")
        else:
            state["finetune"] = {
                "status": "skipped",
                "reason": "profit_gate_passed_or_finetune_disabled",
            }

        should_run_paper = True
        if bool(args.paper_after_first_train) and (not first_train_ok):
            should_run_paper = False
        if should_run_paper:
            try:
                state["paper"] = _run_paper_cycle(
                    sess,
                    api_base=api_base,
                    symbols=symbols,
                    horizon=str(args.horizon),
                    min_confidence=float(args.min_confidence),
                    strategy_id=str(args.strategy_id),
                    capital_per_order_usd=float(args.capital_per_order_usd),
                    max_orders=max(1, int(args.max_orders)),
                )
            except Exception as exc:
                state["paper"] = {"status": "error", "error": str(exc)}
                state["errors"].append(f"paper:{exc}")
                state["status"] = "degraded"
        else:
            state["paper"] = {
                "status": "skipped",
                "reason": "waiting_first_successful_train",
            }

        state["ended_at"] = _now_iso()
        _write_state(latest_file, history_file, state)

        if not bool(args.loop):
            if cycle_id >= target_cycles:
                break
        time.sleep(max(5.0, float(args.interval_sec)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
