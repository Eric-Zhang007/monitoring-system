#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import RealDictCursor


DEFAULT_SYMBOLS = "BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK"


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


def _fetch_latest_predictions(cur, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    cur.execute(
        """
        WITH ranked AS (
          SELECT
            target,
            outputs,
            created_at,
            ROW_NUMBER() OVER (PARTITION BY target ORDER BY created_at DESC) AS rn
          FROM predictions_v2
          WHERE track = 'liquid'
            AND target = ANY(%s)
        )
        SELECT target, outputs, created_at
        FROM ranked
        WHERE rn = 1
        """,
        (symbols,),
    )
    out: Dict[str, Dict[str, Any]] = {}
    for r in cur.fetchall() or []:
        out[str(r.get("target") or "").upper()] = {
            "outputs": r.get("outputs") if isinstance(r.get("outputs"), dict) else {},
            "created_at": r.get("created_at"),
        }
    return out


def _fetch_latest_prices(cur, symbols: List[str], timeframe: str) -> Dict[str, float]:
    cur.execute(
        """
        WITH ranked AS (
          SELECT
            symbol,
            close::double precision AS close,
            ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY ts DESC) AS rn
          FROM market_bars
          WHERE symbol = ANY(%s)
            AND timeframe = %s
        )
        SELECT symbol, close
        FROM ranked
        WHERE rn = 1
        """,
        (symbols, timeframe),
    )
    out: Dict[str, float] = {}
    for r in cur.fetchall() or []:
        out[str(r.get("symbol") or "").upper()] = float(r.get("close") or 0.0)
    return out


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


def _run_cycle(
    *,
    db_url: str,
    symbols: List[str],
    timeframe: str,
    state_path: Path,
    history_path: Path,
    control_path: Path,
) -> Dict[str, Any]:
    state = _load_json(state_path)
    control = _load_json(control_path)
    live_enabled = bool(control.get("live_enabled", False))
    paper_enabled = bool(control.get("paper_enabled", True))

    prev_equity = float(state.get("equity", 1.0) or 1.0)
    prev_prices = state.get("last_prices") if isinstance(state.get("last_prices"), dict) else {}
    prev_positions = state.get("positions") if isinstance(state.get("positions"), dict) else {}

    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            preds = _fetch_latest_predictions(cur, symbols)
            prices = _fetch_latest_prices(cur, symbols, timeframe=timeframe)

    if not prices:
        cycle = {
            "timestamp": _now_iso(),
            "status": "no_prices",
            "equity": prev_equity,
            "paper_enabled": paper_enabled,
            "live_enabled": live_enabled,
            "symbols": symbols,
        }
        _write_json(state_path, cycle)
        _append_jsonl(history_path, cycle)
        return cycle

    pnl = 0.0
    next_positions: Dict[str, float] = {}
    for sym in symbols:
        px = float(prices.get(sym, 0.0) or 0.0)
        old_px = float(prev_prices.get(sym, px) or px)
        old_pos = float(prev_positions.get(sym, 0.0) or 0.0)
        if old_px > 0:
            pnl += old_pos * ((px - old_px) / old_px)

        pred_payload = preds.get(sym, {}).get("outputs") if isinstance(preds.get(sym), dict) else {}
        next_pos = _target_position(pred_payload) if paper_enabled else 0.0
        next_positions[sym] = float(next_pos)

    equity = float(max(0.01, prev_equity * (1.0 + pnl)))

    cycle = {
        "timestamp": _now_iso(),
        "status": "ok",
        "paper_enabled": paper_enabled,
        "live_enabled": live_enabled,
        "equity": equity,
        "equity_change": float(equity - prev_equity),
        "pnl_ratio": float(pnl),
        "positions": next_positions,
        "last_prices": prices,
        "prediction_count": len(preds),
        "timeframe": timeframe,
        "symbols": symbols,
    }

    _write_json(state_path, cycle)
    _append_jsonl(history_path, cycle)
    return cycle


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous paper-trading daemon with manual live control")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", DEFAULT_SYMBOLS))
    parser.add_argument("--timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    parser.add_argument("--interval-sec", type=float, default=float(os.getenv("PAPER_DAEMON_INTERVAL_SEC", "60")))
    parser.add_argument("--state-file", default=os.getenv("PAPER_STATE_FILE", "artifacts/paper/paper_state.json"))
    parser.add_argument("--history-file", default=os.getenv("PAPER_HISTORY_FILE", "artifacts/paper/paper_history.jsonl"))
    parser.add_argument("--control-file", default=os.getenv("LIVE_CONTROL_FILE", "artifacts/ops/live_control_state.json"))
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    if not symbols:
        symbols = [s.strip().upper() for s in DEFAULT_SYMBOLS.split(",") if s.strip()]

    state_path = Path(args.state_file)
    history_path = Path(args.history_file)
    control_path = Path(args.control_file)

    if bool(args.loop):
        while True:
            _run_cycle(
                db_url=args.database_url,
                symbols=symbols,
                timeframe=str(args.timeframe),
                state_path=state_path,
                history_path=history_path,
                control_path=control_path,
            )
            time.sleep(max(1.0, float(args.interval_sec)))
    else:
        cycle = _run_cycle(
            db_url=args.database_url,
            symbols=symbols,
            timeframe=str(args.timeframe),
            state_path=state_path,
            history_path=history_path,
            control_path=control_path,
        )
        print(json.dumps(cycle, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
