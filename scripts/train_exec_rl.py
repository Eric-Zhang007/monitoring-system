#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


ACTIONS = ["passive", "balanced", "aggressive"]


def _run_psql(sql: str) -> str:
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "postgres",
        "psql",
        "-U",
        "monitor",
        "-d",
        "monitor",
        "-At",
        "-F",
        "|",
        "-c",
        sql,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or "psql failed")
    return p.stdout.strip()


def _safe_float(v: Any, d: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)


def _safe_int(v: Any, d: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(d)


def _load_execution_rows(lookback_days: int, limit: int) -> List[Dict[str, Any]]:
    sql = f"""
    SELECT
      COALESCE(track, 'liquid') AS track,
      UPPER(COALESCE(target, 'UNK')) AS target,
      LOWER(COALESCE(side, 'buy')) AS side,
      LOWER(COALESCE(status, 'unknown')) AS status,
      COALESCE(quantity, 0)::double precision AS qty,
      COALESCE(est_cost_bps, 0)::double precision AS est_cost_bps,
      COALESCE(metadata->>'vol_bucket', 'normal') AS vol_bucket,
      COALESCE(metadata->>'exec_profile', '') AS exec_profile,
      EXTRACT(HOUR FROM created_at)::int AS hod
    FROM orders_sim
    WHERE created_at > NOW() - make_interval(days => {max(1, int(lookback_days))})
    ORDER BY created_at DESC
    LIMIT {max(100, int(limit))};
    """
    out = _run_psql(sql)
    rows: List[Dict[str, Any]] = []
    for line in out.splitlines():
        cols = line.split("|")
        if len(cols) != 9:
            continue
        track, target, side, status, qty, est_cost_bps, vol_bucket, exec_profile, hod = cols
        rows.append(
            {
                "track": track,
                "target": target,
                "side": side,
                "status": status,
                "qty": _safe_float(qty, 0.0),
                "est_cost_bps": _safe_float(est_cost_bps, 0.0),
                "vol_bucket": vol_bucket,
                "exec_profile": exec_profile,
                "hod": _safe_int(hod, 0),
            }
        )
    return rows


def _action_from_profile(profile: str, est_cost_bps: float, vol_bucket: str) -> int:
    p = str(profile or "").strip().lower()
    if p in {"passive", "balanced", "aggressive"}:
        return ACTIONS.index(p)
    if est_cost_bps >= 20.0 or vol_bucket == "high":
        return ACTIONS.index("passive")
    if est_cost_bps <= 6.0 and vol_bucket != "high":
        return ACTIONS.index("aggressive")
    return ACTIONS.index("balanced")


def _state_vec(row: Dict[str, Any]) -> np.ndarray:
    qty = max(0.0, _safe_float(row.get("qty"), 0.0))
    cost = max(0.0, _safe_float(row.get("est_cost_bps"), 0.0))
    side = -1.0 if str(row.get("side")).lower() == "sell" else 1.0
    vol_high = 1.0 if str(row.get("vol_bucket")).lower() == "high" else 0.0
    hod = _safe_int(row.get("hod"), 0)
    cyc_s = math.sin((2.0 * math.pi * hod) / 24.0)
    cyc_c = math.cos((2.0 * math.pi * hod) / 24.0)
    return np.array(
        [
            1.0,
            side,
            math.tanh(qty / 2.0),
            min(1.0, cost / 40.0),
            vol_high,
            cyc_s,
            cyc_c,
        ],
        dtype=np.float64,
    )


def _reward(row: Dict[str, Any], action_idx: int) -> float:
    status = str(row.get("status") or "").lower()
    filled = 1.0 if status == "filled" else 0.0
    rejected = 1.0 if status == "rejected" else 0.0
    cost = max(0.0, _safe_float(row.get("est_cost_bps"), 0.0)) / 10000.0
    base = 0.0015 * filled - 0.0020 * rejected - 0.8 * cost
    if action_idx == ACTIONS.index("aggressive"):
        base -= 0.0008 * (1.0 if str(row.get("vol_bucket")).lower() == "high" else 0.4)
    elif action_idx == ACTIONS.index("passive"):
        base += 0.0004 * (1.0 if cost > 0.001 else 0.3)
    return float(base)


def _augment(rows: List[Dict[str, Any]], target_size: int, seed: int) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    out = list(rows)
    while len(out) < target_size:
        out.append(
            {
                "track": "liquid",
                "target": rnd.choice(["BTC", "ETH", "SOL", "XRP", "DOGE"]),
                "side": rnd.choice(["buy", "sell"]),
                "status": rnd.choices(["filled", "rejected", "partial"], weights=[0.8, 0.1, 0.1])[0],
                "qty": rnd.uniform(0.01, 2.5),
                "est_cost_bps": rnd.uniform(2.0, 24.0),
                "vol_bucket": rnd.choice(["normal", "high"]),
                "exec_profile": "",
                "hod": rnd.randint(0, 23),
            }
        )
    return out


def _fit_linear_q(states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, l2: float) -> Dict[str, Any]:
    dim = int(states.shape[1])
    weights = np.zeros((len(ACTIONS), dim), dtype=np.float64)
    action_counts = {}
    for ai in range(len(ACTIONS)):
        mask = actions == ai
        action_counts[ACTIONS[ai]] = int(mask.sum())
        if not np.any(mask):
            continue
        x = states[mask]
        y = rewards[mask]
        xtx = x.T @ x + l2 * np.eye(dim, dtype=np.float64)
        xty = x.T @ y
        w = np.linalg.solve(xtx, xty)
        weights[ai] = w
    pred = np.max(states @ weights.T, axis=1)
    mse = float(np.mean((pred - rewards) ** 2))
    return {
        "weights": weights.tolist(),
        "mse": mse,
        "action_counts": action_counts,
        "feature_dim": dim,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train offline RL execution policy (contextual linear-Q)")
    ap.add_argument("--lookback-days", type=int, default=120)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--min-samples", type=int, default=2000)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="backend/models/execution_rl_policy_v1.json")
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    started = datetime.now(timezone.utc)
    rows = _load_execution_rows(lookback_days=int(args.lookback_days), limit=int(args.limit))
    real_samples = len(rows)
    if real_samples < int(args.min_samples):
        rows = _augment(rows, target_size=int(args.min_samples), seed=int(args.seed))

    states = np.stack([_state_vec(r) for r in rows], axis=0)
    actions = np.array([_action_from_profile(str(r.get("exec_profile") or ""), _safe_float(r.get("est_cost_bps"), 0.0), str(r.get("vol_bucket") or "")) for r in rows], dtype=np.int64)
    rewards = np.array([_reward(r, int(actions[i])) for i, r in enumerate(rows)], dtype=np.float64)
    model = _fit_linear_q(states=states, actions=actions, rewards=rewards, l2=float(args.l2))

    out = {
        "model_name": "execution_rl_policy",
        "model_version": "v1.0",
        "type": "contextual_linear_q",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actions": ACTIONS,
        "features": ["bias", "side_sign", "qty_tanh", "est_cost_norm", "vol_high", "hod_sin", "hod_cos"],
        "weights": model["weights"],
        "train_metrics": {
            "mse": round(float(model["mse"]), 9),
            "samples_total": int(states.shape[0]),
            "samples_real": int(real_samples),
            "samples_augmented": int(max(0, states.shape[0] - real_samples)),
            "action_counts": model["action_counts"],
            "l2": float(args.l2),
            "seed": int(args.seed),
        },
        "policy_runtime": {
            "default_action": "balanced",
            "risk_rules": {
                "high_vol_prefer": "passive",
                "low_cost_prefer": "aggressive",
            },
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    done = datetime.now(timezone.utc)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(out_path),
                "samples": int(states.shape[0]),
                "real_samples": int(real_samples),
                "duration_sec": round((done - started).total_seconds(), 3),
                "mse": round(float(model["mse"]), 9),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
