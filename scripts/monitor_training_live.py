#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List


def _run(cmd: List[str]) -> str:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def _run_psql(sql: str) -> str:
    return _run(
        [
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
    )


def _latest_registry_rows(limit: int) -> List[Dict[str, Any]]:
    sql = f"""
    SELECT model_name, track, model_version, created_at::text, COALESCE(metrics::text, '{{}}')
    FROM model_registry
    ORDER BY created_at DESC
    LIMIT {max(1, int(limit))};
    """
    out = _run_psql(sql)
    rows: List[Dict[str, Any]] = []
    for line in out.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        model_name, track, model_version, created_at, metrics_raw = parts
        try:
            metrics = json.loads(metrics_raw)
        except Exception:
            metrics = {}
        rows.append(
            {
                "model_name": model_name,
                "track": track,
                "model_version": model_version,
                "created_at": created_at,
                "metrics": metrics,
            }
        )
    return rows


def _gpu_status() -> Dict[str, Any]:
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus = []
    for line in out.splitlines():
        cols = [x.strip() for x in line.split(",")]
        if len(cols) < 6:
            continue
        idx, name, util, mem_used, mem_total, temp = cols[:6]
        try:
            gpus.append(
                {
                    "index": int(idx),
                    "name": name,
                    "util_pct": float(util),
                    "mem_used_mb": float(mem_used),
                    "mem_total_mb": float(mem_total),
                    "temp_c": float(temp),
                }
            )
        except Exception:
            continue
    return {"available": bool(gpus), "gpus": gpus}


def _readiness() -> Dict[str, Any]:
    out = _run(["python3", "scripts/check_gpu_cutover_readiness.py"])
    if not out:
        return {}
    try:
        return json.loads(out)
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Live monitor for training/retraining readiness")
    ap.add_argument("--interval-sec", type=float, default=15.0)
    ap.add_argument("--iterations", type=int, default=0, help="0 means run forever")
    ap.add_argument("--models-limit", type=int, default=6)
    args = ap.parse_args()

    i = 0
    while True:
        i += 1
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "ts": now,
            "iteration": i,
            "gpu": _gpu_status(),
            "models": _latest_registry_rows(limit=int(args.models_limit)),
            "readiness": _readiness(),
        }
        print(json.dumps(payload, ensure_ascii=False))
        if int(args.iterations) > 0 and i >= int(args.iterations):
            break
        time.sleep(max(1.0, float(args.interval_sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
