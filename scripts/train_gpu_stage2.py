#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


def _run(cmd: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(p.returncode),
        "stdout": p.stdout[-4000:],
        "stderr": p.stderr[-4000:],
    }


def _check_module(module_name: str, env: Dict[str, str]) -> Dict[str, Any]:
    p = subprocess.run(
        ["python3", "-c", f"import {module_name}"],
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "module": module_name,
        "ok": bool(p.returncode == 0),
        "stderr": (p.stderr or "").strip()[-1000:],
    }


def _estimate_cost_cny(hours: float, compute_tier: str, a100_hourly_cny: float, billing_discount: float) -> float:
    if compute_tier == "a100x2":
        # a100_hourly_cny is interpreted as the full dual-GPU hourly bill.
        return round(float(hours) * float(a100_hourly_cny) * float(max(0.0, billing_discount)), 2)
    return 0.0


def _build_train_cmd(nproc_per_node: int, env: Dict[str, str]) -> List[str]:
    nproc = max(1, int(nproc_per_node))
    if nproc <= 1:
        return ["python3", "training/main.py"]
    torchrun_path = shutil.which("torchrun", path=env.get("PATH"))
    if torchrun_path:
        return [torchrun_path, "--standalone", f"--nproc_per_node={nproc}", "training/main.py"]
    return ["python3", "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={nproc}", "training/main.py"]


def _api_ready(api_base: str, timeout_sec: float = 3.0) -> bool:
    url = f"{str(api_base).rstrip('/')}/health"
    try:
        r = requests.get(url, timeout=float(timeout_sec))
        if r.status_code < 500:
            return True
    except Exception:
        pass
    return False


def _gpu_inventory(env: Dict[str, str]) -> Dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        env=env,
        capture_output=True,
        text=True,
    )
    names: List[str] = []
    if query.returncode == 0:
        for line in (query.stdout or "").splitlines():
            raw = str(line).strip()
            if not raw:
                continue
            name = raw.split(",", 1)[0].strip()
            if name:
                names.append(name)
    return {
        "nvidia_smi_available": bool(query.returncode == 0),
        "gpu_count": int(len(names)),
        "gpu_names": names,
        "stderr": (query.stderr or "").strip()[-1000:],
    }


def _nvlink_status(env: Dict[str, str]) -> Dict[str, Any]:
    p = subprocess.run(
        ["nvidia-smi", "nvlink", "--status"],
        env=env,
        capture_output=True,
        text=True,
    )
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    up_links = sum(1 for line in out.splitlines() if "Link " in line and ": Up" in line)
    return {
        "checked": bool(p.returncode == 0),
        "up_links": int(up_links),
        "stdout_tail": (p.stdout or "")[-2000:],
        "stderr_tail": (p.stderr or "")[-1000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-2 GPU training orchestrator (strict-asof + external events)")
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("LIQUID_EPOCHS", "24")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("LIQUID_BATCH_SIZE", "128")))
    ap.add_argument("--compute-tier", default=os.getenv("COMPUTE_TIER", "local"), choices=["local", "a100x2"])
    ap.add_argument("--run-optuna", action="store_true")
    ap.add_argument("--optuna-trials", type=int, default=80)
    ap.add_argument("--enable-vc", action="store_true")
    ap.add_argument("--enable-liquid", action="store_true", default=True)
    ap.add_argument("--a100-hourly-cny", type=float, default=float(os.getenv("A100_HOURLY_CNY", "11.96")))
    ap.add_argument("--billing-discount", type=float, default=float(os.getenv("AUTODL_BILLING_DISCOUNT", "1.0")))
    ap.add_argument("--estimated-hours", type=float, default=6.0)
    ap.add_argument("--nproc-per-node", type=int, default=int(os.getenv("TRAIN_NPROC_PER_NODE", "0")))
    ap.add_argument("--require-a100x2", action="store_true", default=os.getenv("REQUIRE_A100X2", "0").lower() in {"1", "true", "yes", "y"})
    ap.add_argument("--require-nvlink", action="store_true", default=os.getenv("REQUIRE_NVLINK", "0").lower() in {"1", "true", "yes", "y"})
    ap.add_argument("--skip-gpu-check", action="store_true", default=os.getenv("SKIP_GPU_CHECK", "0").lower() in {"1", "true", "yes", "y"})
    ap.add_argument("--out", default="artifacts/gpu_stage2/train_gpu_stage2_latest.json")
    args = ap.parse_args()
    if int(args.nproc_per_node) <= 0:
        args.nproc_per_node = 2 if str(args.compute_tier) == "a100x2" else 1

    started = datetime.now(timezone.utc)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[1]
    env.setdefault("MODEL_DIR", str(repo_root / "backend" / "models"))
    env.setdefault("FEATURE_VERSION", "feature-store-v2.1")
    env.setdefault("FEATURE_PAYLOAD_SCHEMA_VERSION", "v2.3")
    env.setdefault("DATA_VERSION", "v1")
    env["LIQUID_SYMBOLS"] = str(args.symbols)
    env["LIQUID_EPOCHS"] = str(int(args.epochs))
    env["LIQUID_BATCH_SIZE"] = str(int(args.batch_size))
    env["TRAIN_RUN_ONCE"] = "1"
    env["TRAIN_ENABLE_VC"] = "1" if bool(args.enable_vc) else "0"
    env["TRAIN_ENABLE_LIQUID"] = "1" if bool(args.enable_liquid) else "0"
    env["TRAIN_NPROC_PER_NODE"] = str(int(args.nproc_per_node))
    env["BACKTEST_ALIGNMENT_MODE"] = "strict_asof"
    env["BACKTEST_ALIGNMENT_VERSION"] = "strict_asof_v1"
    env["BACKTEST_MAX_FEATURE_STALENESS_HOURS"] = str(24 * 14)

    steps: List[Dict[str, Any]] = []
    hardware = {
        "gpu_inventory": {},
        "nvlink": {},
    }
    precheck_reasons: List[str] = []
    if not bool(args.skip_gpu_check):
        hardware["gpu_inventory"] = _gpu_inventory(env=env)
        if hardware["gpu_inventory"].get("nvidia_smi_available"):
            hardware["nvlink"] = _nvlink_status(env=env)
    else:
        hardware["gpu_inventory"] = {"skipped": True}
        hardware["nvlink"] = {"skipped": True}

    if str(args.compute_tier) == "a100x2" and not bool(args.skip_gpu_check):
        gpu_count = int((hardware["gpu_inventory"] or {}).get("gpu_count") or 0)
        gpu_names = [str(x) for x in list((hardware["gpu_inventory"] or {}).get("gpu_names") or [])]
        if gpu_count < 2:
            precheck_reasons.append("gpu_count_lt_2")
        if bool(args.require_a100x2) and any("A100" not in name for name in gpu_names):
            precheck_reasons.append("non_a100_detected")
        if bool(args.require_nvlink):
            nv_up = int((hardware["nvlink"] or {}).get("up_links") or 0)
            if nv_up <= 0:
                precheck_reasons.append("nvlink_not_active")
    if precheck_reasons:
        steps.append(
            {
                "cmd": ["hardware_precheck"],
                "returncode": 2,
                "stdout": "",
                "stderr": ";".join(precheck_reasons),
                "hardware": hardware,
            }
        )

    if bool(args.run_optuna) and (not precheck_reasons):
        if _api_ready(env.get("API_BASE", "http://localhost:8000")):
            steps.append(
                _run(
                    [
                        "python3",
                        "scripts/optuna_liquid_hpo.py",
                        "--compute-tier",
                        str(args.compute_tier),
                        "--n-trials",
                        str(int(args.optuna_trials)),
                        "--a100-hourly-cny",
                        str(float(args.a100_hourly_cny)),
                    ],
                    env=env,
                )
            )
        else:
            steps.append(
                {
                    "cmd": ["python3", "scripts/optuna_liquid_hpo.py"],
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "skipped: backtest API is not reachable, optuna disabled for this run",
                }
            )

    if not precheck_reasons:
        dep = _check_module("torch", env=env)
        if not dep["ok"]:
            steps.append(
                {
                    "cmd": ["python3", "-c", "import torch"],
                    "returncode": 1,
                    "stdout": "",
                    "stderr": f"missing_dependency:torch {dep['stderr']}",
                }
            )
        else:
            train_cmd = _build_train_cmd(int(args.nproc_per_node), env=env)
            steps.append(_run(train_cmd, env=env))

    ok = all(int(s.get("returncode", 1)) == 0 for s in steps)
    finished = datetime.now(timezone.utc)
    total_cost = _estimate_cost_cny(
        hours=float(args.estimated_hours),
        compute_tier=str(args.compute_tier),
        a100_hourly_cny=float(args.a100_hourly_cny),
        billing_discount=float(args.billing_discount),
    )
    out = {
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_sec": round((finished - started).total_seconds(), 3),
        "status": "ok" if ok else "failed",
        "compute_tier": str(args.compute_tier),
        "nproc_per_node": int(args.nproc_per_node),
        "gpu_precheck": {
            "skipped": bool(args.skip_gpu_check),
            "require_a100x2": bool(args.require_a100x2),
            "require_nvlink": bool(args.require_nvlink),
            "reasons": precheck_reasons,
            "hardware": hardware,
        },
        "symbols": [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()],
        "estimated_hours": float(args.estimated_hours),
        "cost_estimate_cny": float(total_cost),
        "steps": steps,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
