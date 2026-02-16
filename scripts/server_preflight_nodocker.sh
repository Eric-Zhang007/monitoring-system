#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MIN_DISK_GB="${MIN_DISK_GB:-20}"
MIN_MEM_GB="${MIN_MEM_GB:-8}"
MIN_GPU_COUNT="${MIN_GPU_COUNT:-0}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
REQUIRE_DB="${REQUIRE_DB:-0}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/backend/models}"
RUN_TESTS="${RUN_TESTS:-1}"
REQUIRE_LIVE_API="${REQUIRE_LIVE_API:-0}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
RUN_SECURITY_VALIDATION="${RUN_SECURITY_VALIDATION:-1}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

if [[ -n "${DATABASE_URL:-}" && ( "${DATABASE_URL}" == *"change_me_please"* || "${DATABASE_URL}" == *"REPLACE_WITH_"* ) ]]; then
  echo "[FAIL] DATABASE_URL still uses placeholder secret; update your .env before preflight."
  exit 2
fi
if [[ "$REQUIRE_DB" == "1" && -z "${DATABASE_URL:-}" ]]; then
  echo "[FAIL] REQUIRE_DB=1 but DATABASE_URL is empty"
  exit 2
fi

need_cmds=(python3 git screen awk sed df curl)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done

if [[ "$RUN_TESTS" == "1" ]]; then
  if ! python3 - <<'PY'
import pytest  # noqa: F401
PY
  then
    echo "[FAIL] pytest is required when RUN_TESTS=1"
    exit 2
  fi
fi

avail_disk_kb="$(df -Pk "$ROOT_DIR" | awk 'NR==2 {print $4}')"
avail_disk_gb=$((avail_disk_kb / 1024 / 1024))
mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)"
mem_gb=$((mem_kb / 1024 / 1024))

if (( avail_disk_gb < MIN_DISK_GB )); then
  echo "[FAIL] low disk: ${avail_disk_gb}GB < ${MIN_DISK_GB}GB"
  exit 2
fi
if (( mem_gb < MIN_MEM_GB )); then
  echo "[FAIL] low memory: ${mem_gb}GB < ${MIN_MEM_GB}GB"
  exit 2
fi

gpu_count="0"
if [[ "$REQUIRE_GPU" == "1" || "$MIN_GPU_COUNT" -gt 0 ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FAIL] nvidia-smi is required when REQUIRE_GPU=1 or MIN_GPU_COUNT>0"
    exit 2
  fi
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | awk '{print $1}')"
  if (( gpu_count < MIN_GPU_COUNT )); then
    echo "[FAIL] gpu count too low: ${gpu_count} < ${MIN_GPU_COUNT}"
    exit 2
  fi
fi

mkdir -p "$MODEL_DIR"
if [[ ! -w "$MODEL_DIR" ]]; then
  echo "[FAIL] model dir is not writable: $MODEL_DIR"
  exit 2
fi

if [[ "$RUN_SECURITY_VALIDATION" == "1" ]]; then
  echo "[1/5] security hardening checks"
  RUN_TESTS=0 bash scripts/validate_security_hardening.sh --skip-tests
else
  echo "[1/5] security hardening checks skipped (RUN_SECURITY_VALIDATION=${RUN_SECURITY_VALIDATION})"
fi

echo "[2/5] python syntax checks (nodocker runtime entry points)"
syntax_files=(
  backend/main.py
  backend/v2_router.py
  backend/task_queue.py
  collector/collector.py
  monitoring/health_check.py
  monitoring/model_ops_scheduler.py
  monitoring/task_worker.py
)
for f in "${syntax_files[@]}"; do
  python3 -m py_compile "$f"
done

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[3/5] targeted nodocker/no-gpu tests"
  PYTHONPATH="${ROOT_DIR}/backend:${ROOT_DIR}/monitoring" python3 -m pytest -q \
    backend/tests/test_task_queue.py \
    backend/tests/test_health_slo.py \
    backend/tests/test_health_postgres_fallback.py \
    backend/tests/test_task_worker_loop.py \
    backend/tests/test_security_config.py
else
  echo "[3/5] tests skipped (RUN_TESTS=${RUN_TESTS})"
fi

echo "[4/5] runtime probes (optional if backend not started)"
if curl -fsS "${API_BASE}/health" >/dev/null 2>&1; then
  echo "[INFO] backend health reachable at ${API_BASE}/health"
  if curl -fsS "${API_BASE}/metrics" >/dev/null 2>&1; then
    echo "[INFO] backend metrics reachable at ${API_BASE}/metrics"
  else
    echo "[WARN] backend metrics endpoint not reachable"
  fi
  if curl -fsS "${API_BASE}/api/v2/risk/limits" >/dev/null 2>&1; then
    echo "[INFO] backend risk limits reachable at ${API_BASE}/api/v2/risk/limits"
  else
    echo "[WARN] backend risk limits endpoint not reachable"
  fi
else
  if [[ "$REQUIRE_LIVE_API" == "1" ]]; then
    echo "[FAIL] backend health probe failed at ${API_BASE}/health"
    exit 2
  fi
  echo "[WARN] backend not running; skipped live API probes"
fi

torch_probe='{"skipped":"gpu_not_required"}'
if [[ "$REQUIRE_GPU" == "1" || "$MIN_GPU_COUNT" -gt 0 ]]; then
  torch_probe="$(python3 - <<'PY'
import json
try:
    import torch
    out = {
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
except Exception as exc:
    out = {"torch_import_error": str(exc)}
print(json.dumps(out, ensure_ascii=False))
PY
  )"
fi
echo "torch_probe=${torch_probe}"

if [[ "$REQUIRE_DB" == "1" ]]; then
  if ! python3 - <<'PY'
import os
import psycopg2
dsn = os.getenv("DATABASE_URL", "")
if not dsn:
    raise SystemExit(2)
conn = psycopg2.connect(dsn)
conn.close()
PY
  then
    echo "[FAIL] DATABASE_URL probe failed"
    exit 2
  fi
fi

git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[5/5] summary"
echo "[OK] nodocker preflight passed"
echo "root_dir=$ROOT_DIR"
echo "git_rev=$git_rev"
echo "disk_gb=$avail_disk_gb"
echo "mem_gb=$mem_gb"
echo "gpu_count=$gpu_count"
echo "model_dir=$MODEL_DIR"
