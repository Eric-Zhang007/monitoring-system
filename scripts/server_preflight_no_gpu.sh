#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MIN_DISK_GB="${MIN_DISK_GB:-20}"
MIN_MEM_GB="${MIN_MEM_GB:-8}"
RUN_TESTS="${RUN_TESTS:-1}"
REQUIRE_LIVE_API="${REQUIRE_LIVE_API:-0}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"

need_cmds=(python3 awk sed df curl)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done

if [[ "$RUN_TESTS" == "1" ]] && ! python3 - <<'PY'
import pytest  # noqa: F401
PY
then
  echo "[FAIL] pytest is required when RUN_TESTS=1"
  exit 2
fi

if [[ "${REQUIRE_GPU:-0}" == "1" ]]; then
  echo "[FAIL] REQUIRE_GPU=1 is incompatible with server_preflight_no_gpu.sh"
  exit 2
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

echo "[1/4] python syntax checks (non-training entry points)"
syntax_files=(
  backend/main.py
  backend/v2_router.py
  backend/task_queue.py
  collector/collector.py
  inference/main.py
  monitoring/health_check.py
  monitoring/task_worker.py
)
for f in "${syntax_files[@]}"; do
  python3 -m py_compile "$f"
done

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[2/4] targeted no-GPU tests"
  PYTHONPATH="${ROOT_DIR}/backend:${ROOT_DIR}/monitoring" python3 -m pytest -q \
    backend/tests/test_task_queue.py \
    backend/tests/test_health_slo.py \
    backend/tests/test_health_postgres_fallback.py \
    backend/tests/test_task_worker_loop.py
else
  echo "[2/4] tests skipped (RUN_TESTS=${RUN_TESTS})"
fi

echo "[3/4] live API probe (optional)"
if curl -fsS "${API_BASE}/health" >/dev/null 2>&1; then
  echo "[INFO] backend health reachable at ${API_BASE}/health"
  if curl -fsS "${API_BASE}/metrics" >/dev/null 2>&1; then
    echo "[INFO] backend metrics reachable at ${API_BASE}/metrics"
  else
    echo "[WARN] backend metrics endpoint not reachable"
  fi
else
  if [[ "$REQUIRE_LIVE_API" == "1" ]]; then
    echo "[FAIL] backend health probe failed at ${API_BASE}/health"
    exit 2
  fi
  echo "[WARN] backend not running; skipped live API probe"
fi

git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[4/4] summary"
echo "[OK] no-GPU preflight passed"
echo "root_dir=$ROOT_DIR"
echo "git_rev=$git_rev"
echo "disk_gb=$avail_disk_gb"
echo "mem_gb=$mem_gb"
