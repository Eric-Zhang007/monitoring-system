#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MIN_DISK_GB="${MIN_DISK_GB:-80}"
MIN_MEM_GB="${MIN_MEM_GB:-24}"
MIN_GPU_COUNT="${MIN_GPU_COUNT:-1}"
REQUIRE_DB="${REQUIRE_DB:-0}"

need_cmds=(python3 git screen awk sed df)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[FAIL] missing command: nvidia-smi"
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

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | awk '{print $1}')"
if (( gpu_count < MIN_GPU_COUNT )); then
  echo "[FAIL] gpu count too low: ${gpu_count} < ${MIN_GPU_COUNT}"
  exit 2
fi

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
echo "[OK] nodocker preflight passed"
echo "root_dir=$ROOT_DIR"
echo "git_rev=$git_rev"
echo "disk_gb=$avail_disk_gb"
echo "mem_gb=$mem_gb"
echo "gpu_count=$gpu_count"
