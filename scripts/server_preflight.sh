#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MIN_DISK_GB="${MIN_DISK_GB:-50}"
MIN_MEM_GB="${MIN_MEM_GB:-12}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose "$@"
    return
  fi
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
    return
  fi
  echo "docker compose not found" >&2
  return 127
}

need_cmds=(docker git tar gzip ssh scp awk sed)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done

if ! docker info >/dev/null 2>&1; then
  echo "[FAIL] docker daemon unreachable"
  exit 2
fi

if ! compose_cmd config >/dev/null; then
  echo "[FAIL] docker compose config invalid"
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

if [[ "$REQUIRE_GPU" == "1" ]]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[FAIL] REQUIRE_GPU=1 but nvidia-smi missing"
    exit 2
  fi
  nvidia-smi >/dev/null
fi

git_rev="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
services="$(compose_cmd config --services | tr '\n' ' ')"

echo "[OK] preflight passed"
echo "root_dir=$ROOT_DIR"
echo "git_rev=$git_rev"
echo "disk_gb=$avail_disk_gb"
echo "mem_gb=$mem_gb"
echo "services=$services"
