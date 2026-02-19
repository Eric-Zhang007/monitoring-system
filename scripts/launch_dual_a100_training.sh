#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
cd "$ROOT_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "[ERR] python not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[ERR] missing command: $PYTHON_BIN" >&2; exit 2; }
fi

REQUIRE_A100="${REQUIRE_A100:-1}"
REQUIRE_NVLINK="${REQUIRE_NVLINK:-1}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"
DRY_RUN="${DRY_RUN:-0}"
NPROC="${NPROC:-2}"
OUT_JSON="${OUT_JSON:-artifacts/gpu_stage2/train_gpu_stage2_dual_a100.json}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN="1"; shift 1 ;;
    --skip-gpu-check) SKIP_GPU_CHECK="1"; shift 1 ;;
    --require-a100) REQUIRE_A100="1"; shift 1 ;;
    --allow-non-a100) REQUIRE_A100="0"; shift 1 ;;
    --require-nvlink) REQUIRE_NVLINK="1"; shift 1 ;;
    --allow-no-nvlink) REQUIRE_NVLINK="0"; shift 1 ;;
    --nproc) NPROC="$2"; shift 2 ;;
    --out) OUT_JSON="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift 1 ;;
  esac
done

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

if [[ "$SKIP_GPU_CHECK" != "1" ]]; then
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[ERR] nvidia-smi not found" >&2; exit 2; }
  mapfile -t GPU_NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  GPU_COUNT="${#GPU_NAMES[@]}"
  log "detected_gpus=${GPU_COUNT}"
  if [[ "$GPU_COUNT" -lt 2 ]]; then
    echo "[ERR] dual-GPU required, detected=${GPU_COUNT}" >&2
    exit 2
  fi

  if [[ "$REQUIRE_A100" == "1" ]]; then
    for name in "${GPU_NAMES[@]}"; do
      if [[ "$name" != *"A100"* ]]; then
        echo "[ERR] non-A100 gpu detected: ${name}" >&2
        exit 2
      fi
    done
  fi

  NVLINK_UP=0
  if nvidia-smi nvlink --status >/tmp/nvlink_status.log 2>&1; then
    NVLINK_UP="$(grep -E 'Link [0-9]+: Up' /tmp/nvlink_status.log | wc -l | tr -d ' ')"
  fi
  log "nvlink_up_links=${NVLINK_UP}"
  if [[ "$REQUIRE_NVLINK" == "1" && "${NVLINK_UP}" -le 0 ]]; then
    echo "[ERR] nvlink not active (require_nvlink=1)" >&2
    cat /tmp/nvlink_status.log || true
    exit 2
  fi
fi

export TRAIN_NPROC_PER_NODE="${NPROC}"
export LIQUID_SYMBOL_DDP=1
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

CMD=(
  "$PYTHON_BIN" scripts/train_gpu_stage2.py
  --compute-tier a100x2
  --nproc-per-node "$NPROC"
  --enable-liquid
  --out "$OUT_JSON"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

log "launch_cmd=${CMD[*]}"
if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

"${CMD[@]}"
