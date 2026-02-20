#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
DRY_RUN="${DRY_RUN:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_PHASE_D_BUNDLE="${RUN_PHASE_D_BUNDLE:-1}"
PHASE_D_BUNDLE_DRY_RUN="${PHASE_D_BUNDLE_DRY_RUN:-1}"
REHEARSAL_FINAL_DOWN="${REHEARSAL_FINAL_DOWN:-1}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
PHASE_D_OUT_DIR="${PHASE_D_OUT_DIR:-$ROOT_DIR/artifacts/experiments/phase_d/rehearsal_${TS}}"

cd "$ROOT_DIR"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

run_cmd() {
  local label="$1"
  shift
  echo "[step] ${label}"
  echo "       $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "       (dry-run skip)"
    return 0
  fi
  "$@"
}

echo "[rehearsal] root=${ROOT_DIR}"
echo "[rehearsal] dry_run=${DRY_RUN} run_preflight=${RUN_PREFLIGHT} run_phase_d_bundle=${RUN_PHASE_D_BUNDLE}"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  run_cmd "server preflight" bash scripts/server_preflight.sh
fi

run_cmd "server down (clean start)" bash scripts/server_down.sh
run_cmd "server up" bash scripts/server_up.sh
run_cmd "server readiness" bash scripts/server_readiness.sh

if [[ "$RUN_PHASE_D_BUNDLE" == "1" ]]; then
  if [[ "$PHASE_D_BUNDLE_DRY_RUN" == "1" ]]; then
    run_cmd \
      "phase-d bundle (dry run config)" \
      env \
      PHASE_D_RUN_TRAIN=0 \
      PHASE_D_RUN_BACKBONE=0 \
      PHASE_D_RUN_ABLATION=0 \
      PHASE_D_RUN_REGISTER=0 \
      bash scripts/run_phase_d_multimodal_bundle.sh "$PHASE_D_OUT_DIR"
  else
    run_cmd "phase-d bundle" bash scripts/run_phase_d_multimodal_bundle.sh "$PHASE_D_OUT_DIR"
  fi
fi

if [[ "$REHEARSAL_FINAL_DOWN" == "1" ]]; then
  run_cmd "server down (rollback simulation)" bash scripts/server_down.sh
fi

echo "[OK] rehearsal complete"
echo "phase_d_out_dir=${PHASE_D_OUT_DIR}"
