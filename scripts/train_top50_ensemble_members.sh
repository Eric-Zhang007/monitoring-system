#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3.12}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python not found: ${PYTHON_BIN}"
  exit 2
fi

CACHE_DIR="${CACHE_DIR:-artifacts/cache/liquid_top50}"
PRIMARY_TIMEFRAME="${PRIMARY_TIMEFRAME:-5m}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAMPLE_STRIDE_BUCKETS="${SAMPLE_STRIDE_BUCKETS:-3}"

declare -A OUT_DIRS=(
  [patchtst]="${OUT_PATCHTST:-artifacts/models/liquid_patchtst}"
  [itransformer]="${OUT_ITRANSFORMER:-artifacts/models/liquid_itransformer}"
  [tft]="${OUT_TFT:-artifacts/models/liquid_tft}"
)

for backbone in patchtst itransformer tft; do
  out_dir="${OUT_DIRS[$backbone]}"
  echo "[train] backbone=${backbone} out=${out_dir}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_top50.py" \
    --cache-dir "${CACHE_DIR}" \
    --primary-timeframe "${PRIMARY_TIMEFRAME}" \
    --backbone "${backbone}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --sample-stride-buckets "${SAMPLE_STRIDE_BUCKETS}" \
    --out-dir "${out_dir}" \
    --model-id "liquid_${backbone}"
done

echo "ensemble_member_training_ok"
