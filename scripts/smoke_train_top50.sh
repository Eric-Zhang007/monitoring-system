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

AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT00:00:00Z)}"
START="${START:-$(date -u -d '60 days ago' +%Y-%m-%dT00:00:00Z)}"
END="${END:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
TRACK="${TRACK:-liquid}"
TOP_N="${TOP_N:-50}"
BAR_SIZE="${BAR_SIZE:-5m}"
LOOKBACK="${LOOKBACK:-96}"
OUT_MODEL_DIR="${OUT_MODEL_DIR:-artifacts/models/liquid_main}"
CACHE_DIR="${CACHE_DIR:-artifacts/cache/liquid_top50_smoke}"
UNIVERSE_FILE="${UNIVERSE_FILE:-artifacts/universe/liquid_top50_snapshot.json}"
EVAL_DIR="${EVAL_DIR:-artifacts/eval/top50_smoke}"
READINESS_FILE="${READINESS_FILE:-artifacts/audit/top50_data_readiness_latest.json}"
PREPARE_FEATURE_MATRIX="${PREPARE_FEATURE_MATRIX:-1}"
AUDIT_PARALLEL_WORKERS="${AUDIT_PARALLEL_WORKERS:-4}"
BUILD_MULTI_TF_CONTEXT="${BUILD_MULTI_TF_CONTEXT:-1}"
CONTEXT_TIMEFRAMES="${CONTEXT_TIMEFRAMES:-${ANALYST_CONTEXT_TIMEFRAMES:-5m,15m,1h,4h,1d}}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/update_universe_snapshot.py" \
  --track "${TRACK}" \
  --as-of "${AS_OF}" \
  --top-n "${TOP_N}" \
  --rank-by "${RANK_BY:-volume_usd_30d}" \
  --source db \
  --snapshot-file "${UNIVERSE_FILE}" \
  --persist-db 1

UNIVERSE_SYMBOLS="$(UNIVERSE_FILE="${UNIVERSE_FILE}" "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ["UNIVERSE_FILE"])
obj=json.loads(p.read_text(encoding="utf-8"))
syms=[str(x).strip().upper() for x in (obj.get("symbols") or []) if str(x).strip()]
print(",".join(syms))
PY
)"
if [[ -z "${UNIVERSE_SYMBOLS}" ]]; then
  echo "universe symbols empty: ${UNIVERSE_FILE}"
  exit 2
fi

if [[ "${PREPARE_FEATURE_MATRIX}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_feature_store.py" \
    --start "${START}" \
    --end "${END}" \
    --timeframe "${BAR_SIZE}" \
    --symbols "${UNIVERSE_SYMBOLS}"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_feature_views.py" \
    --start "${START}" \
    --end "${END}"
fi

if [[ "${BUILD_MULTI_TF_CONTEXT}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_multi_timeframe_context.py" \
    --symbols "${UNIVERSE_SYMBOLS}" \
    --start "${START}" \
    --end "${END}" \
    --primary-timeframe "${BAR_SIZE}" \
    --context-timeframes "${CONTEXT_TIMEFRAMES}"
fi

if [[ "${AUDIT_PARALLEL_WORKERS}" -gt 1 ]]; then
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_offline_training_data_parallel.py" \
    --track "${TRACK}" \
    --symbols "" \
    --as-of "${AS_OF}" \
    --top-n "${TOP_N}" \
    --start "${START}" \
    --end "${END}" \
    --lookback "${LOOKBACK}" \
    --bucket "${BAR_SIZE}" \
    --parallel-workers "${AUDIT_PARALLEL_WORKERS}" \
    --output "${READINESS_FILE}"
else
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_offline_training_data.py" \
    --track "${TRACK}" \
    --symbols "" \
    --as-of "${AS_OF}" \
    --top-n "${TOP_N}" \
    --start "${START}" \
    --end "${END}" \
    --lookback "${LOOKBACK}" \
    --bucket "${BAR_SIZE}" \
    --output "${READINESS_FILE}"
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_training_cache.py" \
  --universe-snapshot "${UNIVERSE_FILE}" \
  --start "${START}" \
  --end "${END}" \
  --bar-size "${BAR_SIZE}" \
  --lookback-len "${LOOKBACK}" \
  --context-timeframes "${CONTEXT_TIMEFRAMES}" \
  --require-multi-tf-context 1 \
  --readiness-file "${READINESS_FILE}" \
  --exclude-blocked 1 \
  --output-dir "${CACHE_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/check_training_cache_audit.py" --cache-dir "${CACHE_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_top50.py" \
  --cache-dir "${CACHE_DIR}" \
  --epochs 1 \
  --batch-size "${BATCH_SIZE:-64}" \
  --primary-timeframe "${BAR_SIZE}" \
  --sample-stride-buckets "${SAMPLE_STRIDE_BUCKETS:-3}" \
  --out-dir "${OUT_MODEL_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/eval_top50.py" \
  --artifact-dir "${OUT_MODEL_DIR}" \
  --cache-dir "${CACHE_DIR}" \
  --universe-snapshot "${UNIVERSE_FILE}" \
  --out-dir "${EVAL_DIR}"

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/run_inference_smoke.py" \
  --artifact-dir "${OUT_MODEL_DIR}" \
  --cache-dir "${CACHE_DIR}"

LIQUID_MODEL_DIR="${OUT_MODEL_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/inference/main.py" --symbol BTC

echo "smoke_train_top50_ok"
