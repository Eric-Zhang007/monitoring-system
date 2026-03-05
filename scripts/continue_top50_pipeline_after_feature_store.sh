#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

START="${START:-2018-01-01T00:00:00Z}"
END="${END:-2026-03-04T00:00:00Z}"
AS_OF="${AS_OF:-2026-03-04T00:00:00Z}"
TRACK="${TRACK:-liquid}"
TOP_N="${TOP_N:-50}"
BAR_SIZE="${BAR_SIZE:-5m}"
READINESS_OUT="${READINESS_OUT:-${ROOT_DIR}/artifacts/audit/top50_data_readiness_latest.json}"

echo "[continue_top50_pipeline] wait for feature_store finish..."
while pgrep -f "scripts/build_feature_store.py" >/dev/null 2>&1; do
  echo "[continue_top50_pipeline] feature_store still running, sleep 30s"
  sleep 30
done
echo "[continue_top50_pipeline] feature_store finished, continue next steps"

echo "[1/3] build_text_embeddings"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_text_embeddings.py" \
  --database-url "${DATABASE_URL}" \
  --start "${START}" \
  --end "${END}" \
  --truncate

echo "[2/3] merge_feature_views"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_feature_views.py" \
  --database-url "${DATABASE_URL}" \
  --start "${START}" \
  --end "${END}" \
  --truncate

echo "[3/3] audit_offline_training_data"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_offline_training_data.py" \
  --database-url "${DATABASE_URL}" \
  --track "${TRACK}" \
  --as-of "${AS_OF}" \
  --top-n "${TOP_N}" \
  --start "${START}" \
  --end "${END}" \
  --lookback 288 \
  --bucket "${BAR_SIZE}" \
  --output "${READINESS_OUT}"

echo "[continue_top50_pipeline] done"
