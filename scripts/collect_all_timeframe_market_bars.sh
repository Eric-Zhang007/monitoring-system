#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
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
  echo "[FAIL] python missing: ${PYTHON_BIN}"
  exit 2
fi
if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[FAIL] DATABASE_URL is required (set in env or .env)"
  exit 2
fi

START="${START:-2018-01-01T00:00:00Z}"
END="${END:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
MARKET="${MARKET:-perp}"
SYMBOLS="${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT}"
SYMBOL_MAP="${SYMBOL_MAP:-BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK}"
TIMEFRAMES="${TIMEFRAMES:-1m,3m,5m,15m,30m,1h,4h,6h,12h,1d}"
MAX_WORKERS="${MAX_WORKERS:-4}"
CHUNK_DAYS="${CHUNK_DAYS:-30}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-artifacts/checkpoints/all_timeframes}"
AUDIT_DIR="${AUDIT_DIR:-artifacts/audit/all_timeframes}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-3650}"
FALLBACK_SOURCE="${FALLBACK_SOURCE:-none}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
MIN_WORKERS="${MIN_WORKERS:-1}"
ADAPTIVE_WORKERS="${ADAPTIVE_WORKERS:-1}"
RATE_LIMIT_BACKOFF_SEC="${RATE_LIMIT_BACKOFF_SEC:-5}"
MAX_RATE_LIMIT_BACKOFF_SEC="${MAX_RATE_LIMIT_BACKOFF_SEC:-120}"
DB_PAGE_SIZE="${DB_PAGE_SIZE:-1200}"
DB_COMMIT_EVERY_SYMBOLS="${DB_COMMIT_EVERY_SYMBOLS:-4}"
SPLIT_FAILED_SYMBOLS="${SPLIT_FAILED_SYMBOLS:-1}"
MIN_SPLIT_CHUNK_DAYS="${MIN_SPLIT_CHUNK_DAYS:-2}"
STATUS_DIR="${STATUS_DIR:-artifacts/runtime/bg/status}"
PROGRESS_FILE="${PROGRESS_FILE:-${STATUS_DIR}/full_history_progress.json}"
LAST_EXIT_FILE="${LAST_EXIT_FILE:-${STATUS_DIR}/full_history_last_exit.json}"
CURRENT_TF="bootstrap"

mkdir -p "${CHECKPOINT_DIR}" "${AUDIT_DIR}" "${STATUS_DIR}"

write_status_json() {
  local kind="$1"
  local path="$2"
  local exit_code="${3:-0}"
  "${PYTHON_BIN}" - "$kind" "$path" "$exit_code" "${CURRENT_TF}" "${START}" "${END}" "${TIMEFRAMES}" <<'PY'
import json
import sys
from datetime import datetime, timezone
kind, path, code, tf, start, end, tfs = sys.argv[1:8]
payload = {
    "kind": kind,
    "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "exit_code": int(code),
    "current_timeframe": tf,
    "start": start,
    "end": end,
    "timeframes": [x.strip() for x in tfs.split(",") if x.strip()],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
PY
}

on_exit() {
  local code=$?
  write_status_json "collect_exit" "${LAST_EXIT_FILE}" "${code}"
}
trap on_exit EXIT

echo "[info] full-history multi-timeframe ingestion"
echo "[info] start=${START} end=${END} market=${MARKET} timeframes=${TIMEFRAMES}"
write_status_json "collect_start" "${PROGRESS_FILE}" 0

IFS=',' read -r -a TF_ARR <<< "${TIMEFRAMES}"
for raw_tf in "${TF_ARR[@]}"; do
  tf="$(echo "${raw_tf}" | xargs | tr '[:upper:]' '[:lower:]')"
  [[ -z "${tf}" ]] && continue
  CURRENT_TF="${tf}"
  run_tag="$(printf "%s|%s|%s|%s|%s|%s" "${MARKET}" "${tf}" "${START}" "${END}" "${SYMBOLS}" "${SYMBOL_MAP}" | sha256sum | awk '{print substr($1,1,16)}')"
  ckpt="${CHECKPOINT_DIR}/ingest_${MARKET}_${tf}_${run_tag}.json"
  echo "[ingest] tf=${tf}"
  allow_partial_flag=()
  if [[ "${ALLOW_PARTIAL}" == "1" ]]; then
    allow_partial_flag=(--allow-partial)
  fi
  DATABASE_URL="${DATABASE_URL}" "${PYTHON_BIN}" scripts/ingest_bitget_market_bars.py \
    --database-url "${DATABASE_URL}" \
    --market "${MARKET}" \
    --timeframe "${tf}" \
    --symbols "${SYMBOLS}" \
    --symbol-map "${SYMBOL_MAP}" \
    --fallback-source "${FALLBACK_SOURCE}" \
    --start "${START}" \
    --end "${END}" \
    --chunk-days "${CHUNK_DAYS}" \
    --workers "${MAX_WORKERS}" \
    --min-workers "${MIN_WORKERS}" \
    --adaptive-workers "${ADAPTIVE_WORKERS}" \
    --rate-limit-backoff-sec "${RATE_LIMIT_BACKOFF_SEC}" \
    --max-rate-limit-backoff-sec "${MAX_RATE_LIMIT_BACKOFF_SEC}" \
    --db-page-size "${DB_PAGE_SIZE}" \
    --db-commit-every-symbols "${DB_COMMIT_EVERY_SYMBOLS}" \
    --split-failed-symbols "${SPLIT_FAILED_SYMBOLS}" \
    --min-split-chunk-days "${MIN_SPLIT_CHUNK_DAYS}" \
    --progress-file "${STATUS_DIR}/ingest_${MARKET}_${tf}_progress.json" \
    "${allow_partial_flag[@]}" \
    --resume \
    --checkpoint-file "${ckpt}"

  echo "[audit] tf=${tf}"
  DATABASE_URL="${DATABASE_URL}" "${PYTHON_BIN}" scripts/audit_training_data_completeness.py \
    --database-url "${DATABASE_URL}" \
    --symbols "$(echo "${SYMBOL_MAP}" | sed 's/[^:,]*://g')" \
    --lookback-days "${LOOKBACK_DAYS}" \
    --timeframe "${tf}" \
    > "${AUDIT_DIR}/audit_${tf}.json"
done

CURRENT_TF="align"
echo "[align] build aligned multi-timeframe context"
DATABASE_URL="${DATABASE_URL}" "${PYTHON_BIN}" scripts/build_multi_timeframe_context.py \
  --database-url "${DATABASE_URL}" \
  --symbols "$(echo "${SYMBOL_MAP}" | sed 's/[^:,]*://g')" \
  --start "${START}" \
  --end "${END}" \
  --primary-timeframe "${PRIMARY_TIMEFRAME:-5m}" \
  --context-timeframes "${TIMEFRAMES}"

echo "[ok] collect_all_timeframe_market_bars_done"
write_status_json "collect_done" "${PROGRESS_FILE}" 0
