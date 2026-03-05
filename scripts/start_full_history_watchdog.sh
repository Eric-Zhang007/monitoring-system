#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TASK_NAME="${TASK_NAME:-full_history_mtf}"
WATCHDOG_TASK_NAME="${WATCHDOG_TASK_NAME:-full_history_mtf_watchdog}"
END_TS="${END_TS:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
START_CMD="${START_CMD:-FALLBACK_SOURCE=none START=2018-01-01T00:00:00Z END=${END_TS} TIMEFRAMES=1m,3m,5m,15m,30m,1h,4h,6h,12h,1d MAX_WORKERS=2 CHUNK_DAYS=15 ALLOW_PARTIAL=1 bash scripts/collect_all_timeframe_market_bars.sh}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-$(ls -t artifacts/checkpoints/all_timeframes/ingest_perp_1m_*.json 2>/dev/null | head -n 1)}"
if [[ -z "${CHECKPOINT_FILE}" ]]; then
  run_tag="$(printf "%s|%s|%s|%s|%s|%s" perp 1m 2018-01-01T00:00:00Z "${END_TS}" "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT" "BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK" | sha256sum | awk '{print substr($1,1,16)}')"
  CHECKPOINT_FILE="artifacts/checkpoints/all_timeframes/ingest_perp_1m_${run_tag}.json"
fi

scripts/run_bg_task.sh start "${TASK_NAME}" -- bash -lc "${START_CMD}" || true
scripts/run_bg_task.sh start "${WATCHDOG_TASK_NAME}" -- python3 scripts/watch_ingest_task.py \
  --task-name "${TASK_NAME}" \
  --checkpoint-file "${CHECKPOINT_FILE}" \
  --poll-seconds "${WATCHDOG_POLL_SECONDS:-60}" \
  --stale-seconds "${WATCHDOG_STALE_SECONDS:-1800}" \
  --max-restarts "${WATCHDOG_MAX_RESTARTS:-10}" \
  --start-cmd "${START_CMD}" \
  --status-file "${WATCHDOG_STATUS_FILE:-artifacts/runtime/bg/status/ingest_watchdog_status.json}"

echo "ingest_task=${TASK_NAME} watchdog_task=${WATCHDOG_TASK_NAME}"
