#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

HOST="${HOST:-connect.bjb1.seetacloud.com}"
PORT="${PORT:-40111}"
USER_NAME="${USER_NAME:-root}"
REMOTE_DIR="${REMOTE_DIR:-/root/autodl-tmp/workspace/monitoring-system}"
DB_URL="${DB_URL:-postgresql://monitor@localhost:5432/monitor}"
PRIMARY_TF="${PRIMARY_TF:-5m}"
SECONDARY_TF="${SECONDARY_TF:-1h}"

MARKET_CSV="${MARKET_CSV:-${ROOT_DIR}/artifacts/server_bundle/market_bars_top10_${PRIMARY_TF}_2025_now.csv}"
MARKET_CSV_2="${MARKET_CSV_2:-${ROOT_DIR}/artifacts/server_bundle/market_bars_top10_${SECONDARY_TF}_2025_now.csv}"
EVENTS_JSONL="${EVENTS_JSONL:-${ROOT_DIR}/artifacts/server_bundle/events_multisource_2025_now.jsonl}"
SOCIAL_JSONL="${SOCIAL_JSONL:-${ROOT_DIR}/artifacts/server_bundle/social_history_2025_now.jsonl}"
ORDERBOOK_CSV="${ORDERBOOK_CSV:-${ROOT_DIR}/artifacts/server_bundle/orderbook_l2_2025_now.csv}"
FUNDING_CSV="${FUNDING_CSV:-${ROOT_DIR}/artifacts/server_bundle/funding_rates_2025_now.csv}"
ONCHAIN_CSV="${ONCHAIN_CSV:-${ROOT_DIR}/artifacts/server_bundle/onchain_signals_2025_now.csv}"

if [[ ! -f "${MARKET_CSV}" ]]; then
  echo "[ERR] missing market csv: ${MARKET_CSV}" >&2
  exit 2
fi
if [[ ! -f "${EVENTS_JSONL}" ]]; then
  echo "[ERR] missing events jsonl: ${EVENTS_JSONL}" >&2
  exit 2
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=no -p "${PORT}" "${USER_NAME}@${HOST}")
SCP_BASE=(scp -o StrictHostKeyChecking=no -P "${PORT}")

if [[ -n "${SSHPASS:-}" ]]; then
  SSH_BASE=(sshpass -p "${SSHPASS}" "${SSH_BASE[@]}")
  SCP_BASE=(sshpass -p "${SSHPASS}" "${SCP_BASE[@]}")
fi

echo "[1/6] upload data files"
"${SCP_BASE[@]}" "${MARKET_CSV}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/market_bars_top10_${PRIMARY_TF}_2025_now.csv"
if [[ -f "${MARKET_CSV_2}" ]]; then
  "${SCP_BASE[@]}" "${MARKET_CSV_2}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/market_bars_top10_${SECONDARY_TF}_2025_now.csv"
fi
"${SCP_BASE[@]}" "${EVENTS_JSONL}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/events_multisource_2025_now.jsonl"
if [[ -f "${SOCIAL_JSONL}" ]]; then
  "${SCP_BASE[@]}" "${SOCIAL_JSONL}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/social_history_2025_now.jsonl"
fi
if [[ -f "${ORDERBOOK_CSV}" ]]; then
  "${SCP_BASE[@]}" "${ORDERBOOK_CSV}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/orderbook_l2_2025_now.csv"
fi
if [[ -f "${FUNDING_CSV}" ]]; then
  "${SCP_BASE[@]}" "${FUNDING_CSV}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/funding_rates_2025_now.csv"
fi
if [[ -f "${ONCHAIN_CSV}" ]]; then
  "${SCP_BASE[@]}" "${ONCHAIN_CSV}" "${USER_NAME}@${HOST}:${REMOTE_DIR}/artifacts/server_bundle/onchain_signals_2025_now.csv"
fi

echo "[2/6] import market bars"
"${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && python3 scripts/import_market_bars_csv.py --csv artifacts/server_bundle/market_bars_top10_${PRIMARY_TF}_2025_now.csv --database-url '${DB_URL}'"
if [[ -f "${MARKET_CSV_2}" ]]; then
  "${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && python3 scripts/import_market_bars_csv.py --csv artifacts/server_bundle/market_bars_top10_${SECONDARY_TF}_2025_now.csv --database-url '${DB_URL}'"
fi

echo "[3/6] import events"
"${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && python3 scripts/import_events_jsonl.py --jsonl artifacts/server_bundle/events_multisource_2025_now.jsonl --database-url '${DB_URL}' --batch-size 200"
if [[ -f "${SOCIAL_JSONL}" ]]; then
  "${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && python3 scripts/import_social_events_jsonl.py --jsonl artifacts/server_bundle/social_history_2025_now.jsonl --database-url '${DB_URL}' --batch-size 200"
fi

if [[ -f "${ORDERBOOK_CSV}" || -f "${FUNDING_CSV}" || -f "${ONCHAIN_CSV}" ]]; then
  echo "[3.5/6] import aux/orderbook csv"
  AUX_CMD="cd '${REMOTE_DIR}' && python3 scripts/import_liquid_data_csv.py --database-url '${DB_URL}' --replace-window"
  if [[ -f "${ORDERBOOK_CSV}" ]]; then
    AUX_CMD="${AUX_CMD} --orderbook-csv artifacts/server_bundle/orderbook_l2_2025_now.csv"
  fi
  if [[ -f "${FUNDING_CSV}" ]]; then
    AUX_CMD="${AUX_CMD} --funding-csv artifacts/server_bundle/funding_rates_2025_now.csv"
  fi
  if [[ -f "${ONCHAIN_CSV}" ]]; then
    AUX_CMD="${AUX_CMD} --onchain-csv artifacts/server_bundle/onchain_signals_2025_now.csv"
  fi
  "${SSH_BASE[@]}" "${AUX_CMD}"
fi

echo "[4/6] run data completeness audit (420d)"
"${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && python3 scripts/audit_training_data_completeness.py --database-url '${DB_URL}' --lookback-days 420 --timeframe '${PRIMARY_TF}' > artifacts/server_bundle/audit_training_data_420d_${PRIMARY_TF}.json && cat artifacts/server_bundle/audit_training_data_420d_${PRIMARY_TF}.json"

echo "[5/6] quick event time coverage"
"${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && psql '${DB_URL}' -At -F '|' -c \"SELECT COALESCE(MIN(COALESCE(available_at,occurred_at))::text,''), COALESCE(MAX(COALESCE(available_at,occurred_at))::text,''), COUNT(*)::text FROM events WHERE COALESCE(available_at,occurred_at) >= '2025-01-01T00:00:00Z'::timestamptz;\""

echo "[6/6] event source mix snapshot (2025-now)"
"${SSH_BASE[@]}" "cd '${REMOTE_DIR}' && psql '${DB_URL}' -At -F '|' -c \"SELECT COALESCE(payload->>'provider','unknown') AS provider, COUNT(*)::bigint FROM events WHERE COALESCE(available_at,occurred_at) >= '2025-01-01T00:00:00Z'::timestamptz GROUP BY 1 ORDER BY 2 DESC LIMIT 20;\""

echo "done"
