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

DB_URL="${DATABASE_URL:-}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
LIQUID_SYMBOLS="${LIQUID_SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
API_BASE="${API_BASE:-http://127.0.0.1:8000}"
TASK_WORKER_POLL_TIMEOUT_SEC="${TASK_WORKER_POLL_TIMEOUT_SEC:-5}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/backend/models}"
FEATURE_VERSION="${FEATURE_VERSION:-feature-store-v2.1}"
FEATURE_PAYLOAD_SCHEMA_VERSION="${FEATURE_PAYLOAD_SCHEMA_VERSION:-v2.2}"
DATA_VERSION="${DATA_VERSION:-v1}"
COST_FEE_BPS="${COST_FEE_BPS:-5.0}"
COST_SLIPPAGE_BPS="${COST_SLIPPAGE_BPS:-3.0}"
COST_IMPACT_COEFF="${COST_IMPACT_COEFF:-120.0}"
CORS_ALLOW_ORIGINS="${CORS_ALLOW_ORIGINS:-http://localhost:3001,http://127.0.0.1:3001}"
CORS_ALLOW_CREDENTIALS="${CORS_ALLOW_CREDENTIALS:-1}"
COLLECT_INTERVAL="${COLLECT_INTERVAL:-60}"
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-180}"
SESSION_STOP_TIMEOUT_SEC="${SESSION_STOP_TIMEOUT_SEC:-20}"
SESSION_BOOT_TIMEOUT_SEC="${SESSION_BOOT_TIMEOUT_SEC:-20}"

screen_sessions=(backend collector task_worker model_ops trainer)

require_number() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[FAIL] invalid numeric value: ${name}=${value}"
    exit 2
  fi
}

screen_has_session() {
  local name="$1"
  screen -ls 2>/dev/null | grep -Eq "[[:space:]]+[0-9]+\\.${name}[[:space:]]"
}

stop_screen_session() {
  local name="$1"
  screen -S "$name" -X quit >/dev/null 2>&1 || true
  local timeout_i="${SESSION_STOP_TIMEOUT_SEC%.*}"
  local deadline=$(( $(date +%s) + timeout_i ))
  while screen_has_session "$name"; do
    if (( $(date +%s) >= deadline )); then
      echo "[FAIL] timeout stopping screen session: ${name}"
      return 1
    fi
    sleep 1
  done
}

wait_for_http() {
  local url="$1"
  local timeout_sec="$2"
  local timeout_i="${timeout_sec%.*}"
  local deadline=$(( $(date +%s) + timeout_i ))
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 2
  done
}

wait_for_session_boot() {
  local name="$1"
  local timeout_i="${SESSION_BOOT_TIMEOUT_SEC%.*}"
  local deadline=$(( $(date +%s) + timeout_i ))
  while true; do
    if screen_has_session "$name"; then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 1
  done
}

if [[ -z "$DB_URL" ]]; then
  echo "[FAIL] DATABASE_URL is required. Copy .env.example to .env and set DATABASE_URL."
  exit 2
fi
if [[ "$DB_URL" == *"change_me_please"* || "$DB_URL" == *"REPLACE_WITH_"* ]]; then
  echo "[FAIL] DATABASE_URL still uses placeholder secret; update your .env before boot."
  exit 2
fi
require_number "COLLECT_INTERVAL" "$COLLECT_INTERVAL"
require_number "HEALTH_TIMEOUT_SEC" "$HEALTH_TIMEOUT_SEC"
require_number "SESSION_STOP_TIMEOUT_SEC" "$SESSION_STOP_TIMEOUT_SEC"
require_number "SESSION_BOOT_TIMEOUT_SEC" "$SESSION_BOOT_TIMEOUT_SEC"

mkdir -p "$MODEL_DIR"

need_cmds=(screen python3 curl)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done

if command -v service >/dev/null 2>&1; then
  service postgresql start >/dev/null 2>&1 || true
  service redis-server start >/dev/null 2>&1 || true
fi

if ! ( cd backend && DATABASE_URL="$DB_URL" python3 -m alembic upgrade head >/tmp/alembic_upgrade_boot.log 2>&1 ); then
  echo "[FAIL] alembic upgrade head failed"
  tail -n 80 /tmp/alembic_upgrade_boot.log || true
  exit 2
fi

for name in "${screen_sessions[@]}"; do
  stop_screen_session "$name"
done

screen -dmS backend env \
  WORK_DIR="$ROOT_DIR/backend" \
  DATABASE_URL="$DB_URL" \
  REDIS_URL="$REDIS_URL" \
  MODEL_DIR="$MODEL_DIR" \
  FEATURE_VERSION="$FEATURE_VERSION" \
  FEATURE_PAYLOAD_SCHEMA_VERSION="$FEATURE_PAYLOAD_SCHEMA_VERSION" \
  DATA_VERSION="$DATA_VERSION" \
  COST_FEE_BPS="$COST_FEE_BPS" \
  COST_SLIPPAGE_BPS="$COST_SLIPPAGE_BPS" \
  COST_IMPACT_COEFF="$COST_IMPACT_COEFF" \
  CORS_ALLOW_ORIGINS="$CORS_ALLOW_ORIGINS" \
  CORS_ALLOW_CREDENTIALS="$CORS_ALLOW_CREDENTIALS" \
  bash -lc 'cd "$WORK_DIR" && exec python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 |& tee /tmp/backend_screen.log'

screen -dmS task_worker env \
  WORK_DIR="$ROOT_DIR" \
  DATABASE_URL="$DB_URL" \
  REDIS_URL="$REDIS_URL" \
  MODEL_DIR="$MODEL_DIR" \
  FEATURE_VERSION="$FEATURE_VERSION" \
  FEATURE_PAYLOAD_SCHEMA_VERSION="$FEATURE_PAYLOAD_SCHEMA_VERSION" \
  DATA_VERSION="$DATA_VERSION" \
  COST_FEE_BPS="$COST_FEE_BPS" \
  COST_SLIPPAGE_BPS="$COST_SLIPPAGE_BPS" \
  COST_IMPACT_COEFF="$COST_IMPACT_COEFF" \
  TASK_WORKER_POLL_TIMEOUT_SEC="$TASK_WORKER_POLL_TIMEOUT_SEC" \
  bash -lc 'cd "$WORK_DIR" && exec python3 monitoring/task_worker.py |& tee /tmp/task_worker_screen.log'

screen -dmS model_ops env \
  WORK_DIR="$ROOT_DIR" \
  API_BASE="$API_BASE" \
  COST_FEE_BPS="$COST_FEE_BPS" \
  COST_SLIPPAGE_BPS="$COST_SLIPPAGE_BPS" \
  COST_IMPACT_COEFF="$COST_IMPACT_COEFF" \
  bash -lc 'cd "$WORK_DIR" && exec python3 monitoring/model_ops_scheduler.py |& tee /tmp/model_ops_screen.log'

screen -dmS collector env \
  WORK_DIR="$ROOT_DIR/collector" \
  API_BASE="$API_BASE" \
  REDIS_URL="$REDIS_URL" \
  LIQUID_SYMBOLS="$LIQUID_SYMBOLS" \
  COLLECT_INTERVAL="$COLLECT_INTERVAL" \
  bash -lc 'cd "$WORK_DIR" && exec python3 collector.py |& tee /tmp/collector_screen.log'

for name in backend collector task_worker model_ops; do
  if ! wait_for_session_boot "$name"; then
    echo "[FAIL] screen session did not start: ${name}"
    screen -ls || true
    exit 2
  fi
done

if ! wait_for_http "${API_BASE}/health" "$HEALTH_TIMEOUT_SEC"; then
  echo "[FAIL] backend health timeout at ${API_BASE}/health"
  echo "[INFO] recent backend log tail"
  tail -n 80 /tmp/backend_screen.log || true
  exit 2
fi

echo "[screen sessions]"
screen -ls || true
echo "[health]"
curl -sS "${API_BASE}/health" || true
echo
