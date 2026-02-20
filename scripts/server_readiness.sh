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

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
COLLECTOR_METRICS_URL="${COLLECTOR_METRICS_URL:-http://127.0.0.1:9101/metrics}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
TASK_MAX_WAIT_SEC="${TASK_MAX_WAIT_SEC:-900}"
TASK_STALL_TIMEOUT_SEC="${TASK_STALL_TIMEOUT_SEC:-240}"
TASK_POLL_SEC="${TASK_POLL_SEC:-2}"
HTTP_TIMEOUT_SEC="${HTTP_TIMEOUT_SEC:-10}"
REQUIRE_SCREEN_SESSIONS="${REQUIRE_SCREEN_SESSIONS:-1}"
REQUIRE_COLLECTOR_METRICS="${REQUIRE_COLLECTOR_METRICS:-0}"
RUN_HEALTH_CHECK_SCRIPT="${RUN_HEALTH_CHECK_SCRIPT:-1}"
SCREEN_NAMES="${SCREEN_NAMES:-backend collector task_worker model_ops}"
ENABLE_CONTINUOUS_OPS_LOOP="${ENABLE_CONTINUOUS_OPS_LOOP:-1}"
VERIFY_SCREEN_DURING_TASK="${VERIFY_SCREEN_DURING_TASK:-1}"

if [[ "$ENABLE_CONTINUOUS_OPS_LOOP" == "1" ]]; then
  SCREEN_NAMES="${SCREEN_NAMES} ops_loop"
fi

require_number() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[FAIL] invalid numeric value: ${name}=${value}"
    exit 2
  fi
}

need_cmds=(curl screen awk sed grep)
for c in "${need_cmds[@]}"; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $c"
    exit 2
  fi
done
if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[FAIL] python interpreter not executable: $PYTHON_BIN"
    exit 2
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $PYTHON_BIN"
    exit 2
  fi
fi

require_number "TIMEOUT_SEC" "$TIMEOUT_SEC"
require_number "TASK_MAX_WAIT_SEC" "$TASK_MAX_WAIT_SEC"
require_number "TASK_STALL_TIMEOUT_SEC" "$TASK_STALL_TIMEOUT_SEC"
require_number "TASK_POLL_SEC" "$TASK_POLL_SEC"
require_number "HTTP_TIMEOUT_SEC" "$HTTP_TIMEOUT_SEC"

screen_has_session() {
  local name="$1"
  screen -ls 2>/dev/null | grep -Eq "[[:space:]]+[0-9]+\\.${name}[[:space:]]"
}

verify_required_screens() {
  local screen_ls
  screen_ls="$(screen -ls 2>/dev/null || true)"
  for name in $SCREEN_NAMES; do
    if ! grep -Eq "[[:space:]]+[0-9]+\\.${name}[[:space:]]" <<<"$screen_ls"; then
      echo "[FAIL] missing screen session: ${name}"
      echo "$screen_ls"
      return 1
    fi
  done
}

wait_http() {
  local url="$1"
  local timeout_sec="$2"
  local timeout_i="${timeout_sec%.*}"
  local deadline=$(( $(date +%s) + timeout_i ))
  while true; do
    if curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      return 1
    fi
    sleep 2
  done
}

if [[ "$REQUIRE_SCREEN_SESSIONS" == "1" ]]; then
  verify_required_screens
fi

if ! wait_http "${API_BASE}/health" "$TIMEOUT_SEC"; then
  echo "[FAIL] backend health timeout at ${API_BASE}/health"
  exit 2
fi

curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "${API_BASE}/health" >/dev/null
curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "${API_BASE}/metrics" >/dev/null
curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "${API_BASE}/api/v2/risk/limits" >/dev/null

task_submit="$(curl --max-time "$HTTP_TIMEOUT_SEC" -fsS -X POST "${API_BASE}/api/v2/tasks/pnl-attribution" \
  -H "content-type: application/json" \
  -d '{"track":"liquid","lookback_hours":24}')"
task_id="$(TASK_JSON="$task_submit" "$PYTHON_BIN" - <<'PY'
import json
import os

raw = os.environ.get("TASK_JSON") or "{}"
obj = json.loads(raw)
print(obj.get("task_id", ""))
PY
)"
if [[ -z "$task_id" ]]; then
  echo "[FAIL] unable to parse task_id from async task submit response"
  echo "response=${task_submit}"
  exit 2
fi

task_deadline=$(( $(date +%s) + ${TASK_MAX_WAIT_SEC%.*} ))
task_stall_deadline=$(( $(date +%s) + ${TASK_STALL_TIMEOUT_SEC%.*} ))
task_status="queued"
task_payload='{}'

while true; do
  task_payload="$(curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "${API_BASE}/api/v2/tasks/${task_id}")"
  read -r parsed_status parsed_error <<<"$(TASK_JSON="$task_payload" "$PYTHON_BIN" - <<'PY'
import json
import os

raw = os.environ.get("TASK_JSON") or "{}"
obj = json.loads(raw)
status = str(obj.get("status") or "unknown")
error = str(obj.get("error") or "")
print(status, error.replace("\n", " "))
PY
)"

  if [[ "$parsed_status" != "$task_status" ]]; then
    task_status="$parsed_status"
    task_stall_deadline=$(( $(date +%s) + ${TASK_STALL_TIMEOUT_SEC%.*} ))
  fi

  if [[ "$task_status" == "completed" ]]; then
    break
  fi
  if [[ "$task_status" == "failed" ]]; then
    echo "[FAIL] pnl attribution async task failed"
    echo "task_id=${task_id}"
    echo "task_error=${parsed_error}"
    echo "task_payload=${task_payload}"
    exit 2
  fi

  if (( $(date +%s) >= task_deadline )); then
    echo "[FAIL] async task max-wait timeout task_id=${task_id} status=${task_status}"
    echo "task_payload=${task_payload}"
    exit 2
  fi
  if (( $(date +%s) >= task_stall_deadline )); then
    echo "[FAIL] async task stalled task_id=${task_id} status=${task_status} stall_timeout_sec=${TASK_STALL_TIMEOUT_SEC}"
    echo "task_payload=${task_payload}"
    exit 2
  fi

  if [[ "$VERIFY_SCREEN_DURING_TASK" == "1" && "$REQUIRE_SCREEN_SESSIONS" == "1" ]]; then
    verify_required_screens
  fi

  sleep "$TASK_POLL_SEC"
done

if [[ "$REQUIRE_COLLECTOR_METRICS" == "1" ]]; then
  if ! curl --max-time "$HTTP_TIMEOUT_SEC" -fsS "${COLLECTOR_METRICS_URL}" >/dev/null 2>&1; then
    echo "[FAIL] collector metrics endpoint unreachable: ${COLLECTOR_METRICS_URL}"
    exit 2
  fi
fi

if [[ "$RUN_HEALTH_CHECK_SCRIPT" == "1" ]]; then
  CHECK_RUNTIME_SERVICES=0 API_BASE="$API_BASE" DATABASE_URL="${DATABASE_URL:-}" REDIS_URL="${REDIS_URL:-}" \
    "$PYTHON_BIN" monitoring/health_check.py >/tmp/health_check_runtime.log 2>&1 || {
      echo "[FAIL] monitoring/health_check.py returned non-zero"
      tail -n 80 /tmp/health_check_runtime.log || true
      exit 2
    }
fi

echo "[OK] runtime readiness passed"
echo "api_base=${API_BASE}"
echo "task_id=${task_id}"
echo "task_status=${task_status}"
echo "task_max_wait_sec=${TASK_MAX_WAIT_SEC}"
echo "task_stall_timeout_sec=${TASK_STALL_TIMEOUT_SEC}"
