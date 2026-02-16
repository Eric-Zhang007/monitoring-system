#!/usr/bin/env bash
set -euo pipefail

SESSION_STOP_TIMEOUT_SEC="${SESSION_STOP_TIMEOUT_SEC:-20}"
SCREEN_NAMES="${SCREEN_NAMES:-backend collector task_worker model_ops trainer}"

if ! command -v screen >/dev/null 2>&1; then
  echo "[FAIL] missing command: screen"
  exit 2
fi
if ! [[ "$SESSION_STOP_TIMEOUT_SEC" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "[FAIL] invalid numeric value: SESSION_STOP_TIMEOUT_SEC=${SESSION_STOP_TIMEOUT_SEC}"
  exit 2
fi

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

for name in $SCREEN_NAMES; do
  stop_screen_session "$name"
done

echo "[screen sessions]"
screen -ls || true
