#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="/opt/monitoring-system/current"
ENV_FILE=""
TIMEOUT_SEC="120"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --timeout-sec)
      TIMEOUT_SEC="$2"
      shift 2
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

compose=(docker compose -f "$DEPLOY_DIR/docker-compose.yml")
if [[ -n "$ENV_FILE" ]]; then
  compose=(docker compose --env-file "$ENV_FILE" -f "$DEPLOY_DIR/docker-compose.yml")
elif [[ -f "$DEPLOY_DIR/.env" ]]; then
  compose=(docker compose --env-file "$DEPLOY_DIR/.env" -f "$DEPLOY_DIR/docker-compose.yml")
fi

"${compose[@]}" ps

deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while true; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    break
  fi
  if (( $(date +%s) >= deadline )); then
    echo "[FAIL] backend health timeout"
    exit 2
  fi
  sleep 3
done

curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8000/api/v2/risk/limits >/dev/null
curl -fsS http://127.0.0.1:9090/-/ready >/dev/null
curl -fsS http://127.0.0.1:9093/-/ready >/dev/null

echo "[OK] runtime verification passed"
