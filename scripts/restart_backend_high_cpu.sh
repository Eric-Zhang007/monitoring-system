#!/usr/bin/env bash
set -euo pipefail

WORKERS="${1:-$(nproc)}"
if [[ "${WORKERS}" -lt 2 ]]; then
  WORKERS=2
fi

echo "restart backend with workers=${WORKERS}"
BACKEND_UVICORN_WORKERS="${WORKERS}" docker compose up -d --build backend
sleep 2
docker compose ps backend
docker compose top backend || true
