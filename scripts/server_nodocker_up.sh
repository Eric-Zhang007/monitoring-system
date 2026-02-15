#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_URL="${DATABASE_URL:-postgresql://monitor:change_me_please@localhost:5432/monitor}"
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379}"
LIQUID_SYMBOLS="${LIQUID_SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
TRAIN_INTERVAL_SEC="${TRAIN_INTERVAL_SEC:-3600}"

cd "$ROOT_DIR"

service postgresql start >/dev/null 2>&1 || true
service redis-server start >/dev/null 2>&1 || true

( cd backend && DATABASE_URL="$DB_URL" alembic upgrade head >/tmp/alembic_upgrade_boot.log 2>&1 ) || true

screen -S backend -X quit >/dev/null 2>&1 || true
screen -S collector -X quit >/dev/null 2>&1 || true
screen -S trainer -X quit >/dev/null 2>&1 || true

screen -dmS backend bash -lc \
  "cd '$ROOT_DIR/backend' && DATABASE_URL='$DB_URL' REDIS_URL='$REDIS_URL' python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 |& tee /tmp/backend_screen.log"

screen -dmS collector bash -lc \
  "cd '$ROOT_DIR/collector' && API_BASE='http://127.0.0.1:8000' REDIS_URL='$REDIS_URL' LIQUID_SYMBOLS='$LIQUID_SYMBOLS' COLLECT_INTERVAL=60 python3 collector.py |& tee /tmp/collector_screen.log"

screen -dmS trainer bash -lc \
  "cd '$ROOT_DIR' && DATABASE_URL='$DB_URL' LIQUID_SYMBOLS='$LIQUID_SYMBOLS' TRAIN_INTERVAL_SEC='$TRAIN_INTERVAL_SEC' TRAIN_RUN_ONCE=0 TRAIN_ENABLE_VC=1 TRAIN_ENABLE_LIQUID=1 torchrun --standalone --nproc_per_node=2 training/main.py |& tee /tmp/trainer_screen.log"

sleep 3
echo "[screen sessions]"
screen -ls || true
echo "[health]"
curl -sS http://127.0.0.1:8000/health || true
echo
