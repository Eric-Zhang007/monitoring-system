#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "[ERR] python not executable: $PYTHON_BIN" >&2; exit 2; }
else
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "[ERR] missing command: $PYTHON_BIN" >&2; exit 2; }
fi

DATABASE_URL="${DATABASE_URL:-postgresql://monitor@localhost:5432/monitor}"
START_TS="${START_TS:-2018-01-01T00:00:00Z}"
END_TS="${END_TS:-}"
DATA_MODE="${DATA_MODE:-production}"
PRODUCTION_LOOKBACK_DAYS="${PRODUCTION_LOOKBACK_DAYS:-420}"
PRIMARY_TF="${PRIMARY_TF:-5m}"
SECONDARY_TF="${SECONDARY_TF:-1h}"
SYMBOLS="${SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
BITGET_SYMBOLS="${BITGET_SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT}"
BITGET_SYMBOL_MAP="${BITGET_SYMBOL_MAP:-BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK}"

RUN_PRE_AUDIT="${RUN_PRE_AUDIT:-1}"
RUN_PRIMARY_REMEDIATION="${RUN_PRIMARY_REMEDIATION:-1}"
RUN_SECONDARY_MARKET="${RUN_SECONDARY_MARKET:-1}"
RUN_DERIVATIVES="${RUN_DERIVATIVES:-1}"
RUN_ORCHESTRATED_EVENTS="${RUN_ORCHESTRATED_EVENTS:-1}"
RUN_SOCIAL_RAW_IMPORT="${RUN_SOCIAL_RAW_IMPORT:-1}"
RUN_POST_AUDIT="${RUN_POST_AUDIT:-1}"
RUN_BUILD_OFFLINE_BUNDLE="${RUN_BUILD_OFFLINE_BUNDLE:-1}"

INGEST_WORKERS="${INGEST_WORKERS:-3}"
INGEST_CHUNK_DAYS="${INGEST_CHUNK_DAYS:-30}"
INGEST_MAX_CHUNKS="${INGEST_MAX_CHUNKS:-0}"
MARKET_FALLBACK_SOURCE="${MARKET_FALLBACK_SOURCE:-none}"
EVENT_CHUNK_DAYS="${EVENT_CHUNK_DAYS:-30}"
EVENT_MAX_CHUNKS="${EVENT_MAX_CHUNKS:-0}"
EVENT_SOCIAL_SOURCES="${EVENT_SOCIAL_SOURCES:-twitter,reddit,youtube,telegram}"
EVENT_DISABLE_GOOGLE="${EVENT_DISABLE_GOOGLE:-0}"
EVENT_DISABLE_GDELT="${EVENT_DISABLE_GDELT:-0}"
EVENT_DISABLE_OFFICIAL_RSS="${EVENT_DISABLE_OFFICIAL_RSS:-0}"
EVENT_DISABLE_SOURCE_BALANCE="${EVENT_DISABLE_SOURCE_BALANCE:-0}"
EVENT_GDELT_MAX_RECORDS="${EVENT_GDELT_MAX_RECORDS:-50}"
EVENT_DAY_STEP="${EVENT_DAY_STEP:-7}"
BUNDLE_OUT_ROOT="${BUNDLE_OUT_ROOT:-$ROOT_DIR/artifacts/offline_bundle}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/artifacts/data_collection}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUT_ROOT/run_${RUN_ID}"
mkdir -p "$RUN_DIR"

effective_start_ts="$START_TS"
if [[ "$DATA_MODE" == "production" ]]; then
  effective_start_ts="$(
    "$PYTHON_BIN" - "$END_TS" "$PRODUCTION_LOOKBACK_DAYS" <<'PY'
from datetime import datetime, timedelta, timezone
import sys

end_raw = (sys.argv[1] or "").strip()
lookback_days = int(sys.argv[2] or "420")
if end_raw:
    text = end_raw.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    end_dt = dt.astimezone(timezone.utc)
else:
    end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=max(1, lookback_days))
print(start_dt.isoformat().replace("+00:00", "Z"))
PY
  )"
fi

END_ARGS=()
if [[ -n "$END_TS" ]]; then
  END_ARGS=(--end "$END_TS")
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

run_json_to_file() {
  local out_file="$1"
  shift
  local tmp_file
  tmp_file="$(mktemp)"
  if "$@" | tee "$tmp_file"; then
    local last
    last="$(grep -E "\{.*\}" "$tmp_file" | tail -n 1 || true)"
    if [[ -n "$last" ]]; then
      printf '%s\n' "$last" > "$out_file"
    else
      cp "$tmp_file" "$out_file"
    fi
    rm -f "$tmp_file"
    return 0
  fi
  local rc=$?
  cp "$tmp_file" "$out_file" || true
  rm -f "$tmp_file"
  return "$rc"
}

log "collect_required_data run_id=${RUN_ID}"
log "artifacts_dir=${RUN_DIR}"
log "data_mode=${DATA_MODE} requested_start=${START_TS} effective_start=${effective_start_ts} end=${END_TS:-now}"

if [[ "$RUN_PRE_AUDIT" == "1" ]]; then
  log "[0/8] pre readiness audit"
  run_json_to_file "$RUN_DIR/readiness_before.json" \
    "$PYTHON_BIN" scripts/audit_required_data_readiness.py \
    --mode "$DATA_MODE" \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --primary-timeframe "$PRIMARY_TF" \
    --secondary-timeframe "$SECONDARY_TF" \
    --start "$effective_start_ts" \
    --lookback-days "$PRODUCTION_LOOKBACK_DAYS" \
    "${END_ARGS[@]}"
else
  log "[0/8] pre readiness audit skipped"
fi

if [[ "$RUN_PRIMARY_REMEDIATION" == "1" ]]; then
  log "[1/8] primary remediation (market+aux+orderbook+social)"
  RUN_ID="${RUN_ID}" \
  DATABASE_URL="$DATABASE_URL" \
  START_TS="$effective_start_ts" \
  END_TS="$END_TS" \
  TIMEFRAME="$PRIMARY_TF" \
  SYMBOLS="$SYMBOLS" \
  BITGET_SYMBOLS="$BITGET_SYMBOLS" \
  BITGET_SYMBOL_MAP="$BITGET_SYMBOL_MAP" \
  INGEST_WORKERS="$INGEST_WORKERS" \
  INGEST_CHUNK_DAYS="$INGEST_CHUNK_DAYS" \
  INGEST_MAX_CHUNKS="$INGEST_MAX_CHUNKS" \
  MARKET_FALLBACK_SOURCE="$MARKET_FALLBACK_SOURCE" \
  RUN_PRE_AUDIT=0 \
  RUN_FINAL_AUDIT=0 \
  SOCIAL_STRICT=0 \
  bash scripts/remediate_liquid_data_gaps.sh | tee "$RUN_DIR/remediate_primary.log"
else
  log "[1/8] primary remediation skipped"
fi

if [[ "$RUN_SECONDARY_MARKET" == "1" ]]; then
  log "[2/8] secondary timeframe market bars (${SECONDARY_TF})"
  CMD=(
    "$PYTHON_BIN" scripts/ingest_bitget_market_bars.py
    --database-url "$DATABASE_URL"
    --market perp
    --timeframe "$SECONDARY_TF"
    --start "$effective_start_ts"
    --symbols "$BITGET_SYMBOLS"
    --symbol-map "$BITGET_SYMBOL_MAP"
    --fallback-source coingecko
    --workers "$INGEST_WORKERS"
    --chunk-days "$INGEST_CHUNK_DAYS"
    --checkpoint-file "$RUN_DIR/market_${SECONDARY_TF}_checkpoint.json"
    --resume
    --allow-partial
  )
  if [[ "$INGEST_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$INGEST_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  run_json_to_file "$RUN_DIR/market_${SECONDARY_TF}.json" "${CMD[@]}"
else
  log "[2/8] secondary timeframe market bars skipped"
fi

if [[ "$RUN_DERIVATIVES" == "1" ]]; then
  log "[3/8] derivatives regime signals"
  run_json_to_file "$RUN_DIR/derivatives_signals.json" \
    "$PYTHON_BIN" scripts/ingest_binance_derivatives_signals.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --start "$effective_start_ts" \
    --period "$PRIMARY_TF" \
    "${END_ARGS[@]}"
else
  log "[3/8] derivatives regime signals skipped"
fi

if [[ "$RUN_ORCHESTRATED_EVENTS" == "1" ]]; then
  log "[4/8] orchestrated event/social backfill"
  CMD=(
    "$PYTHON_BIN" scripts/orchestrate_event_social_backfill.py
    --start "$effective_start_ts"
    --chunk-days "$EVENT_CHUNK_DAYS"
    --checkpoint-file "$RUN_DIR/event_social_checkpoint.json"
    --resume
    --social-sources "$EVENT_SOCIAL_SOURCES"
    --event-gdelt-max-records "$EVENT_GDELT_MAX_RECORDS"
    --event-day-step "$EVENT_DAY_STEP"
    --out-events-jsonl "$RUN_DIR/events_multisource_2018_now.jsonl"
    --out-social-jsonl "$RUN_DIR/social_history_2018_now.jsonl"
  )
  if [[ "$EVENT_DISABLE_GOOGLE" == "1" ]]; then
    CMD+=(--event-disable-google)
  fi
  if [[ "$EVENT_DISABLE_GDELT" == "1" ]]; then
    CMD+=(--event-disable-gdelt)
  fi
  if [[ "$EVENT_DISABLE_OFFICIAL_RSS" == "1" ]]; then
    CMD+=(--event-disable-official-rss)
  fi
  if [[ "$EVENT_DISABLE_SOURCE_BALANCE" == "1" ]]; then
    CMD+=(--event-disable-source-balance)
  fi
  if [[ "$EVENT_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$EVENT_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  run_json_to_file "$RUN_DIR/event_social_orchestrator.json" "${CMD[@]}"

  if [[ -s "$RUN_DIR/events_multisource_2018_now.jsonl" ]]; then
    log "[5/8] import events"
    run_json_to_file "$RUN_DIR/events_import.json" \
      "$PYTHON_BIN" scripts/import_events_jsonl.py \
      --jsonl "$RUN_DIR/events_multisource_2018_now.jsonl" \
      --database-url "$DATABASE_URL" \
      --batch-size 200
  else
    log "[5/8] skip events import (no jsonl)"
  fi

  if [[ -s "$RUN_DIR/social_history_2018_now.jsonl" ]]; then
    log "[6/8] import social events"
    run_json_to_file "$RUN_DIR/social_import.json" \
      "$PYTHON_BIN" scripts/import_social_events_jsonl.py \
      --jsonl "$RUN_DIR/social_history_2018_now.jsonl" \
      --database-url "$DATABASE_URL" \
      --batch-size 200
    if [[ "$RUN_SOCIAL_RAW_IMPORT" == "1" ]]; then
      log "[6.1/8] import social raw tables"
      run_json_to_file "$RUN_DIR/social_raw_import.json" \
        "$PYTHON_BIN" scripts/import_social_raw_jsonl.py \
        --jsonl "$RUN_DIR/social_history_2018_now.jsonl" \
        --database-url "$DATABASE_URL"
    else
      log "[6.1/8] social raw import skipped"
    fi
  else
    log "[6/8] skip social import (no jsonl)"
  fi
else
  log "[4-6/8] orchestrated event/social skipped"
fi

if [[ "$RUN_POST_AUDIT" == "1" ]]; then
  log "[7/8] post audits"
  run_json_to_file "$RUN_DIR/full_history_after.json" \
    "$PYTHON_BIN" scripts/audit_full_history_completeness.py \
    --database-url "$DATABASE_URL" \
    --start 2018-01-01T00:00:00Z \
    --timeframe "$PRIMARY_TF" \
    --symbols "$SYMBOLS" \
    "${END_ARGS[@]}"

  run_json_to_file "$RUN_DIR/training_${PRIMARY_TF}_after.json" \
    "$PYTHON_BIN" scripts/audit_training_data_completeness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$PRIMARY_TF" \
    --lookback-days 420

  run_json_to_file "$RUN_DIR/training_${SECONDARY_TF}_after.json" \
    "$PYTHON_BIN" scripts/audit_training_data_completeness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$SECONDARY_TF" \
    --lookback-days 420

  run_json_to_file "$RUN_DIR/readiness_after.json" \
    "$PYTHON_BIN" scripts/audit_required_data_readiness.py \
    --mode "$DATA_MODE" \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --primary-timeframe "$PRIMARY_TF" \
    --secondary-timeframe "$SECONDARY_TF" \
    --start "$effective_start_ts" \
    --lookback-days "$PRODUCTION_LOOKBACK_DAYS" \
    "${END_ARGS[@]}"
else
  log "[7/8] post audits skipped"
fi

if [[ "$RUN_BUILD_OFFLINE_BUNDLE" == "1" ]]; then
  log "[8/8] build offline bundle from current local db"
  RUN_ID="$RUN_ID" \
  OUT_ROOT="$BUNDLE_OUT_ROOT" \
  DATABASE_URL="$DATABASE_URL" \
  START_TS="$effective_start_ts" \
  END_TS="$END_TS" \
  TIMEFRAME="$PRIMARY_TF" \
  SECONDARY_TF="$SECONDARY_TF" \
  SYMBOLS="$SYMBOLS" \
  RUN_MARKET=0 \
  RUN_MARKET_SECONDARY=0 \
  RUN_AUX=0 \
  RUN_DERIVATIVES=0 \
  RUN_ORDERBOOK_PROXY=0 \
  RUN_EVENTS=0 \
  RUN_SOCIAL=0 \
  RUN_AUDIT=1 \
  RUN_EXPORT=1 \
  RUN_PACKAGE=1 \
  bash scripts/build_offline_data_bundle.sh | tee "$RUN_DIR/offline_bundle.log"
else
  log "[8/8] build offline bundle skipped"
fi

log "done"
log "run_dir=$RUN_DIR"
