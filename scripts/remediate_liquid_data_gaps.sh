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
TIMEFRAME="${TIMEFRAME:-5m}"
SYMBOLS="${SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
BITGET_SYMBOLS="${BITGET_SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT}"
BITGET_SYMBOL_MAP="${BITGET_SYMBOL_MAP:-BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK}"

RUN_MARKET="${RUN_MARKET:-1}"
RUN_AUX="${RUN_AUX:-1}"
RUN_ORDERBOOK_PROXY="${RUN_ORDERBOOK_PROXY:-1}"
RUN_SOCIAL="${RUN_SOCIAL:-1}"
RUN_PRE_AUDIT="${RUN_PRE_AUDIT:-1}"
RUN_FINAL_AUDIT="${RUN_FINAL_AUDIT:-1}"
SOCIAL_STRICT="${SOCIAL_STRICT:-0}"
FIXED_AUDIT_START="2018-01-01T00:00:00Z"

INGEST_CHUNK_DAYS="${INGEST_CHUNK_DAYS:-30}"
INGEST_WORKERS="${INGEST_WORKERS:-3}"
INGEST_MAX_CHUNKS="${INGEST_MAX_CHUNKS:-0}"
MARKET_FALLBACK_SOURCE="${MARKET_FALLBACK_SOURCE:-none}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/artifacts/data_remediation}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${OUT_ROOT}/run_${RUN_ID}"
mkdir -p "$RUN_DIR"

END_ARGS=()
if [[ -n "$END_TS" ]]; then
  END_ARGS=(--end "$END_TS")
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

if [[ "$START_TS" != "$FIXED_AUDIT_START" ]]; then
  if [[ "$RUN_PRE_AUDIT" == "1" || "$RUN_FINAL_AUDIT" == "1" ]]; then
    log "audit_full_history_completeness 固定要求 --start=${FIXED_AUDIT_START}; 当前 START_TS=${START_TS}，自动跳过 pre/final audit"
  fi
  RUN_PRE_AUDIT=0
  RUN_FINAL_AUDIT=0
fi

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
  else
    local rc=$?
    cp "$tmp_file" "$out_file" || true
    rm -f "$tmp_file"
    return "$rc"
  fi
}

if [[ "$RUN_PRE_AUDIT" == "1" ]]; then
  log "[0/7] pre-audit (full history completeness)"
  run_json_to_file "$RUN_DIR/audit_before.json" \
    "$PYTHON_BIN" scripts/audit_full_history_completeness.py \
    --database-url "$DATABASE_URL" \
    --start "$FIXED_AUDIT_START" \
    --timeframe "$TIMEFRAME" \
    --symbols "$SYMBOLS" \
    "${END_ARGS[@]}"
else
  log "[0/7] pre-audit skipped (RUN_PRE_AUDIT=${RUN_PRE_AUDIT})"
fi

if [[ "$RUN_MARKET" == "1" ]]; then
  log "[1/7] market bars backfill via bitget"
  CMD=(
    "$PYTHON_BIN" scripts/ingest_bitget_market_bars.py
    --database-url "$DATABASE_URL"
    --market perp
    --timeframe "$TIMEFRAME"
    --start "$START_TS"
    --symbols "$BITGET_SYMBOLS"
    --symbol-map "$BITGET_SYMBOL_MAP"
    --fallback-source "$MARKET_FALLBACK_SOURCE"
    --workers "$INGEST_WORKERS"
    --chunk-days "$INGEST_CHUNK_DAYS"
    --checkpoint-file "$RUN_DIR/market_backfill_checkpoint.json"
    --resume
    --allow-partial
  )
  if [[ "$INGEST_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$INGEST_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  run_json_to_file "$RUN_DIR/market_backfill.json" "${CMD[@]}"
else
  log "[1/7] skip market bars backfill (RUN_MARKET=${RUN_MARKET})"
fi

if [[ "$RUN_AUX" == "1" ]]; then
  log "[2/7] funding + oi proxy ingest"
  run_json_to_file "$RUN_DIR/aux_signals.json" \
    "$PYTHON_BIN" scripts/ingest_binance_aux_signals.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --start "$START_TS" \
    --timeframe "$TIMEFRAME" \
    "${END_ARGS[@]}"
else
  log "[2/7] skip aux ingest (RUN_AUX=${RUN_AUX})"
fi

if [[ "$RUN_ORDERBOOK_PROXY" == "1" ]]; then
  log "[3/7] orderbook cold-start backfill from market_bars"
  run_json_to_file "$RUN_DIR/orderbook_proxy.json" \
    "$PYTHON_BIN" scripts/backfill_orderbook_from_market_bars.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$TIMEFRAME" \
    --start "$START_TS" \
    "${END_ARGS[@]}"
else
  log "[3/7] skip orderbook proxy backfill (RUN_ORDERBOOK_PROXY=${RUN_ORDERBOOK_PROXY})"
fi

if [[ "$RUN_SOCIAL" == "1" ]]; then
  log "[4/7] social history build"
  SOCIAL_JSONL="$RUN_DIR/social_history.jsonl"
  set +e
  "$PYTHON_BIN" scripts/backfill_social_history.py \
    --start "$START_TS" \
    --timeframe "$TIMEFRAME" \
    --symbols "$SYMBOLS" \
    --pipeline posts_comments \
    --out-jsonl "$SOCIAL_JSONL" \
    --summary-json "$RUN_DIR/social_backfill_summary.json" \
    "${END_ARGS[@]}"
  SOCIAL_RC=$?
  set -e
  if [[ "$SOCIAL_RC" -ne 0 ]]; then
    if [[ "$SOCIAL_STRICT" == "1" ]]; then
      echo "[ERR] social backfill failed rc=${SOCIAL_RC}" >&2
      exit "$SOCIAL_RC"
    fi
    log "social backfill returned rc=${SOCIAL_RC}; continue because SOCIAL_STRICT=${SOCIAL_STRICT}"
  fi

  if [[ -s "$SOCIAL_JSONL" ]]; then
    log "[5/7] import social events into canonical events"
    run_json_to_file "$RUN_DIR/social_import.json" \
      "$PYTHON_BIN" scripts/import_social_events_jsonl.py \
      --jsonl "$SOCIAL_JSONL" \
      --database-url "$DATABASE_URL" \
      --batch-size 200
  else
    log "[5/7] skip social import (no jsonl output)"
  fi
else
  log "[4/7] skip social backfill/import (RUN_SOCIAL=${RUN_SOCIAL})"
fi

if [[ "$RUN_FINAL_AUDIT" == "1" ]]; then
  log "[6/7] post-audit (full history completeness)"
  run_json_to_file "$RUN_DIR/audit_after.json" \
    "$PYTHON_BIN" scripts/audit_full_history_completeness.py \
    --database-url "$DATABASE_URL" \
    --start "$FIXED_AUDIT_START" \
    --timeframe "$TIMEFRAME" \
    --symbols "$SYMBOLS" \
    "${END_ARGS[@]}"

  log "[7/7] training-view audit"
  run_json_to_file "$RUN_DIR/training_audit_after.json" \
    "$PYTHON_BIN" scripts/audit_training_data_completeness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$TIMEFRAME" \
    --lookback-days 180
else
  log "[6/7] final audits skipped (RUN_FINAL_AUDIT=${RUN_FINAL_AUDIT})"
fi

log "done"
log "artifacts: $RUN_DIR"
