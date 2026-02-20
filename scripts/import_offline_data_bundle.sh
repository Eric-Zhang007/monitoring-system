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

BUNDLE_DIR="${BUNDLE_DIR:-}"
DATABASE_URL="${DATABASE_URL:-postgresql://monitor@localhost:5432/monitor}"
PRIMARY_TF="${PRIMARY_TF:-5m}"
SYMBOLS="${SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-420}"
REPLACE_WINDOW="${REPLACE_WINDOW:-1}"
SKIP_EVENTS="${SKIP_EVENTS:-0}"
SKIP_SOCIAL="${SKIP_SOCIAL:-0}"
STRICT_ASOF_AFTER_IMPORT="${STRICT_ASOF_AFTER_IMPORT:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"; shift 2 ;;
    --database-url)
      DATABASE_URL="$2"; shift 2 ;;
    --primary-timeframe)
      PRIMARY_TF="$2"; shift 2 ;;
    --symbols)
      SYMBOLS="$2"; shift 2 ;;
    --lookback-days)
      LOOKBACK_DAYS="$2"; shift 2 ;;
    --replace-window)
      REPLACE_WINDOW="1"; shift 1 ;;
    --no-replace-window)
      REPLACE_WINDOW="0"; shift 1 ;;
    --skip-events)
      SKIP_EVENTS="1"; shift 1 ;;
    --skip-social)
      SKIP_SOCIAL="1"; shift 1 ;;
    *)
      echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$BUNDLE_DIR" ]]; then
  echo "usage: $0 --bundle-dir <dir> [--database-url ...]" >&2
  exit 2
fi
if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "[ERR] bundle dir not found: $BUNDLE_DIR" >&2
  exit 2
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

MANIFEST="$BUNDLE_DIR/bundle_manifest.json"
if [[ -f "$MANIFEST" ]]; then
  log "manifest found: $MANIFEST"
else
  log "manifest not found, continue with filename conventions"
fi

MARKET_CSV=""
for f in "$BUNDLE_DIR/bundle_market_bars_${PRIMARY_TF}.csv" "$BUNDLE_DIR/raw_market_bars_${PRIMARY_TF}.csv"; do
  if [[ -f "$f" ]]; then
    MARKET_CSV="$f"
    break
  fi
done
ORDERBOOK_CSV=""
[[ -f "$BUNDLE_DIR/bundle_orderbook_l2.csv" ]] && ORDERBOOK_CSV="$BUNDLE_DIR/bundle_orderbook_l2.csv"
FUNDING_CSV=""
[[ -f "$BUNDLE_DIR/bundle_funding_rates.csv" ]] && FUNDING_CSV="$BUNDLE_DIR/bundle_funding_rates.csv"
ONCHAIN_CSV=""
[[ -f "$BUNDLE_DIR/bundle_onchain_signals.csv" ]] && ONCHAIN_CSV="$BUNDLE_DIR/bundle_onchain_signals.csv"
EVENTS_JSONL=""
[[ -f "$BUNDLE_DIR/events_multisource_2018_now.jsonl" ]] && EVENTS_JSONL="$BUNDLE_DIR/events_multisource_2018_now.jsonl"
SOCIAL_JSONL=""
[[ -f "$BUNDLE_DIR/social_history_2018_now.jsonl" ]] && SOCIAL_JSONL="$BUNDLE_DIR/social_history_2018_now.jsonl"

if [[ -z "$MARKET_CSV" ]]; then
  echo "[ERR] market csv not found under bundle dir" >&2
  exit 2
fi

log "[1/5] import market bars"
"$PYTHON_BIN" scripts/import_market_bars_csv.py \
  --csv "$MARKET_CSV" \
  --database-url "$DATABASE_URL"

log "[2/5] import aux/orderbook csv"
AUX_CMD=(
  "$PYTHON_BIN" scripts/import_liquid_data_csv.py
  --database-url "$DATABASE_URL"
)
if [[ -n "$ORDERBOOK_CSV" ]]; then AUX_CMD+=(--orderbook-csv "$ORDERBOOK_CSV"); fi
if [[ -n "$FUNDING_CSV" ]]; then AUX_CMD+=(--funding-csv "$FUNDING_CSV"); fi
if [[ -n "$ONCHAIN_CSV" ]]; then AUX_CMD+=(--onchain-csv "$ONCHAIN_CSV"); fi
if [[ "$REPLACE_WINDOW" == "1" ]]; then AUX_CMD+=(--replace-window); else AUX_CMD+=(--no-replace-window); fi
"${AUX_CMD[@]}"

log "[3/5] import events/social jsonl"
if [[ "$SKIP_EVENTS" != "1" && -n "$EVENTS_JSONL" ]]; then
  "$PYTHON_BIN" scripts/import_events_jsonl.py \
    --jsonl "$EVENTS_JSONL" \
    --database-url "$DATABASE_URL" \
    --batch-size 200
else
  log "skip events import (SKIP_EVENTS=${SKIP_EVENTS}, file_present=$([[ -n "$EVENTS_JSONL" ]] && echo 1 || echo 0))"
fi

if [[ "$SKIP_SOCIAL" != "1" && -n "$SOCIAL_JSONL" ]]; then
  "$PYTHON_BIN" scripts/import_social_events_jsonl.py \
    --jsonl "$SOCIAL_JSONL" \
    --database-url "$DATABASE_URL" \
    --batch-size 200
else
  log "skip social import (SKIP_SOCIAL=${SKIP_SOCIAL}, file_present=$([[ -n "$SOCIAL_JSONL" ]] && echo 1 || echo 0))"
fi

log "[4/5] data completeness audit"
"$PYTHON_BIN" scripts/audit_training_data_completeness.py \
  --database-url "$DATABASE_URL" \
  --symbols "$SYMBOLS" \
  --timeframe "$PRIMARY_TF" \
  --lookback-days "$LOOKBACK_DAYS"

log "[5/5] as-of alignment check"
set +e
"$PYTHON_BIN" scripts/validate_asof_alignment.py --database-url "$DATABASE_URL"
ASOF_RC=$?
set -e
if [[ "$ASOF_RC" -ne 0 ]]; then
  if [[ "$STRICT_ASOF_AFTER_IMPORT" == "1" ]]; then
    echo "[ERR] validate_asof_alignment failed rc=$ASOF_RC" >&2
    exit "$ASOF_RC"
  fi
  log "validate_asof_alignment rc=$ASOF_RC (non-blocking; STRICT_ASOF_AFTER_IMPORT=${STRICT_ASOF_AFTER_IMPORT})"
fi

log "done"
