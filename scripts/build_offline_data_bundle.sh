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
SECONDARY_TF="${SECONDARY_TF:-1h}"
SYMBOLS="${SYMBOLS:-BTC,ETH,SOL,BNB,XRP,ADA,DOGE,TRX,AVAX,LINK}"
BITGET_SYMBOLS="${BITGET_SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT}"
BITGET_SYMBOL_MAP="${BITGET_SYMBOL_MAP:-BTCUSDT:BTC,ETHUSDT:ETH,SOLUSDT:SOL,BNBUSDT:BNB,XRPUSDT:XRP,ADAUSDT:ADA,DOGEUSDT:DOGE,TRXUSDT:TRX,AVAXUSDT:AVAX,LINKUSDT:LINK}"

RUN_MARKET="${RUN_MARKET:-1}"
RUN_MARKET_SECONDARY="${RUN_MARKET_SECONDARY:-0}"
RUN_AUX="${RUN_AUX:-1}"
RUN_DERIVATIVES="${RUN_DERIVATIVES:-1}"
RUN_ORDERBOOK_PROXY="${RUN_ORDERBOOK_PROXY:-1}"
RUN_EVENTS="${RUN_EVENTS:-1}"
RUN_SOCIAL="${RUN_SOCIAL:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_EXPORT="${RUN_EXPORT:-1}"
RUN_PACKAGE="${RUN_PACKAGE:-1}"

MARKET_FALLBACK_SOURCE="${MARKET_FALLBACK_SOURCE:-none}"
INGEST_WORKERS="${INGEST_WORKERS:-3}"
INGEST_CHUNK_DAYS="${INGEST_CHUNK_DAYS:-30}"
INGEST_MAX_CHUNKS="${INGEST_MAX_CHUNKS:-0}"

EVENT_CHUNK_DAYS="${EVENT_CHUNK_DAYS:-30}"
EVENT_MAX_CHUNKS="${EVENT_MAX_CHUNKS:-0}"
EVENT_SOCIAL_SOURCES="${EVENT_SOCIAL_SOURCES:-twitter,reddit,youtube,telegram}"

OUT_ROOT="${OUT_ROOT:-$ROOT_DIR/artifacts/offline_bundle}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUT_ROOT/run_${RUN_ID}"
mkdir -p "$RUN_DIR"

END_ARGS=()
if [[ -n "$END_TS" ]]; then
  END_ARGS=(--end "$END_TS")
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

log "offline bundle run_id=${RUN_ID}"

if [[ "$RUN_MARKET" == "1" ]]; then
  log "[1/10] collect market bars (${TIMEFRAME}) -> local db + csv"
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
    --checkpoint-file "$RUN_DIR/market_${TIMEFRAME}_checkpoint.json"
    --resume
    --allow-partial
    --out-csv "$RUN_DIR/raw_market_bars_${TIMEFRAME}.csv"
  )
  if [[ "$INGEST_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$INGEST_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  "${CMD[@]}" | tee "$RUN_DIR/market_${TIMEFRAME}.log"
else
  log "[1/10] skip market ${TIMEFRAME} (RUN_MARKET=${RUN_MARKET})"
fi

if [[ "$RUN_MARKET_SECONDARY" == "1" ]]; then
  log "[2/10] collect market bars (${SECONDARY_TF}) -> local db + csv"
  CMD=(
    "$PYTHON_BIN" scripts/ingest_bitget_market_bars.py
    --database-url "$DATABASE_URL"
    --market perp
    --timeframe "$SECONDARY_TF"
    --start "$START_TS"
    --symbols "$BITGET_SYMBOLS"
    --symbol-map "$BITGET_SYMBOL_MAP"
    --fallback-source coingecko
    --workers "$INGEST_WORKERS"
    --chunk-days "$INGEST_CHUNK_DAYS"
    --checkpoint-file "$RUN_DIR/market_${SECONDARY_TF}_checkpoint.json"
    --resume
    --allow-partial
    --out-csv "$RUN_DIR/raw_market_bars_${SECONDARY_TF}.csv"
  )
  if [[ "$INGEST_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$INGEST_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  "${CMD[@]}" | tee "$RUN_DIR/market_${SECONDARY_TF}.log"
else
  log "[2/10] skip market ${SECONDARY_TF} (RUN_MARKET_SECONDARY=${RUN_MARKET_SECONDARY})"
fi

if [[ "$RUN_AUX" == "1" ]]; then
  log "[3/10] collect aux signals (funding + OI proxy)"
  "$PYTHON_BIN" scripts/ingest_binance_aux_signals.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --start "$START_TS" \
    --timeframe "$TIMEFRAME" \
    --out-json "$RUN_DIR/aux_signals.json" \
    "${END_ARGS[@]}"
else
  log "[3/10] skip aux signals (RUN_AUX=${RUN_AUX})"
fi

if [[ "$RUN_DERIVATIVES" == "1" ]]; then
  log "[4/10] collect derivatives regime signals"
  "$PYTHON_BIN" scripts/ingest_binance_derivatives_signals.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --start "$START_TS" \
    --period "$TIMEFRAME" \
    --out-json "$RUN_DIR/derivatives_signals.json" \
    "${END_ARGS[@]}"
else
  log "[4/10] skip derivatives signals (RUN_DERIVATIVES=${RUN_DERIVATIVES})"
fi

if [[ "$RUN_ORDERBOOK_PROXY" == "1" ]]; then
  log "[5/10] build orderbook proxy from market_bars"
  "$PYTHON_BIN" scripts/backfill_orderbook_from_market_bars.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$TIMEFRAME" \
    --start "$START_TS" \
    --out-json "$RUN_DIR/orderbook_proxy.json" \
    "${END_ARGS[@]}"
else
  log "[5/10] skip orderbook proxy (RUN_ORDERBOOK_PROXY=${RUN_ORDERBOOK_PROXY})"
fi

if [[ "$RUN_EVENTS" == "1" || "$RUN_SOCIAL" == "1" ]]; then
  log "[6/10] collect event/social jsonl (chunked orchestrator)"
  CMD=(
    "$PYTHON_BIN" scripts/orchestrate_event_social_backfill.py
    --start "$START_TS"
    --chunk-days "$EVENT_CHUNK_DAYS"
    --checkpoint-file "$RUN_DIR/event_social_checkpoint.json"
    --resume
    --social-sources "$EVENT_SOCIAL_SOURCES"
    --out-events-jsonl "$RUN_DIR/events_multisource_2018_now.jsonl"
    --out-social-jsonl "$RUN_DIR/social_history_2018_now.jsonl"
  )
  if [[ "$EVENT_MAX_CHUNKS" -gt 0 ]]; then
    CMD+=(--max-chunks "$EVENT_MAX_CHUNKS")
  fi
  if [[ -n "$END_TS" ]]; then
    CMD+=(--end "$END_TS")
  fi
  if [[ "$RUN_EVENTS" != "1" ]]; then
    CMD+=(--skip-events)
  fi
  if [[ "$RUN_SOCIAL" != "1" ]]; then
    CMD+=(--skip-social)
  fi
  "${CMD[@]}" | tee "$RUN_DIR/event_social.log"
else
  log "[6/10] skip event/social collection (RUN_EVENTS=${RUN_EVENTS}, RUN_SOCIAL=${RUN_SOCIAL})"
fi

if [[ "$RUN_AUDIT" == "1" ]]; then
  log "[7/10] run audits"
  "$PYTHON_BIN" scripts/audit_full_history_completeness.py \
    --database-url "$DATABASE_URL" \
    --start 2018-01-01T00:00:00Z \
    --timeframe "$TIMEFRAME" \
    --symbols "$SYMBOLS" \
    "${END_ARGS[@]}" | tee "$RUN_DIR/audit_full_history.json"

  "$PYTHON_BIN" scripts/audit_training_data_completeness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$TIMEFRAME" \
    --lookback-days 420 | tee "$RUN_DIR/audit_training_420d.json"

  "$PYTHON_BIN" scripts/audit_training_data_completeness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$SECONDARY_TF" \
    --lookback-days 420 | tee "$RUN_DIR/audit_training_420d_${SECONDARY_TF}.json"

  "$PYTHON_BIN" scripts/audit_required_data_readiness.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --primary-timeframe "$TIMEFRAME" \
    --secondary-timeframe "$SECONDARY_TF" \
    --start "$START_TS" \
    "${END_ARGS[@]}" | tee "$RUN_DIR/audit_required_readiness.json"
else
  log "[7/10] skip audits (RUN_AUDIT=${RUN_AUDIT})"
fi

if [[ "$RUN_EXPORT" == "1" ]]; then
  log "[8/10] export market/aux/orderbook csv from local db"
  "$PYTHON_BIN" scripts/export_liquid_data_csv.py \
    --database-url "$DATABASE_URL" \
    --symbols "$SYMBOLS" \
    --timeframe "$TIMEFRAME" \
    --start "$START_TS" \
    --out-dir "$RUN_DIR" \
    --prefix bundle \
    "${END_ARGS[@]}" | tee "$RUN_DIR/export_liquid_data.log"
else
  log "[8/10] skip db export (RUN_EXPORT=${RUN_EXPORT})"
fi

log "[9/10] build bundle manifest"
"$PYTHON_BIN" - "$RUN_DIR" "$RUN_ID" "$TIMEFRAME" "$SECONDARY_TF" "$START_TS" "${END_TS:-}" "$SYMBOLS" <<'PY'
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

run_dir = Path(sys.argv[1]).resolve()
run_id = str(sys.argv[2])
timeframe = str(sys.argv[3])
secondary_tf = str(sys.argv[4])
start_ts = str(sys.argv[5])
end_ts = str(sys.argv[6])
symbols = [x.strip().upper() for x in str(sys.argv[7]).split(',') if x.strip()]

files = [
    f"bundle_market_bars_{timeframe}.csv",
    "bundle_orderbook_l2.csv",
    "bundle_funding_rates.csv",
    "bundle_onchain_signals.csv",
    "events_multisource_2018_now.jsonl",
    "social_history_2018_now.jsonl",
    f"raw_market_bars_{timeframe}.csv",
    "raw_market_bars_1h.csv",
    "audit_full_history.json",
    "audit_training_420d.json",
    f"audit_training_420d_{secondary_tf}.json",
    "audit_required_readiness.json",
    "derivatives_signals.json",
]

def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size <= 0:
        return 0
    if path.suffix.lower() == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            return max(0, sum(1 for _ in f) - 1)
    with open(path, 'r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

items = []
for name in files:
    p = run_dir / name
    if not p.exists():
        continue
    items.append(
        {
            'file': name,
            'bytes': int(p.stat().st_size),
            'rows': int(count_rows(p)),
            'sha256': sha256(p),
        }
    )

manifest = {
    'status': 'ok',
    'bundle_type': 'offline_data_bundle_v1',
    'run_id': run_id,
    'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
    'window': {
        'start': start_ts,
        'end': end_ts,
        'timeframe': timeframe,
    },
    'symbols': symbols,
    'files': items,
}

out = run_dir / 'bundle_manifest.json'
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'ok', 'manifest': str(out), 'file_count': len(items)}, ensure_ascii=False))
PY

if [[ "$RUN_PACKAGE" == "1" ]]; then
  log "[10/10] create tar.gz bundle"
  TAR_PATH="$OUT_ROOT/offline_data_bundle_${RUN_ID}.tar.gz"
  tar -C "$RUN_DIR" -czf "$TAR_PATH" .
  sha256sum "$TAR_PATH" | tee "$TAR_PATH.sha256"
  log "bundle_tar=$TAR_PATH"
  log "bundle_sha256=$TAR_PATH.sha256"
else
  log "[10/10] skip packaging (RUN_PACKAGE=${RUN_PACKAGE})"
fi

log "done"
log "run_dir=$RUN_DIR"
