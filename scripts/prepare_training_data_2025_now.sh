#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/artifacts/server_bundle}"
START_TS="${START_TS:-2025-01-01T00:00:00Z}"
END_TS="${END_TS:-}"
DAY_STEP="${DAY_STEP:-7}"
SLEEP_SEC="${SLEEP_SEC:-0.05}"
PRIMARY_TF="${PRIMARY_TF:-5m}"
SECONDARY_TF="${SECONDARY_TF:-1h}"
INGEST_WORKERS="${INGEST_WORKERS:-3}"
MAX_GOOGLE_SHARE="${MAX_GOOGLE_SHARE:-0.68}"
MIN_GOOGLE_EVENTS="${MIN_GOOGLE_EVENTS:-2500}"
ENABLE_SOCIAL_BACKFILL="${ENABLE_SOCIAL_BACKFILL:-1}"

mkdir -p "${OUT_DIR}"
END_ARGS=()
if [[ -n "${END_TS}" ]]; then
  END_ARGS=(--end "${END_TS}")
fi

echo "[1/4] build market bars (top10, ${PRIMARY_TF}, 2025-now)"
python3 "${ROOT_DIR}/scripts/ingest_bitget_market_bars.py" \
  --market perp \
  --timeframe "${PRIMARY_TF}" \
  --start "${START_TS}" \
  "${END_ARGS[@]}" \
  --out-csv "${OUT_DIR}/market_bars_top10_${PRIMARY_TF}_2025_now.csv" \
  --skip-db \
  --fallback-source none \
  --workers "${INGEST_WORKERS}"

echo "[2/4] optional secondary bars (${SECONDARY_TF})"
python3 "${ROOT_DIR}/scripts/ingest_bitget_market_bars.py" \
  --market perp \
  --timeframe "${SECONDARY_TF}" \
  --start "${START_TS}" \
  "${END_ARGS[@]}" \
  --out-csv "${OUT_DIR}/market_bars_top10_${SECONDARY_TF}_2025_now.csv" \
  --skip-db \
  --fallback-source coingecko \
  --workers "${INGEST_WORKERS}"

echo "[3/5] build multi-source events (google+gdelt+official rss)"
python3 "${ROOT_DIR}/scripts/build_multisource_events_2025.py" \
  --start "${START_TS}" \
  "${END_ARGS[@]}" \
  --day-step "${DAY_STEP}" \
  --sleep-sec "${SLEEP_SEC}" \
  --max-google-share "${MAX_GOOGLE_SHARE}" \
  --min-google-events "${MIN_GOOGLE_EVENTS}" \
  --out-jsonl "${OUT_DIR}/events_multisource_2025_now.jsonl"

if [[ "${ENABLE_SOCIAL_BACKFILL}" == "1" ]]; then
  echo "[4/5] build social-source events (x/reddit/youtube/telegram)"
  python3 "${ROOT_DIR}/scripts/backfill_social_history.py" \
    --out-jsonl "${OUT_DIR}/social_history_2025_now.jsonl" || true
else
  echo "[4/5] skip social backfill (ENABLE_SOCIAL_BACKFILL=${ENABLE_SOCIAL_BACKFILL})"
fi

echo "[5/5] bundle summary"
wc -l \
  "${OUT_DIR}/market_bars_top10_${PRIMARY_TF}_2025_now.csv" \
  "${OUT_DIR}/market_bars_top10_${SECONDARY_TF}_2025_now.csv" \
  "${OUT_DIR}/events_multisource_2025_now.jsonl"
if [[ -f "${OUT_DIR}/social_history_2025_now.jsonl" ]]; then
  wc -l "${OUT_DIR}/social_history_2025_now.jsonl"
fi
ls -lh \
  "${OUT_DIR}/market_bars_top10_${PRIMARY_TF}_2025_now.csv" \
  "${OUT_DIR}/market_bars_top10_${SECONDARY_TF}_2025_now.csv" \
  "${OUT_DIR}/events_multisource_2025_now.jsonl"
if [[ -f "${OUT_DIR}/social_history_2025_now.jsonl" ]]; then
  ls -lh "${OUT_DIR}/social_history_2025_now.jsonl"
fi

echo "done"
