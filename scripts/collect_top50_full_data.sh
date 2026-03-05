#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT00:00:00Z)}"
START="${START:-2018-01-01T00:00:00Z}"
END="${END:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
TOP_N="${TOP_N:-50}"
BAR_SIZE="${BAR_SIZE:-5m}"
TRACK="${TRACK:-liquid}"
SNAPSHOT_FILE="${SNAPSHOT_FILE:-${ROOT_DIR}/artifacts/universe/liquid_top50_snapshot.json}"
READINESS_OUT="${READINESS_OUT:-${ROOT_DIR}/artifacts/audit/top50_data_readiness_latest.json}"
ALLOW_SYNTHETIC="${ALLOW_SYNTHETIC:-0}"
RUN_UNIVERSE_UPDATE="${RUN_UNIVERSE_UPDATE:-1}"

echo "[collect_top50_full_data] python=${PYTHON_BIN}"
echo "[collect_top50_full_data] as_of=${AS_OF} start=${START} end=${END} top_n=${TOP_N} bar_size=${BAR_SIZE}"

if [[ "${RUN_UNIVERSE_UPDATE}" == "1" ]]; then
  echo "[1/6] update universe snapshot (top${TOP_N})"
  "${PYTHON_BIN}" "${ROOT_DIR}/scripts/update_universe_top50.py" \
    --database-url "${DATABASE_URL}" \
    --track "${TRACK}" \
    --as-of "${AS_OF}" \
    --top-n "${TOP_N}" \
    --rank-by volume_usd_30d \
    --source db \
    --lookback-days 30 \
    --timeframe 1h \
    --min-notional-usd 0 \
    --exclude-stable 1 \
    --exclude-leveraged 1 \
    --hysteresis-keep-rank 60 \
    --snapshot-file "${SNAPSHOT_FILE}" \
    --persist-db 1
else
  echo "[1/6] update universe snapshot skipped (RUN_UNIVERSE_UPDATE=0)"
fi

SYMBOLS_CSV="$(SNAPSHOT_FILE="${SNAPSHOT_FILE}" "${PYTHON_BIN}" - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["SNAPSHOT_FILE"])
obj = json.loads(p.read_text(encoding="utf-8"))
syms = [str(x).strip().upper() for x in (obj.get("symbols") or []) if str(x).strip()]
if not syms:
    raise SystemExit("universe_snapshot_empty_symbols")
print(",".join(syms))
PY
)"
SYMBOL_COUNT="$(awk -F',' '{print NF}' <<<"${SYMBOLS_CSV}")"
echo "[collect_top50_full_data] symbols=${SYMBOL_COUNT}"

echo "[2/6] ingest funding + open-interest proxy (Binance aux)"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/ingest_binance_aux_signals.py" \
  --database-url "${DATABASE_URL}" \
  --symbols "${SYMBOLS_CSV}" \
  --start "${START}" \
  --end "${END}" \
  --timeframe "${BAR_SIZE}" \
  --oi-period "${BAR_SIZE}" \
  --oi-max-lookback-days 30 \
  --funding-limit 1000 \
  --oi-limit 500 \
  --max-funding-pages 64 \
  --max-oi-pages 96 \
  --sleep-sec 0.12 \
  --batch-size 2000 \
  --replace-window

echo "[3/6] rebuild feature_snapshots_main from market + aux (strict columns)"
BUILD_FEATURE_ARGS=(
  --database-url "${DATABASE_URL}"
  --start "${START}"
  --end "${END}"
  --timeframe "${BAR_SIZE}"
  --symbols "${SYMBOLS_CSV}"
  --truncate
)
if [[ "${ALLOW_SYNTHETIC}" == "1" ]]; then
  BUILD_FEATURE_ARGS+=(--allow-synthetic)
fi
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_feature_store.py" "${BUILD_FEATURE_ARGS[@]}"

echo "[4/6] rebuild semantic text embeddings"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_text_embeddings.py" \
  --database-url "${DATABASE_URL}" \
  --start "${START}" \
  --end "${END}" \
  --truncate

echo "[5/6] rebuild feature_matrix_main by merging snapshots + text embeddings"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_feature_views.py" \
  --database-url "${DATABASE_URL}" \
  --start "${START}" \
  --end "${END}" \
  --truncate

echo "[6/6] run offline readiness audit (top${TOP_N})"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/audit_offline_training_data.py" \
  --database-url "${DATABASE_URL}" \
  --track "${TRACK}" \
  --as-of "${AS_OF}" \
  --top-n "${TOP_N}" \
  --start "${START}" \
  --end "${END}" \
  --lookback 288 \
  --bucket "${BAR_SIZE}" \
  --output "${READINESS_OUT}"

echo "[collect_top50_full_data] done"
