#!/usr/bin/env bash
set -euo pipefail

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-artifacts/manual_runs/${TS}}"
CPU_NPROC="$(nproc)"
PAR_DEFAULT="$(( CPU_NPROC * 3 ))"
if [ "${PAR_DEFAULT}" -gt 64 ]; then
  PAR_DEFAULT=64
fi
if [ "${PAR_DEFAULT}" -lt 8 ]; then
  PAR_DEFAULT=8
fi
PAR="${PARALLELISM:-${PAR_DEFAULT}}"
MIN_SHARPE_DAILY="${MIN_SHARPE_DAILY:-1.5}"
API_BASE="${API_BASE:-http://localhost:8000}"

mkdir -p "${OUT_DIR}"

if ! curl -fsS "${API_BASE}/health" > /dev/null; then
  echo "[ERROR] backend not reachable at ${API_BASE}; start backend first."
  exit 2
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "[ERROR] psql not found; install postgresql-client first."
  exit 2
fi

echo "[1/5] 2025全年回测（perp+spot）"
python3 scripts/run_bitget_2025_backtest.py \
  --market perp \
  --run-source maintenance \
  --lookback-days 420 \
  > "${OUT_DIR}/01a_bitget_2025_perp.jsonl" &
P1=$!
python3 scripts/run_bitget_2025_backtest.py \
  --market spot \
  --run-source maintenance \
  --lookback-days 420 \
  > "${OUT_DIR}/01b_bitget_2025_spot.jsonl" &
P2=$!
wait "${P1}" "${P2}"

echo "[2/5] 2025 PERP+SPOT 并发网格调参"
python3 scripts/tune_liquid_strategy_grid.py \
  --api-base "${API_BASE}" \
  --run-source maintenance \
  --data-regime maintenance_replay \
  --score-source model \
  --targets BTC_BG2025_PERP,ETH_BG2025_PERP,SOL_BG2025_PERP \
  --data-version bitget_2025_perp_1h \
  --lookback-days 420 \
  --train-days 35 \
  --test-days 7 \
  --fee-bps 5.0 \
  --slippage-bps 3.0 \
  --max-trials 64 \
  --request-timeout-sec 180 \
  --parallelism "${PAR}" \
  --max-retries 2 \
  --retry-backoff-sec 2 \
  --entry-grid 0.015,0.02,0.03,0.04 \
  --exit-grid 0.003,0.005,0.008 \
  --base-weight-grid 0.05,0.07,0.09 \
  --high-vol-mult-grid 0.4,0.55 \
  --cost-lambda-grid 0.8,1.2,1.8,2.2 \
  --min-turnover 0.01 \
  --min-trades 3 \
  --min-active-targets 2 \
  --top-k 10 \
  > "${OUT_DIR}/02a_tune_2025_perp.json" &
P3=$!

python3 scripts/tune_liquid_strategy_grid.py \
  --api-base "${API_BASE}" \
  --run-source maintenance \
  --data-regime maintenance_replay \
  --score-source model \
  --targets BTC_BG2025_SPOT,ETH_BG2025_SPOT,SOL_BG2025_SPOT \
  --data-version bitget_2025_spot_1h \
  --lookback-days 420 \
  --train-days 35 \
  --test-days 7 \
  --fee-bps 5.0 \
  --slippage-bps 3.0 \
  --max-trials 64 \
  --request-timeout-sec 180 \
  --parallelism "${PAR}" \
  --max-retries 2 \
  --retry-backoff-sec 2 \
  --entry-grid 0.015,0.02,0.03,0.04 \
  --exit-grid 0.003,0.005,0.008 \
  --base-weight-grid 0.05,0.07,0.09 \
  --high-vol-mult-grid 0.4,0.55 \
  --cost-lambda-grid 0.8,1.2,1.8,2.2 \
  --min-turnover 0.01 \
  --min-trades 3 \
  --min-active-targets 2 \
  --top-k 10 \
  > "${OUT_DIR}/02b_tune_2025_spot.json" &
P4=$!
wait "${P3}" "${P4}"

echo "[3/5] 无泄露检查（2025至今）"
python3 scripts/validate_no_leakage.py \
  --track liquid \
  --lookback-days 420 \
  --include-sources prod,maintenance \
  --exclude-sources smoke,async_test \
  --data-regimes prod_live,maintenance_replay \
  > "${OUT_DIR}/03_no_leakage_420d.json"

echo "[4/5] 硬指标检查（2025至今）"
python3 scripts/evaluate_hard_metrics.py \
  --track liquid \
  --lookback-days 420 \
  --include-sources prod,maintenance \
  --exclude-sources smoke,async_test \
  --data-regimes prod_live,maintenance_replay \
  --min-sharpe-daily "${MIN_SHARPE_DAILY}" \
  > "${OUT_DIR}/04_hard_metrics_420d.json"

echo "[5/5] GPU切换就绪评估（当前严格门禁）"
GPU_CUTOVER_MIN_SHARPE_DAILY="${MIN_SHARPE_DAILY}" python3 scripts/check_gpu_cutover_readiness.py \
  --track liquid \
  --lookback-days 180 \
  --include-sources prod \
  --exclude-sources smoke,async_test,maintenance \
  --data-regimes prod_live \
  > "${OUT_DIR}/05_gpu_cutover_readiness_180d.json"

echo "done=${OUT_DIR}"
