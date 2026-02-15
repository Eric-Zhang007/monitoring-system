#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-180}"
PARITY_MAX_DEVIATION="${PARITY_MAX_DEVIATION:-0.10}"
PARITY_MIN_COMPLETED_RUNS="${PARITY_MIN_COMPLETED_RUNS:-5}"

echo "[gate] hard metrics (liquid/enforce)"
python3 scripts/evaluate_hard_metrics.py --track liquid --lookback-days "${LOOKBACK_DAYS}" --enforce

echo "[gate] parity 30d"
python3 scripts/check_backtest_paper_parity.py \
  --track liquid \
  --max-deviation "${PARITY_MAX_DEVIATION}" \
  --min-completed-runs "${PARITY_MIN_COMPLETED_RUNS}"

echo "[gate] alerts config"
python3 scripts/validate_phase45_alerts.py

echo "ci phase4-5 gate passed"
