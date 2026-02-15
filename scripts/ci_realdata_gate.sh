#!/usr/bin/env bash
set -euo pipefail

LOOKBACK_DAYS="${LOOKBACK_DAYS:-500}"
TARGETS="${TARGETS:-BTC_BG2025_PERP,ETH_BG2025_PERP,SOL_BG2025_PERP,BTC_BG2025_SPOT,ETH_BG2025_SPOT,SOL_BG2025_SPOT}"

echo "[real-gate] strict backtest contract"
python3 scripts/validate_backtest_contracts.py \
  --track liquid \
  --lookback-days "${LOOKBACK_DAYS}" \
  --score-source model \
  --include-sources prod \
  --exclude-sources smoke,async_test,maintenance \
  --data-regimes prod_live \
  --min-valid 20 \
  --enforce

echo "[real-gate] hard metrics (liquid/prod-only/enforce)"
python3 scripts/evaluate_hard_metrics.py \
  --track liquid \
  --lookback-days "${LOOKBACK_DAYS}" \
  --enforce \
  --score-source model \
  --include-sources prod \
  --exclude-sources smoke,async_test,maintenance \
  --data-regimes prod_live \
  --targets "${TARGETS}"

echo "[real-gate] parity 30d (prod-only, target-scoped)"
python3 scripts/check_backtest_paper_parity.py \
  --track liquid \
  --max-deviation 0.10 \
  --min-completed-runs 5 \
  --score-source model \
  --include-sources prod \
  --exclude-sources smoke,async_test,maintenance \
  --data-regimes prod_live \
  --targets "${TARGETS}"

echo "[real-gate] alerts config"
python3 scripts/validate_phase45_alerts.py

echo "[real-gate] status snapshot sync"
python3 scripts/generate_status_snapshot.py --write >/dev/null
python3 scripts/verify_status_snapshot.py

echo "ci realdata gate passed"
