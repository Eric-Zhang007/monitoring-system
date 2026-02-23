#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3.12}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python not found: ${PYTHON_BIN}"
  exit 2
fi

run_case() {
  local name="$1"
  shift
  echo "=== ablation ${name} ==="
  env "$@" "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path
ROOT = Path.cwd()
for p in (ROOT, ROOT / "backend", ROOT / "training"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from account_state.models import AccountHealth, AccountState, BalanceState, ExecutionStats
from risk_manager import RiskManager

account = AccountState(
    balances=BalanceState(cash=1000.0, equity=1000.0, free_margin=700.0, used_margin=300.0, margin_ratio=2.0),
    execution_stats=ExecutionStats(slippage_bps_p90=22.0, reject_rate_5m=0.05),
    health=AccountHealth(is_fresh=True, recon_ok=True, ws_ok=True, last_error=""),
)
risk = RiskManager().evaluate(account=account, symbol="BTC", order_intent={}, market_snapshot={"realized_vol": 0.09})
print({"regime": str(risk.regime.value), "hard_limits_ok": bool(risk.hard_limits_ok), "penalties": dict(risk.soft_penalty_factors)})
PY
  env "$@" pytest -q \
    tests/test_position_sizer_monotonic.py \
    tests/test_vol_spike_switches_to_marketable.py \
    tests/test_training_loss_outputs_calibratable.py
}

# A: most aggressive (guards off)
run_case "A" \
  ENABLE_ACCOUNT_STATE_GUARD=0 \
  ENABLE_RISK_HARD_LIMITS=0 \
  ENABLE_SOFT_PENALTIES=0 \
  ENABLE_STYLE_SWITCHING=0 \
  ENABLE_RECONCILIATION=0

# B: hard limits + reconciliation only
run_case "B" \
  ENABLE_ACCOUNT_STATE_GUARD=1 \
  ENABLE_RISK_HARD_LIMITS=1 \
  ENABLE_SOFT_PENALTIES=0 \
  ENABLE_STYLE_SWITCHING=0 \
  ENABLE_RECONCILIATION=1

# C: strict full path
run_case "C" \
  ENABLE_ACCOUNT_STATE_GUARD=1 \
  ENABLE_RISK_HARD_LIMITS=1 \
  ENABLE_SOFT_PENALTIES=1 \
  ENABLE_STYLE_SWITCHING=1 \
  ENABLE_RECONCILIATION=1

echo "risk_ablation_matrix_ok"
