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

"${PYTHON_BIN}" - <<'PY'
import psycopg2, os
from pathlib import Path

dsn = os.getenv('DATABASE_URL', 'postgresql://monitor@localhost:5432/monitor')
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        sql = Path('scripts/init_db.sql').read_text(encoding='utf-8')
        cur.execute(sql)
    conn.commit()
print('init_db_ok')
PY

"${PYTHON_BIN}" scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
TEXT_EMBED_MODEL_PATH="${TEXT_EMBED_MODEL_PATH:-artifacts/models/text_encoder/multilingual-e5-small}"
if [[ ! -d "${TEXT_EMBED_MODEL_PATH}" ]]; then
  echo "text encoder dir missing: ${TEXT_EMBED_MODEL_PATH}"
  echo "run: ${PYTHON_BIN} scripts/setup_text_encoder.py --out-dir ${TEXT_EMBED_MODEL_PATH}"
  exit 2
fi
"${PYTHON_BIN}" scripts/build_text_embeddings.py --start 2018-01-01T00:00:00Z --model-path "${TEXT_EMBED_MODEL_PATH}"
"${PYTHON_BIN}" scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z

"${PYTHON_BIN}" training/train_liquid.py --model-id liquid_main --out-dir artifacts/models/liquid_main

"${PYTHON_BIN}" inference/main.py --symbol BTC

pytest -q \
  tests/test_train_infer_parity.py \
  tests/test_execution_e2e_paper.py \
  tests/test_account_state_contract.py \
  tests/test_account_state_ttl_enforced.py \
  tests/test_signal_pipeline_uses_account_state.py \
  tests/test_risk_manager_soft_penalties.py \
  tests/test_position_sizer_monotonic.py \
  tests/test_execution_style_selector.py \
  tests/test_vol_spike_switches_to_marketable.py \
  tests/test_reconciliation_triggers_red.py \
  tests/test_recon_failure_triggers_kill_switch.py \
  tests/test_training_loss_outputs_calibratable.py
