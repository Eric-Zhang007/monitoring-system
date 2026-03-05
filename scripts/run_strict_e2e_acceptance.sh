#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

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
        cur.execute("SELECT default_version FROM pg_available_extensions WHERE name='vector'")
        row = cur.fetchone()
        if row is None:
            raise RuntimeError(
                "pgvector_extension_unavailable: install pgvector on PostgreSQL host "
                "(example Debian/Ubuntu: sudo apt-get install postgresql-16-pgvector), then rerun"
            )
print('pgvector_available_ok')
PY

"${PYTHON_BIN}" - <<'PY'
import psycopg2, os
from pathlib import Path

dsn = os.getenv('DATABASE_URL', 'postgresql://monitor@localhost:5432/monitor')
root = Path(os.getenv("ROOT_DIR", "."))
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        sql = (root / 'scripts' / 'init_db.sql').read_text(encoding='utf-8')
        cur.execute(sql)
    conn.commit()
print('init_db_ok')
PY

ROOT_DIR="${ROOT_DIR}" "${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_feature_store.py" --start 2018-01-01T00:00:00Z
TEXT_EMBED_MODEL_PATH="${TEXT_EMBED_MODEL_PATH:-artifacts/models/text_encoder/multilingual-e5-small}"
if [[ ! -d "${TEXT_EMBED_MODEL_PATH}" ]]; then
  echo "text encoder dir missing: ${TEXT_EMBED_MODEL_PATH}"
  echo "run: ${PYTHON_BIN} ${ROOT_DIR}/scripts/setup_text_encoder.py --out-dir ${TEXT_EMBED_MODEL_PATH}"
  exit 2
fi
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/build_text_embeddings.py" --start 2018-01-01T00:00:00Z --model-path "${TEXT_EMBED_MODEL_PATH}"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/merge_feature_views.py" --start 2018-01-01T00:00:00Z

"${PYTHON_BIN}" "${ROOT_DIR}/training/train_liquid.py" --model-id liquid_main --out-dir artifacts/models/liquid_main

"${PYTHON_BIN}" "${ROOT_DIR}/inference/main.py" --symbol BTC

pytest -q \
  tests/test_clean_clone_imports.py \
  tests/test_init_db_has_universe_table.py \
  tests/test_init_db_has_control_plane_tables.py \
  tests/test_train_report_records_universe.py \
  tests/test_offline_data_audit_minimal_tables.py \
  tests/test_runtime_config_hot_reload.py \
  tests/test_process_manager_start_stop.py \
  tests/test_email_command_parser.py \
  tests/test_connectivity_probe_mocked.py \
  tests/test_connectivity_probe_with_proxy_profile.py \
  tests/test_proxy_profile_binding.py \
  tests/test_live_start_blocked_when_unreachable.py \
  tests/test_clock_drift_blocks_live.py \
  tests/test_rbac_permissions.py \
  tests/test_secrets_not_returned.py \
  tests/test_email_risk_notification.py \
  tests/test_live_adapter_idempotency.py \
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
