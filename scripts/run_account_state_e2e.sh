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

DB_READY=0
if "${PYTHON_BIN}" - <<'PY'
import os
import psycopg2
dsn = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
try:
    with psycopg2.connect(dsn):
        pass
    print("db_connection_ok")
except Exception:
    raise SystemExit(1)
PY
then
  DB_READY=1
else
  echo "db_connection_unavailable: skipping live-db daemon checks"
fi

if [[ "${DB_READY}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path
import psycopg2

dsn = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(Path("scripts/init_db.sql").read_text(encoding="utf-8"))
    conn.commit()
print("init_db_ok")
PY

  "${PYTHON_BIN}" monitoring/account_state_daemon.py --database-url "${DATABASE_URL:-postgresql://monitor@localhost:5432/monitor}" --adapter paper --venue coinbase --ttl-sec 10 --refresh-sec 5 --fast-sec 1
  "${PYTHON_BIN}" monitoring/reconciliation_daemon.py --database-url "${DATABASE_URL:-postgresql://monitor@localhost:5432/monitor}"
fi

pytest -q \
  tests/test_account_state_contract.py \
  tests/test_account_state_ttl_enforced.py \
  tests/test_signal_pipeline_uses_account_state.py \
  tests/test_execution_e2e_paper.py \
  tests/test_reconciliation_triggers_red.py \
  tests/test_recon_failure_triggers_kill_switch.py

echo "account_state_e2e_ok"
