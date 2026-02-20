#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TESTS="${RUN_TESTS:-1}"
if [[ "${1:-}" == "--skip-tests" ]]; then
  RUN_TESTS="0"
fi

echo "[1/5] validate .env.example keys"
required_env_keys=(
  DATABASE_URL
  REDIS_URL
  CORS_ALLOW_ORIGINS
  CORS_ALLOW_CREDENTIALS
  GF_SECURITY_ADMIN_PASSWORD
)
for key in "${required_env_keys[@]}"; do
  if ! rg -q "^${key}=" .env.example; then
    echo "[FAIL] .env.example missing key: ${key}"
    exit 2
  fi
done

echo "[2/5] block known insecure defaults"
if rg -n "postgresql://[^[:space:]\"']*change_me_please|GF_SECURITY_ADMIN_PASSWORD=admin|admin/admin" .env.example scripts/server_up.sh backend/main.py README.md >/tmp/security_hardening_hits.log; then
  echo "[FAIL] insecure default detected"
  cat /tmp/security_hardening_hits.log
  exit 2
fi

if rg -n 'allow_origins=\["\*"\]' backend/main.py >/tmp/security_hardening_cors_hits.log; then
  echo "[FAIL] wildcard CORS origin configured in backend/main.py"
  cat /tmp/security_hardening_cors_hits.log
  exit 2
fi

echo "[3/5] validate strict alignment hard gate"
if ! rg -q '^BACKTEST_STRICT_ASOF_HARD_FAIL=1' .env.example; then
  echo "[FAIL] .env.example missing BACKTEST_STRICT_ASOF_HARD_FAIL=1"
  exit 2
fi

echo "[4/5] validate CORS settings from environment"
PYTHONPATH="${ROOT_DIR}/backend" python3 - <<'PY'
from security_config import build_cors_settings

cfg = build_cors_settings()
print(f"[OK] cors_allow_origins={cfg['allow_origins']}")
print(f"[OK] cors_allow_credentials={cfg['allow_credentials']}")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[5/5] run security config tests"
  if PYTHONPATH="${ROOT_DIR}/backend" python3 -c "import pytest" >/dev/null 2>&1; then
    PYTHONPATH="${ROOT_DIR}/backend" python3 -m pytest -q backend/tests/test_security_config.py
  else
    echo "[WARN] pytest not available in current python; skipping tests"
  fi
else
  echo "[5/5] security tests skipped (RUN_TESTS=${RUN_TESTS})"
fi

echo "[OK] security hardening validation passed"
