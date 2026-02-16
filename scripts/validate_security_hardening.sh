#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_TESTS="${RUN_TESTS:-1}"
if [[ "${1:-}" == "--skip-tests" ]]; then
  RUN_TESTS="0"
fi

echo "[1/4] validate .env.example keys"
required_env_keys=(
  DATABASE_URL
  REDIS_URL
  POSTGRES_PASSWORD
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

echo "[2/4] block known insecure defaults"
if rg -n "postgresql://[^[:space:]\"']*change_me_please|GF_SECURITY_ADMIN_PASSWORD=admin|admin/admin" docker-compose.yml scripts/server_nodocker_up.sh backend/main.py scripts/deploy.sh README.md >/tmp/security_hardening_hits.log; then
  echo "[FAIL] insecure default detected"
  cat /tmp/security_hardening_hits.log
  exit 2
fi

if rg -n 'allow_origins=\["\*"\]' backend/main.py >/tmp/security_hardening_cors_hits.log; then
  echo "[FAIL] wildcard CORS origin configured in backend/main.py"
  cat /tmp/security_hardening_cors_hits.log
  exit 2
fi

echo "[3/4] validate CORS settings from environment"
PYTHONPATH="${ROOT_DIR}/backend" python3 - <<'PY'
from security_config import build_cors_settings

cfg = build_cors_settings()
print(f"[OK] cors_allow_origins={cfg['allow_origins']}")
print(f"[OK] cors_allow_credentials={cfg['allow_credentials']}")
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  echo "[4/4] run security config tests"
  PYTHONPATH="${ROOT_DIR}/backend" python3 -m pytest -q backend/tests/test_security_config.py
else
  echo "[4/4] security tests skipped (RUN_TESTS=${RUN_TESTS})"
fi

echo "[OK] security hardening validation passed"
