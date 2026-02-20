#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
POSTGRES_ADMIN_USER="${POSTGRES_ADMIN_USER:-postgres}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "$PYTHON_BIN" == */* ]]; then
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "[FAIL] python interpreter not executable: $PYTHON_BIN"
    exit 2
  fi
else
  if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $PYTHON_BIN"
    exit 2
  fi
fi

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "[FAIL] DATABASE_URL is required"
  exit 2
fi

tmp_py="$(mktemp /tmp/ms_pg_sync_XXXX.py)"
trap 'rm -f "$tmp_py"' EXIT
cat >"$tmp_py" <<'PY'
import os
import re
import psycopg2

raw = os.getenv("DATABASE_URL", "").strip()
m = re.match(r"([^:]+)://([^:@]+):?([^@]*)@([^:/?#]+)(?::(\d+))?/(.+)", raw)
if not m:
    raise SystemExit("invalid DATABASE_URL")
_, db_user, db_pass, _, _, db_name = m.groups()

conn = psycopg2.connect(dbname="postgres", user="postgres")
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (db_user,))
if cur.fetchone() is None:
    cur.execute(f'CREATE ROLE "{db_user}" LOGIN PASSWORD %s', (db_pass,))
else:
    cur.execute(f'ALTER ROLE "{db_user}" WITH LOGIN PASSWORD %s', (db_pass,))
cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
if cur.fetchone() is None:
    cur.execute(f'CREATE DATABASE "{db_name}" OWNER "{db_user}"')
else:
    cur.execute(f'ALTER DATABASE "{db_name}" OWNER TO "{db_user}"')
cur.close()
conn.close()
print("db_role_db_synced")
print(f"db_user={db_user}")
print(f"db_name={db_name}")
PY
chmod 0644 "$tmp_py"

echo "[info] syncing postgres role/database from DATABASE_URL via sudo user=${POSTGRES_ADMIN_USER}"
sudo -u "$POSTGRES_ADMIN_USER" env DATABASE_URL="$DATABASE_URL" "$PYTHON_BIN" "$tmp_py"
echo "[ok] postgres role/database sync finished"
