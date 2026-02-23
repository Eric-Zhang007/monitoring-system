#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-all}" # runtime|train|dev|all
PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "[1/3] bootstrap python env"
PROFILE="$PROFILE" PYTHON_BIN="$PYTHON_BIN" bash scripts/bootstrap_env.sh
source .venv/bin/activate

echo "[2/3] ensure .env exists"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "[INFO] .env created from .env.example; please review credentials."
fi

echo "[3/3] done"
cat <<'EOF'
Next steps:
1) export DATABASE_URL=postgresql://<user>:<pass>@<host>:5432/<db>
2) python - <<'PY'
import os
from pathlib import Path
import psycopg2
dsn = os.getenv("DATABASE_URL")
with psycopg2.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(Path("scripts/init_db.sql").read_text(encoding="utf-8"))
    conn.commit()
print("init_db_ok")
PY
3) python scripts/setup_text_encoder.py --model-id intfloat/multilingual-e5-small --out-dir artifacts/models/text_encoder/multilingual-e5-small
4) run training/inference commands from README.md
EOF
