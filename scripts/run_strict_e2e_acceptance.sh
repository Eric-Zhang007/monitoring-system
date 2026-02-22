#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
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

python3 scripts/build_feature_store.py --start 2018-01-01T00:00:00Z
TEXT_EMBED_MODEL_PATH="${TEXT_EMBED_MODEL_PATH:-artifacts/models/text_encoder/multilingual-e5-small}"
if [[ ! -d "${TEXT_EMBED_MODEL_PATH}" ]]; then
  echo "text encoder dir missing: ${TEXT_EMBED_MODEL_PATH}"
  echo "run: python3 scripts/setup_text_encoder.py --out-dir ${TEXT_EMBED_MODEL_PATH}"
  exit 2
fi
python3 scripts/build_text_embeddings.py --start 2018-01-01T00:00:00Z --model-path "${TEXT_EMBED_MODEL_PATH}"
python3 scripts/merge_feature_views.py --start 2018-01-01T00:00:00Z

python3 training/train_liquid.py --model-id liquid_main --out-dir artifacts/models/liquid_main

python3 inference/main.py --symbol BTC

pytest -q
