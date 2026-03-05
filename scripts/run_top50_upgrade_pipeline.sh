#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
if [[ -f "${ROOT_DIR}/.env" ]]; then
  set -a
  source "${ROOT_DIR}/.env"
  set +a
fi

AS_OF="${AS_OF:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
START="${START:-2025-01-01T00:00:00Z}"
END="${END:-$AS_OF}"
BAR_SIZE="${BAR_SIZE:-5m}"
PREPARE_FEATURE_MATRIX="${PREPARE_FEATURE_MATRIX:-1}"

UNIVERSE_SNAPSHOT="${UNIVERSE_SNAPSHOT:-$ROOT_DIR/artifacts/universe/liquid_top50_snapshot.json}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/artifacts/cache/liquid_top50}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$ROOT_DIR/artifacts/models/liquid_main}"
EVAL_DIR="${EVAL_DIR:-$ROOT_DIR/artifacts/eval/top50_latest}"
READINESS_FILE="${READINESS_FILE:-$ROOT_DIR/artifacts/audit/top50_data_readiness_latest.json}"
AUDIT_PARALLEL_WORKERS="${AUDIT_PARALLEL_WORKERS:-4}"

echo "[1/10] update universe"
"$PYTHON_BIN" "$ROOT_DIR/scripts/update_universe_top50.py" --as-of "$AS_OF" --snapshot-file "$UNIVERSE_SNAPSHOT"

UNIVERSE_SYMBOLS="$(UNIVERSE_SNAPSHOT="${UNIVERSE_SNAPSHOT}" "$PYTHON_BIN" - <<'PY'
import json, os
from pathlib import Path
p=Path(os.environ["UNIVERSE_SNAPSHOT"])
obj=json.loads(p.read_text(encoding="utf-8"))
syms=[str(x).strip().upper() for x in (obj.get("symbols") or []) if str(x).strip()]
print(",".join(syms))
PY
)"
if [[ -z "${UNIVERSE_SYMBOLS}" ]]; then
  echo "universe symbols empty: ${UNIVERSE_SNAPSHOT}"
  exit 2
fi

if [[ "${PREPARE_FEATURE_MATRIX}" == "1" ]]; then
  echo "[2/10] build feature matrix from market bars"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/build_feature_store.py" \
    --start "$START" \
    --end "$END" \
    --timeframe "$BAR_SIZE" \
    --symbols "$UNIVERSE_SYMBOLS"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/merge_feature_views.py" \
    --start "$START" \
    --end "$END"
else
  echo "[2/10] build feature matrix skipped"
fi

echo "[3/10] audit offline training data"
if [[ "${AUDIT_PARALLEL_WORKERS}" -gt 1 ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/scripts/audit_offline_training_data_parallel.py" \
    --track liquid \
    --symbols "" \
    --as-of "$AS_OF" \
    --top-n "${TOP_N:-50}" \
    --start "$START" \
    --end "$END" \
    --lookback "${LOOKBACK:-2016}" \
    --bucket "${BAR_SIZE}" \
    --parallel-workers "${AUDIT_PARALLEL_WORKERS}" \
    --output "$READINESS_FILE"
else
  "$PYTHON_BIN" "$ROOT_DIR/scripts/audit_offline_training_data.py" \
    --track liquid \
    --symbols "" \
    --as-of "$AS_OF" \
    --top-n "${TOP_N:-50}" \
    --start "$START" \
    --end "$END" \
    --lookback "${LOOKBACK:-2016}" \
    --bucket "${BAR_SIZE}" \
    --output "$READINESS_FILE"
fi

echo "[4/10] build cache"
"$PYTHON_BIN" "$ROOT_DIR/scripts/build_training_cache.py" \
  --universe-snapshot "$UNIVERSE_SNAPSHOT" \
  --start "$START" \
  --end "$END" \
  --readiness-file "$READINESS_FILE" \
  --exclude-blocked 1 \
  --output-dir "$CACHE_DIR"

echo "[5/10] cache audit"
"$PYTHON_BIN" "$ROOT_DIR/scripts/check_training_cache_audit.py" --cache-dir "$CACHE_DIR"

echo "[6/10] train"
"$PYTHON_BIN" "$ROOT_DIR/scripts/train_top50.py" --cache-dir "$CACHE_DIR" --out-dir "$ARTIFACT_DIR"

echo "[7/10] evaluate"
"$PYTHON_BIN" "$ROOT_DIR/scripts/eval_top50.py" \
  --artifact-dir "$ARTIFACT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --universe-snapshot "$UNIVERSE_SNAPSHOT" \
  --out-dir "$EVAL_DIR"

echo "[8/10] pack"
"$PYTHON_BIN" "$ROOT_DIR/scripts/pack_artifact.py" \
  --artifact-dir "$ARTIFACT_DIR" \
  --universe-snapshot "$UNIVERSE_SNAPSHOT" \
  --eval-dir "$EVAL_DIR"

echo "[9/10] inference smoke"
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_inference_smoke.py" --artifact-dir "$ARTIFACT_DIR" --cache-dir "$CACHE_DIR"

echo "[10/10] decision smoke"
"$PYTHON_BIN" "$ROOT_DIR/scripts/run_decision_smoke.py"

echo "done"
