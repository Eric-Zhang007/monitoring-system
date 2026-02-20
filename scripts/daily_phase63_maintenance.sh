#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${OUT_DIR:-./artifacts/phase63}"
ENFORCE_GATE="${ENFORCE_GATE:-0}"
STRICT_TARGETS="${STRICT_TARGETS:-BTC,ETH,SOL}"
STRICT_DATA_REGIMES="${STRICT_DATA_REGIMES:-prod_live}"
STRICT_INCLUDE_SOURCES="${STRICT_INCLUDE_SOURCES:-prod}"
STRICT_EXCLUDE_SOURCES="${STRICT_EXCLUDE_SOURCES:-smoke,async_test,maintenance}"
STRICT_SCORE_SOURCE="${STRICT_SCORE_SOURCE:-model}"
PROD_BATCH_RUNS="${PROD_BATCH_RUNS:-12}"
mkdir -p "$OUT_DIR"

python3 scripts/rebuild_liquid_completed_backtests.py --api-base "$API_BASE" --limit 50 > "$OUT_DIR/rebuild_${TS}.json" || true
python3 scripts/run_prod_live_backtest_batch.py \
  --api-base "$API_BASE" \
  --targets "$STRICT_TARGETS" \
  --n-runs "$PROD_BATCH_RUNS" \
  --lookback-days 180 \
  --train-days 56 \
  --test-days 14 \
  > "$OUT_DIR/prod_live_batch_${TS}.json" || true
python3 scripts/validate_backtest_contracts.py \
  --track liquid \
  --lookback-days 180 \
  --score-source "$STRICT_SCORE_SOURCE" \
  --include-sources "$STRICT_INCLUDE_SOURCES" \
  --exclude-sources "$STRICT_EXCLUDE_SOURCES" \
  --data-regimes "$STRICT_DATA_REGIMES" \
  --min-valid 20 \
  > "$OUT_DIR/contracts_${TS}.json" || true
python3 scripts/evaluate_hard_metrics.py \
  --track liquid \
  --lookback-days 180 \
  --score-source "$STRICT_SCORE_SOURCE" \
  --include-sources "$STRICT_INCLUDE_SOURCES" \
  --exclude-sources "$STRICT_EXCLUDE_SOURCES" \
  --data-regimes "$STRICT_DATA_REGIMES" \
  > "$OUT_DIR/hard_metrics_${TS}.json" || true
python3 scripts/check_backtest_paper_parity.py \
  --track liquid \
  --max-deviation 0.10 \
  --min-completed-runs 5 \
  --score-source "$STRICT_SCORE_SOURCE" \
  --include-sources "$STRICT_INCLUDE_SOURCES" \
  --exclude-sources "$STRICT_EXCLUDE_SOURCES" \
  --data-regimes "$STRICT_DATA_REGIMES" \
  > "$OUT_DIR/parity_${TS}.json" || true
python3 scripts/validate_phase45_alerts.py > "$OUT_DIR/alerts_${TS}.json" || true
python3 scripts/check_gpu_cutover_readiness.py > "$OUT_DIR/readiness_${TS}.json" || true
python3 scripts/generate_status_snapshot.py --write > "$OUT_DIR/status_snapshot_${TS}.md" || true

OUT_DIR="$OUT_DIR" TS="$TS" ENFORCE_GATE="$ENFORCE_GATE" python3 - << 'PY'
import json
import os

od = os.getenv("OUT_DIR","./artifacts/phase63")
ts = os.getenv("TS")
enforce = os.getenv("ENFORCE_GATE", "0") == "1"
paths = {
    "rebuild": f"{od}/rebuild_{ts}.json",
    "prod_live_batch": f"{od}/prod_live_batch_{ts}.json",
    "contracts": f"{od}/contracts_{ts}.json",
    "hard_metrics": f"{od}/hard_metrics_{ts}.json",
    "parity": f"{od}/parity_{ts}.json",
    "alerts": f"{od}/alerts_{ts}.json",
    "readiness": f"{od}/readiness_{ts}.json",
}

reports = {}
for k, p in paths.items():
    try:
        with open(p, "r", encoding="utf-8") as f:
            reports[k] = json.load(f)
    except Exception as e:
        reports[k] = {"parse_error": str(e), "passed": False}

gate = {
    "strict_batch_completed": int((reports.get("prod_live_batch") or {}).get("completed", 0)) > 0,
    "strict_contract_passed": bool((reports.get("contracts") or {}).get("passed")),
    "hard_metrics_passed": bool((reports.get("hard_metrics") or {}).get("hard_passed") or (reports.get("hard_metrics") or {}).get("passed")),
    "parity_30d_passed": str((reports.get("parity") or {}).get("status", "")).lower() == "passed",
    "alerts_config_passed": bool((reports.get("alerts") or {}).get("passed")),
    "readiness_passed": bool((reports.get("readiness") or {}).get("ready_for_gpu_cutover")),
}
gate["all_passed"] = all(gate.values())

bundle = {
    "files": paths,
    "gate": gate,
    "reports": reports,
}
print(json.dumps(bundle, ensure_ascii=False))
if enforce and not gate["all_passed"]:
    raise SystemExit(2)
PY
