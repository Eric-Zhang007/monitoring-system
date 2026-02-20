#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[ERROR] python executable not found: $PYTHON_BIN"
  exit 2
fi

DATABASE_URL="${DATABASE_URL:-}"
if [[ -z "$DATABASE_URL" ]]; then
  echo "[ERROR] DATABASE_URL is required"
  exit 2
fi
if [[ "$DATABASE_URL" == *"change_me_please"* || "$DATABASE_URL" == *"REPLACE_WITH_"* ]]; then
  echo "[ERROR] DATABASE_URL still uses placeholder secret"
  exit 2
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-artifacts/experiments/phase_d/${TS}}"
mkdir -p "$OUT_DIR"

PHASE_D_START="${PHASE_D_START:-2018-01-01T00:00:00Z}"
PHASE_D_END="${PHASE_D_END:-}"
PHASE_D_SYMBOLS="${PHASE_D_SYMBOLS:-${LIQUID_SYMBOLS:-BTC,ETH,SOL}}"
PHASE_D_HORIZON_STEPS="${PHASE_D_HORIZON_STEPS:-${MULTIMODAL_TARGET_HORIZON_STEPS:-1}}"

PHASE_D_RUN_TRAIN="${PHASE_D_RUN_TRAIN:-1}"
PHASE_D_RUN_BACKBONE="${PHASE_D_RUN_BACKBONE:-1}"
PHASE_D_RUN_ABLATION="${PHASE_D_RUN_ABLATION:-1}"
PHASE_D_RUN_REGISTER="${PHASE_D_RUN_REGISTER:-1}"

CANDIDATE_OUT="${CANDIDATE_OUT:-$OUT_DIR/multimodal_candidate.json}"
BACKBONE_OUT="${BACKBONE_OUT:-$OUT_DIR/backbone_suite.json}"
EVAL_OUT="${EVAL_OUT:-$OUT_DIR/multimodal_eval.json}"
REGISTRY_OUT="${REGISTRY_OUT:-$OUT_DIR/candidate_registry.jsonl}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-$OUT_DIR/phase_d_unified_summary.json}"
SUMMARY_MD_OUT="${SUMMARY_MD_OUT:-$OUT_DIR/phase_d_unified_summary.md}"

BACKBONE_EXP_BACKBONES="${BACKBONE_EXP_BACKBONES:-ridge,itransformer,patchtst,tft}"
BACKBONE_EXP_LOOKBACK_STEPS="${BACKBONE_EXP_LOOKBACK_STEPS:-32}"
BACKBONE_EXP_MAX_SAMPLES="${BACKBONE_EXP_MAX_SAMPLES:-30000}"
BACKBONE_EXP_EPOCHS="${BACKBONE_EXP_EPOCHS:-6}"
BACKBONE_EXP_BATCH_SIZE="${BACKBONE_EXP_BATCH_SIZE:-256}"
BACKBONE_EXP_LR="${BACKBONE_EXP_LR:-0.001}"

MULTIMODAL_ABLATIONS="${MULTIMODAL_ABLATIONS:-full,no_text,no_macro,event_window}"
MULTIMODAL_EVENT_WINDOW_THRESHOLD="${MULTIMODAL_EVENT_WINDOW_THRESHOLD:-0.05}"
MULTIMODAL_L2="${MULTIMODAL_L2:-0.05}"

CANDIDATE_ENFORCE_GATES="${CANDIDATE_ENFORCE_GATES:-0}"
CANDIDATE_MIN_OOS_HIT_RATE="${CANDIDATE_MIN_OOS_HIT_RATE:-0.5}"
CANDIDATE_MAX_OOS_MSE="${CANDIDATE_MAX_OOS_MSE:-1.0}"
CANDIDATE_MIN_BACKBONE_READY="${CANDIDATE_MIN_BACKBONE_READY:-0}"
CANDIDATE_REQUIRED_BACKBONES="${CANDIDATE_REQUIRED_BACKBONES:-}"
CANDIDATE_MAX_DELTA_MSE_NO_TEXT_VS_FULL="${CANDIDATE_MAX_DELTA_MSE_NO_TEXT_VS_FULL:-1.0}"
CANDIDATE_MAX_DELTA_MSE_NO_MACRO_VS_FULL="${CANDIDATE_MAX_DELTA_MSE_NO_MACRO_VS_FULL:-1.0}"

echo "[phase-d] out_dir=$OUT_DIR"
echo "[phase-d] python=$PYTHON_BIN"
echo "[phase-d] symbols=$PHASE_D_SYMBOLS horizon_steps=$PHASE_D_HORIZON_STEPS"

if [[ "$PHASE_D_RUN_TRAIN" == "1" ]]; then
  echo "[1/4] train multimodal candidate"
  "$PYTHON_BIN" training/train_multimodal.py \
    --database-url "$DATABASE_URL" \
    --start "$PHASE_D_START" \
    ${PHASE_D_END:+--end "$PHASE_D_END"} \
    --symbols "$PHASE_D_SYMBOLS" \
    --horizon-steps "$PHASE_D_HORIZON_STEPS" \
    --l2 "$MULTIMODAL_L2" \
    --out "$CANDIDATE_OUT"
else
  echo "[1/4] train skipped (PHASE_D_RUN_TRAIN=$PHASE_D_RUN_TRAIN)"
fi

if [[ "$PHASE_D_RUN_BACKBONE" == "1" ]]; then
  echo "[2/4] run backbone experiments"
  "$PYTHON_BIN" training/backbone_experiments.py \
    --database-url "$DATABASE_URL" \
    --start "$PHASE_D_START" \
    ${PHASE_D_END:+--end "$PHASE_D_END"} \
    --symbols "$PHASE_D_SYMBOLS" \
    --horizon-steps "$PHASE_D_HORIZON_STEPS" \
    --lookback-steps "$BACKBONE_EXP_LOOKBACK_STEPS" \
    --max-samples "$BACKBONE_EXP_MAX_SAMPLES" \
    --backbones "$BACKBONE_EXP_BACKBONES" \
    --epochs "$BACKBONE_EXP_EPOCHS" \
    --batch-size "$BACKBONE_EXP_BATCH_SIZE" \
    --lr "$BACKBONE_EXP_LR" \
    --l2 "$MULTIMODAL_L2" \
    --out "$BACKBONE_OUT"
else
  echo "[2/4] backbone experiments skipped (PHASE_D_RUN_BACKBONE=$PHASE_D_RUN_BACKBONE)"
fi

if [[ "$PHASE_D_RUN_ABLATION" == "1" ]]; then
  echo "[3/4] run oos + ablations"
  "$PYTHON_BIN" training/eval_multimodal_oos.py \
    --database-url "$DATABASE_URL" \
    --start "$PHASE_D_START" \
    ${PHASE_D_END:+--end "$PHASE_D_END"} \
    --symbols "$PHASE_D_SYMBOLS" \
    --horizon-steps "$PHASE_D_HORIZON_STEPS" \
    --l2 "$MULTIMODAL_L2" \
    --ablations "$MULTIMODAL_ABLATIONS" \
    --event-strength-threshold "$MULTIMODAL_EVENT_WINDOW_THRESHOLD" \
    --out "$EVAL_OUT"
else
  echo "[3/4] ablation eval skipped (PHASE_D_RUN_ABLATION=$PHASE_D_RUN_ABLATION)"
fi

if [[ "$PHASE_D_RUN_REGISTER" == "1" ]]; then
  echo "[4/4] register candidate (gate configurable)"
  set +e
  "$PYTHON_BIN" training/register_candidate_model.py \
    --database-url "$DATABASE_URL" \
    --candidate "$CANDIDATE_OUT" \
    --eval "$EVAL_OUT" \
    --backbone-report "$BACKBONE_OUT" \
    --registry "$REGISTRY_OUT" \
    --enforce-gates "$CANDIDATE_ENFORCE_GATES" \
    --min-oos-hit-rate "$CANDIDATE_MIN_OOS_HIT_RATE" \
    --max-oos-mse "$CANDIDATE_MAX_OOS_MSE" \
    --min-backbone-ready "$CANDIDATE_MIN_BACKBONE_READY" \
    --required-backbones "$CANDIDATE_REQUIRED_BACKBONES" \
    --max-delta-mse-no-text-vs-full "$CANDIDATE_MAX_DELTA_MSE_NO_TEXT_VS_FULL" \
    --max-delta-mse-no-macro-vs-full "$CANDIDATE_MAX_DELTA_MSE_NO_MACRO_VS_FULL"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[WARN] register returned non-zero rc=$rc"
    if [[ "$CANDIDATE_ENFORCE_GATES" == "1" ]]; then
      echo "[FAIL] candidate gate failed under enforce mode"
      exit $rc
    fi
  fi
else
  echo "[4/4] register skipped (PHASE_D_RUN_REGISTER=$PHASE_D_RUN_REGISTER)"
fi

"$PYTHON_BIN" - <<'PY' "$OUT_DIR" "$CANDIDATE_OUT" "$BACKBONE_OUT" "$EVAL_OUT" "$REGISTRY_OUT"
import json
import os
import sys
from datetime import datetime, timezone

out_dir, c_path, b_path, e_path, r_path = sys.argv[1:]

def load(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

candidate = load(c_path)
backbone = load(b_path)
eval_report = load(e_path)

summary = {
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "paths": {
        "candidate": c_path if os.path.exists(c_path) else "",
        "backbone": b_path if os.path.exists(b_path) else "",
        "eval": e_path if os.path.exists(e_path) else "",
        "registry": r_path if os.path.exists(r_path) else "",
    },
    "candidate": {
        "model_name": candidate.get("model_name"),
        "fusion_mode": candidate.get("fusion_mode"),
        "val_mse": candidate.get("val_mse"),
        "val_mae": candidate.get("val_mae"),
    },
    "backbone": {
        "ready_backbones": backbone.get("ready_backbones", []),
        "torch_available": backbone.get("torch_available", False),
    },
    "eval": {
        "mse": eval_report.get("mse"),
        "mae": eval_report.get("mae"),
        "hit_rate": eval_report.get("hit_rate"),
        "primary_ablation": eval_report.get("primary_ablation"),
        "ablation_count": len(eval_report.get("ablation_results", [])) if isinstance(eval_report.get("ablation_results"), list) else 0,
    },
}
os.makedirs(out_dir, exist_ok=True)
summary_path = os.path.join(out_dir, "phase_d_summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps({"status": "ok", "summary": summary_path}, ensure_ascii=False))
PY

echo "[phase-d] build unified comparison summary"
"$PYTHON_BIN" scripts/summarize_phase_d_results.py \
  --candidate "$CANDIDATE_OUT" \
  --backbone "$BACKBONE_OUT" \
  --eval "$EVAL_OUT" \
  --out-json "$SUMMARY_JSON_OUT" \
  --out-md "$SUMMARY_MD_OUT"

echo "[phase-d] done out_dir=$OUT_DIR"
