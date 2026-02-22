#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="HPO for sequence model representation learning")
    ap.add_argument("--trials", type=int, default=int(os.getenv("HPO_TRIALS", "10")))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out-json", default="artifacts/models/hpo_best_trial.json")
    args = ap.parse_args()

    try:
        import optuna  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"optuna_missing:{exc}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def objective(trial):
        lookback = trial.suggest_categorical("lookback", [64, 96, 128])
        d_model = trial.suggest_categorical("d_model", [96, 128, 192])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.05, 0.3)
        lr = trial.suggest_float("lr", 5e-5, 3e-3, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 5e-3, log=True)

        model_id = f"liquid_hpo_trial_{trial.number}"
        out_dir = f"artifacts/models/{model_id}"
        cmd = [
            str(args.python),
            "training/train_liquid.py",
            "--lookback",
            str(lookback),
            "--d-model",
            str(d_model),
            "--n-layers",
            str(n_layers),
            "--dropout",
            str(dropout),
            "--lr",
            str(lr),
            "--weight-decay",
            str(wd),
            "--model-id",
            model_id,
            "--out-dir",
            out_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"trial_failed:{proc.stderr[-400:]}")
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        payload = json.loads(lines[-1])
        metrics = (((payload.get("manifest") or {}).get("metrics_summary") or {}).get("oos") or {})
        # Maximize sharpe_proxy and hit_rate, penalize mse.
        score = float(metrics.get("sharpe_proxy", 0.0)) + 0.5 * float(metrics.get("hit_rate", 0.0)) - 0.2 * float(metrics.get("mse", 0.0))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("model_id", model_id)
        return score

    study = optuna.create_study(direction="maximize", study_name="liquid_sequence_hpo")
    study.optimize(objective, n_trials=max(1, int(args.trials)))

    best = {
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "best_trial": int(study.best_trial.number),
        "best_model_id": str(study.best_trial.user_attrs.get("model_id") or ""),
        "best_metrics": dict(study.best_trial.user_attrs.get("metrics") or {}),
    }
    out_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", **best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
