#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch

from mlops_artifacts.pack import pack_model_artifact
from training.calibration.calibrate import fit_temperature_scaling
from vc.feature_spec import VC_FEATURE_KEYS, VC_SCHEMA_HASH, VC_SCHEMA_PATH, label_from_event_row, vector_from_event_row


VC_HORIZONS = ("6m", "12m", "24m")


class VCModel(torch.nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 64),
            torch.nn.GELU(),
            torch.nn.Linear(64, 32),
            torch.nn.GELU(),
        )
        self.round_logits = torch.nn.Linear(32, len(VC_HORIZONS))
        self.exit_logit = torch.nn.Linear(32, 1)
        self.moic_mu = torch.nn.Linear(32, 1)
        self.moic_log_sigma = torch.nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {
            "round_logits": self.round_logits(h),
            "exit_logit": self.exit_logit(h).squeeze(-1),
            "moic_mu": self.moic_mu(h).squeeze(-1),
            "moic_log_sigma": self.moic_log_sigma(h).squeeze(-1).clamp(min=-4.0, max=2.0),
        }


def _targets_from_event_row(row: Dict[str, object]) -> Tuple[np.ndarray, float, float]:
    event_type = str(row.get("event_type") or "").strip().lower()
    conf = float(row.get("confidence_score") or 0.0)
    imp = float(row.get("event_importance") or 0.0)
    nov = float(row.get("novelty_score") or 0.0)

    y12 = float(label_from_event_row(row))
    y6 = 1.0 if event_type == "funding" else 0.0
    y24 = 1.0 if (y12 > 0 or event_type == "product" or conf >= 0.8) else 0.0
    y_exit = 1.0 - y24

    moic_target = 0.8 + 1.0 * y6 + 1.2 * y12 + 0.8 * y24 + 0.4 * conf + 0.3 * imp + 0.2 * nov
    moic_target = float(np.clip(moic_target, 0.2, 8.0))
    return np.array([y6, y12, y24], dtype=np.float32), float(y_exit), moic_target


def _build_dataset(db_url: str, limit: int = 8000):
    X: List[np.ndarray] = []
    y_round: List[np.ndarray] = []
    y_exit: List[float] = []
    y_moic: List[float] = []
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_type, source_tier, confidence_score, event_importance, novelty_score
                FROM events
                ORDER BY COALESCE(available_at, occurred_at) DESC
                LIMIT %s
                """,
                (max(200, int(limit)),),
            )
            rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        X.append(vector_from_event_row(r))
        y_r, y_e, y_m = _targets_from_event_row(r)
        y_round.append(y_r)
        y_exit.append(y_e)
        y_moic.append(y_m)
    if len(X) < 128:
        raise RuntimeError(f"insufficient_vc_samples:{len(X)}")
    return (
        np.stack(X, axis=0).astype(np.float32),
        np.stack(y_round, axis=0).astype(np.float32),
        np.array(y_exit, dtype=np.float32),
        np.array(y_moic, dtype=np.float32),
    )


def _gaussian_nll(mu: torch.Tensor, log_sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sigma = torch.exp(log_sigma).clamp(min=1e-4)
    z = (y - mu) / sigma
    return 0.5 * (z**2 + 2.0 * log_sigma)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train strict VC model and package strict artifact")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out-dir", default=os.getenv("VC_MODEL_DIR", "artifacts/models/vc_main"))
    ap.add_argument("--model-id", default="vc_main")
    args = ap.parse_args()

    x, y_round, y_exit, y_moic = _build_dataset(args.database_url, args.limit)
    split = int(len(x) * 0.8)
    x_tr, x_te = x[:split], x[split:]
    y_round_tr, y_round_te = y_round[:split], y_round[split:]
    y_exit_tr, y_exit_te = y_exit[:split], y_exit[split:]
    y_moic_tr, y_moic_te = y_moic[:split], y_moic[split:]

    model = VCModel(in_dim=x.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = torch.nn.BCEWithLogitsLoss()

    tx = torch.tensor(x_tr, dtype=torch.float32)
    ty_round = torch.tensor(y_round_tr, dtype=torch.float32)
    ty_exit = torch.tensor(y_exit_tr, dtype=torch.float32)
    ty_moic = torch.tensor(y_moic_tr, dtype=torch.float32)

    train_trace: List[Dict[str, float]] = []
    for _ in range(max(1, int(args.epochs))):
        out = model(tx)
        loss_round = bce(out["round_logits"], ty_round)
        loss_exit = bce(out["exit_logit"], ty_exit)
        loss_moic = _gaussian_nll(out["moic_mu"], out["moic_log_sigma"], ty_moic).mean()
        total = loss_round + 0.6 * loss_exit + 0.2 * loss_moic
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        train_trace.append(
            {
                "loss_total": float(total.detach().cpu().item()),
                "loss_round": float(loss_round.detach().cpu().item()),
                "loss_exit": float(loss_exit.detach().cpu().item()),
                "loss_moic": float(loss_moic.detach().cpu().item()),
            }
        )

    with torch.no_grad():
        out_te = model(torch.tensor(x_te, dtype=torch.float32))
        logits_round_te = out_te["round_logits"].cpu().numpy()
        logits_exit_te = out_te["exit_logit"].cpu().numpy()
        moic_mu_te = out_te["moic_mu"].cpu().numpy()
        moic_sigma_te = np.exp(out_te["moic_log_sigma"].cpu().numpy())

    prob_12 = 1.0 / (1.0 + np.exp(-np.clip(logits_round_te[:, 1], -40.0, 40.0)))
    pred_12 = (prob_12 >= 0.5).astype(np.float32)
    acc_12 = float(np.mean(pred_12 == y_round_te[:, 1]))
    brier_12 = float(np.mean((prob_12 - y_round_te[:, 1]) ** 2))
    moic_mae = float(np.mean(np.abs(moic_mu_te - y_moic_te)))

    round_temperature = {
        VC_HORIZONS[i]: float(fit_temperature_scaling(logits_round_te[:, i], y_round_te[:, i]))
        for i in range(len(VC_HORIZONS))
    }
    exit_temperature = float(fit_temperature_scaling(logits_exit_te, y_exit_te))
    sigma_scale = float(np.sqrt(np.mean(((y_moic_te - moic_mu_te) / np.clip(moic_sigma_te, 1e-6, None)) ** 2)))

    state = {
        "state_dict": model.state_dict(),
        "in_dim": int(x.shape[1]),
        "schema_hash": VC_SCHEMA_HASH,
        "feature_keys": list(VC_FEATURE_KEYS),
        "horizons": list(VC_HORIZONS),
        "round_temperature": round_temperature,
        "exit_temperature": exit_temperature,
        "moic_sigma_scale": sigma_scale,
    }

    training_report = {
        "status": "ok",
        "train_samples": int(len(x_tr)),
        "oos_samples": int(len(x_te)),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "feature_keys": list(VC_FEATURE_KEYS),
        "horizons": list(VC_HORIZONS),
        "metrics": {
            "acc_12m": acc_12,
            "brier_12m": brier_12,
            "moic_mae": moic_mae,
        },
        "calibration": {
            "round_temperature": round_temperature,
            "exit_temperature": exit_temperature,
            "moic_sigma_scale": sigma_scale,
        },
        "train_trace_tail": train_trace[-10:],
    }

    manifest = pack_model_artifact(
        model_dir=Path(args.out_dir),
        model_id=str(args.model_id),
        state_dict=state,
        schema_path=str(VC_SCHEMA_PATH),
        lookback=1,
        feature_dim=int(x.shape[1]),
        data_version="main",
        metrics_summary=training_report["metrics"],
        extra_payload=training_report,
    )
    print(json.dumps({"status": "ok", "manifest": manifest, "training_report": training_report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
