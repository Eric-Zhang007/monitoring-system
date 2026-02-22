#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch

from artifacts.pack import pack_model_artifact
from features.feature_contract import SCHEMA_HASH
from training.calibration.calibrate import fit_temperature_scaling
from vc.feature_spec import VC_FEATURE_KEYS, label_from_event_row, vector_from_context, vector_from_event_row


class VCModel(torch.nn.Module):
    def __init__(self, in_dim: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.net(x)


def _build_dataset(db_url: str, limit: int = 8000):
    X: List[List[float]] = []
    y: List[int] = []
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_type, source_tier, confidence_score, event_importance, novelty_score
                FROM events
                ORDER BY COALESCE(available_at, occurred_at) DESC
                LIMIT %s
                """,
                (max(100, int(limit)),),
            )
            rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        X.append(vector_from_event_row(r).tolist())
        y.append(label_from_event_row(r))
    if len(X) < 128:
        raise RuntimeError(f"insufficient_vc_samples:{len(X)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train VC model with strict artifact contract")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out-dir", default=os.getenv("VC_MODEL_DIR", "artifacts/models/vc_main"))
    ap.add_argument("--model-id", default="vc_main")
    args = ap.parse_args()

    X, y = _build_dataset(args.database_url, args.limit)
    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    model = VCModel(in_dim=X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    tx = torch.tensor(Xtr, dtype=torch.float32)
    ty = torch.tensor(ytr, dtype=torch.long)

    for _ in range(max(1, int(args.epochs))):
        logits = model(tx)
        loss = loss_fn(logits, ty)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits_te = model(torch.tensor(Xte, dtype=torch.float32)).cpu().numpy()
        prob_te = torch.softmax(torch.tensor(logits_te), dim=-1)[:, 1].numpy()
        pred_te = (prob_te >= 0.5).astype(np.int64)
    acc = float(np.mean((pred_te == yte).astype(np.float32)))
    temperature = float(fit_temperature_scaling(logits_te[:, 1], yte.astype(np.float64)))

    state = {
        "state_dict": model.state_dict(),
        "in_dim": int(X.shape[1]),
        "schema_hash": SCHEMA_HASH,
        "feature_keys": list(VC_FEATURE_KEYS),
        "temperature": temperature,
    }

    manifest = pack_model_artifact(
        model_dir=Path(args.out_dir),
        model_id=str(args.model_id),
        state_dict=state,
        schema_path="schema/liquid_feature_schema.yaml",
        lookback=1,
        feature_dim=int(X.shape[1]),
        data_version="main",
        metrics_summary={"oos_acc": float(acc)},
        extra_payload={
            "train_samples": int(len(Xtr)),
            "oos_samples": int(len(Xte)),
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "feature_keys": list(VC_FEATURE_KEYS),
            "temperature": temperature,
        },
    )
    print(json.dumps({"status": "ok", "manifest": manifest, "oos_acc": acc}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
