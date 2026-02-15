from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Dict

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_pipeline import FeaturePipeline

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
MODEL_DIR = "/app/models"
GRAD_ACC_STEPS = int(os.getenv("TRAIN_GRAD_ACC_STEPS", "4"))
EPOCHS = int(os.getenv("VC_EPOCHS", "12"))
BATCH_SIZE = int(os.getenv("VC_BATCH_SIZE", "128"))
SEED = int(os.getenv("TRAIN_SEED", "42"))
VAL_RATIO = float(os.getenv("VC_VAL_RATIO", "0.2"))
PATIENCE = int(os.getenv("VC_EARLY_STOP_PATIENCE", "3"))
CHECKPOINT_PATH = os.getenv("VC_CHECKPOINT_PATH", os.path.join(MODEL_DIR, "vc_checkpoint.pt"))


class TinyVCModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class VCModelTrainer:
    def __init__(self, pipeline: FeaturePipeline, db_url: str = DATABASE_URL):
        self.pipeline = pipeline
        self.db_url = db_url
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _connect(self):
        return psycopg2.connect(self.db_url, cursor_factory=RealDictCursor)

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def train(self) -> Dict:
        self._set_seed(SEED)
        dq = self.pipeline.check_data_quality("BTC", timeframe="5m", lookback_hours=48)
        if dq.get("quality_passed", 0.0) < 0.5:
            return {"status": "blocked_by_data_quality", "data_quality": dq}

        batch = self.pipeline.load_vc_training_batch(limit=3000)
        if batch.X.shape[0] == 0:
            return {"status": "no_data", "samples": 0}

        # Closed-form linear baseline as bootstrap model.
        X = batch.X
        y = batch.y
        w = np.linalg.pinv(X.T @ X + np.eye(X.shape[1]) * 1e-3) @ X.T @ y

        teacher_logits = X @ w
        teacher_probs = 1.0 / (1.0 + np.exp(-teacher_logits))
        n = X.shape[0]
        val_n = max(32, int(n * VAL_RATIO))
        if val_n >= n:
            val_n = max(1, n // 4)
        train_idx = np.arange(0, n - val_n)
        val_idx = np.arange(n - val_n, n)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyVCModel(in_dim=X.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        y_train = y[train_idx]
        pos = float(np.sum(y_train > 0.5))
        neg = float(max(1.0, y_train.shape[0] - pos))
        pos_weight = torch.tensor([neg / max(1.0, pos)], dtype=torch.float32, device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mse = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

        xt = torch.tensor(X, dtype=torch.float32)
        x_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y[train_idx], dtype=torch.float32)
        t_train = torch.tensor(teacher_probs[train_idx], dtype=torch.float32)
        x_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val_t = torch.tensor(y[val_idx], dtype=torch.float32)
        t_val = torch.tensor(teacher_probs[val_idx], dtype=torch.float32)
        train_ds = TensorDataset(x_train, y_train_t, t_train)
        val_ds = TensorDataset(x_val, y_val_t, t_val)
        train_bs = BATCH_SIZE

        start_epoch = 0
        best_val_loss = float("inf")
        bad_epochs = 0
        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))

        for epoch in range(start_epoch, EPOCHS):
            train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
            val_dl = DataLoader(val_ds, batch_size=train_bs, shuffle=False, drop_last=False)
            model.train()
            opt.zero_grad(set_to_none=True)
            train_loss_sum = 0.0
            try:
                for step, (bx, by, bt) in enumerate(train_dl, start=1):
                    bx = bx.to(device)
                    by = by.to(device)
                    bt = bt.to(device)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        logits = model(bx)
                        sup_loss = bce(logits, by)
                        distill_loss = mse(torch.sigmoid(logits), bt)
                        loss = 0.8 * sup_loss + 0.2 * distill_loss
                        if torch.isnan(loss):
                            raise RuntimeError("nan_loss_detected")
                        loss = loss / max(1, GRAD_ACC_STEPS)
                    scaler.scale(loss).backward()
                    train_loss_sum += float(loss.item())
                    if step % max(1, GRAD_ACC_STEPS) == 0 or step == len(train_dl):
                        scaler.step(opt)
                        scaler.update()
                        opt.zero_grad(set_to_none=True)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and train_bs > 16:
                    train_bs = max(16, train_bs // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            model.eval()
            with torch.no_grad():
                val_losses = []
                for bx, by, bt in val_dl:
                    bx = bx.to(device)
                    by = by.to(device)
                    bt = bt.to(device)
                    logits = model(bx)
                    v = 0.8 * bce(logits, by) + 0.2 * mse(torch.sigmoid(logits), bt)
                    val_losses.append(float(v.item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                    },
                    CHECKPOINT_PATH,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break

        if os.path.exists(CHECKPOINT_PATH):
            ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        model.eval()
        with torch.no_grad():
            full_x = x_val.to(device)
            pred_logits = model(full_x).detach().cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-pred_logits))
            preds = (probs >= 0.5).astype(np.float32)
            acc = float((preds == y[val_idx]).mean())
            distill_gap = float(np.mean(np.abs(probs - teacher_probs[val_idx])))

        model_path = os.path.join(MODEL_DIR, "vc_survival_baseline_v2.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump({"weights": w.tolist(), "trained_at": datetime.utcnow().isoformat()}, f)
        nn_model_path = os.path.join(MODEL_DIR, "vc_survival_ttm_v2.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": int(X.shape[1]),
                "trained_at": datetime.utcnow().isoformat(),
                "type": "tiny_vc_mlp",
            },
            nn_model_path,
        )
        train_manifest = {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_acc_steps": GRAD_ACC_STEPS,
            "val_ratio": VAL_RATIO,
            "early_stop_patience": PATIENCE,
            "checkpoint_path": CHECKPOINT_PATH,
            "samples": int(X.shape[0]),
            "val_samples": int(val_idx.shape[0]),
            "feature_dim": int(X.shape[1]),
            "best_val_loss": round(best_val_loss, 8),
            "val_accuracy": round(acc, 6),
            "distill_gap": round(distill_gap, 6),
            "created_at": datetime.utcnow().isoformat(),
        }
        manifest_path = os.path.join(MODEL_DIR, "vc_train_manifest_v2.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(train_manifest, f)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO model_registry (
                        model_name, track, model_version, artifact_path, metrics, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (model_name, track, model_version)
                    DO UPDATE SET
                        artifact_path = EXCLUDED.artifact_path,
                        metrics = EXCLUDED.metrics,
                        created_at = NOW()
                    """,
                    (
                        "vc_survival_baseline",
                        "vc",
                        "v2.0",
                        model_path,
                        json.dumps({"val_accuracy": round(acc, 6), "samples": int(X.shape[0]), "seed": SEED, "data_quality": dq}),
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO model_registry (
                        model_name, track, model_version, artifact_path, metrics, created_at
                    ) VALUES (%s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (model_name, track, model_version)
                    DO UPDATE SET
                        artifact_path = EXCLUDED.artifact_path,
                        metrics = EXCLUDED.metrics,
                        created_at = NOW()
                    """,
                    (
                        "vc_survival_ttm",
                        "vc",
                        "v2.1",
                        nn_model_path,
                        json.dumps(
                            {
                                "train_accuracy": round(acc, 6),
                                "distill_gap": round(distill_gap, 6),
                                "samples": int(X.shape[0]),
                                "grad_acc_steps": GRAD_ACC_STEPS,
                                "seed": SEED,
                                "best_val_loss": round(best_val_loss, 8),
                                "train_manifest_path": manifest_path,
                                "data_quality": dq,
                            }
                        ),
                    ),
                )

        return {
            "status": "ok",
            "samples": int(X.shape[0]),
            "val_accuracy": round(acc, 6),
            "artifact": model_path,
            "distilled_artifact": nn_model_path,
            "distill_gap": round(distill_gap, 6),
            "data_quality": dq,
        }
