from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_pipeline import FeaturePipeline

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")
MODEL_DIR = "/app/models"
GRAD_ACC_STEPS = int(os.getenv("TRAIN_GRAD_ACC_STEPS", "4"))
EPOCHS = int(os.getenv("LIQUID_EPOCHS", "14"))
BATCH_SIZE = int(os.getenv("LIQUID_BATCH_SIZE", "128"))
SEED = int(os.getenv("TRAIN_SEED", "42"))
VAL_RATIO = float(os.getenv("LIQUID_VAL_RATIO", "0.2"))
PATIENCE = int(os.getenv("LIQUID_EARLY_STOP_PATIENCE", "3"))
GRAD_CLIP = float(os.getenv("TRAIN_GRAD_CLIP", "1.0"))


class TinyLiquidModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LiquidModelTrainer:
    def __init__(self, pipeline: FeaturePipeline, symbols: List[str], db_url: str = DATABASE_URL):
        self.pipeline = pipeline
        self.symbols = symbols
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

    def train_symbol(self, symbol: str) -> Dict:
        self._set_seed(SEED)
        dq = self.pipeline.check_data_quality(symbol, timeframe="5m", lookback_hours=72)
        if dq.get("quality_passed", 0.0) < 0.5:
            return {"symbol": symbol, "status": "blocked_by_data_quality", "data_quality": dq}

        batch = self.pipeline.load_liquid_training_batch(symbol=symbol, limit=4000)
        if batch.X.shape[0] == 0:
            return {"symbol": symbol, "status": "no_data"}

        X, y = batch.X, batch.y
        n = X.shape[0]
        val_n = max(64, int(n * VAL_RATIO))
        if val_n >= n:
            val_n = max(1, n // 4)
        train_idx = np.arange(0, n - val_n)
        val_idx = np.arange(n - val_n, n)

        x_mean = np.mean(X[train_idx], axis=0, keepdims=True)
        x_std = np.clip(np.std(X[train_idx], axis=0, keepdims=True), 1e-6, None)
        Xn = (X - x_mean) / x_std

        y_train = y[train_idx]
        y_val = y[val_idx]
        w = np.linalg.pinv(Xn[train_idx].T @ Xn[train_idx] + np.eye(X.shape[1]) * 1e-3) @ Xn[train_idx].T @ y_train
        y_hat = Xn @ w
        mse = float(np.mean((y_hat - y) ** 2))
        teacher = y_hat.copy()
        teacher_name = "ridge"
        if HAS_LGB:
            try:
                reg = lgb.LGBMRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=SEED,
                )
                reg.fit(Xn[train_idx], y_train)
                lgb_pred = reg.predict(Xn)
                teacher = 0.6 * lgb_pred + 0.4 * y_hat
                teacher_name = "lightgbm+ridge"
            except Exception:
                teacher_name = "ridge_fallback"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TinyLiquidModel(in_dim=Xn.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

        xt = torch.tensor(Xn, dtype=torch.float32)
        x_train = torch.tensor(Xn[train_idx], dtype=torch.float32)
        x_val = torch.tensor(Xn[val_idx], dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        tt = torch.tensor(teacher, dtype=torch.float32)
        train_ds = TensorDataset(x_train, y_train_t, tt[train_idx])
        val_ds = TensorDataset(x_val, y_val_t, tt[val_idx])
        train_bs = BATCH_SIZE
        ckpt_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_checkpoint.pt")
        best_val = float("inf")
        best_epoch = 0
        bad_epochs = 0
        start_epoch = 0
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_val = float(ckpt.get("best_val", best_val))

        for epoch in range(start_epoch, EPOCHS):
            train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=False, drop_last=False)
            val_dl = DataLoader(val_ds, batch_size=train_bs, shuffle=False, drop_last=False)
            model.train()
            opt.zero_grad(set_to_none=True)
            try:
                for step, (bx, by, bt) in enumerate(train_dl, start=1):
                    bx = bx.to(device)
                    by = by.to(device)
                    bt = bt.to(device)
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        pred = model(bx)
                        sup = mse_loss(pred, by)
                        distill = mse_loss(pred, bt)
                        loss = (0.85 * sup + 0.15 * distill) / max(1, GRAD_ACC_STEPS)
                        if torch.isnan(loss):
                            raise RuntimeError("nan_loss_detected")
                    scaler.scale(loss).backward()
                    if step % max(1, GRAD_ACC_STEPS) == 0 or step == len(train_dl):
                        if GRAD_CLIP > 0:
                            scaler.unscale_(opt)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
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
                    pred = model(bx)
                    lv = 0.85 * mse_loss(pred, by) + 0.15 * mse_loss(pred, bt)
                    val_losses.append(float(lv.item()))
                val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
                scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val": best_val,
                        "model_state": model.state_dict(),
                        "optimizer_state": opt.state_dict(),
                    },
                    ckpt_path,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= PATIENCE:
                    break

        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

        model.eval()
        with torch.no_grad():
            pred_nn = model(x_val.to(device)).detach().cpu().numpy()
            nn_mse = float(np.mean((pred_nn - y_val) ** 2))
            distill_gap = float(np.mean(np.abs(pred_nn - teacher[val_idx])))
            directional_hit = float(np.mean(np.sign(pred_nn) == np.sign(y_val)))

        model_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_baseline_v2.json")
        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "weights": w.tolist(),
                    "x_mean": x_mean.flatten().tolist(),
                    "x_std": x_std.flatten().tolist(),
                    "trained_at": datetime.utcnow().isoformat(),
                    "seed": SEED,
                },
                f,
            )
        nn_model_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_ttm_v2.pt")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "in_dim": int(Xn.shape[1]),
                "trained_at": datetime.utcnow().isoformat(),
                "type": "tiny_liquid_mlp",
                "seed": SEED,
                "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
            },
            nn_model_path,
        )

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
                        f"liquid_baseline_{symbol.lower()}",
                        "liquid",
                        "v2.0",
                        model_path,
                        json.dumps({"train_mse": round(mse, 9), "samples": int(X.shape[0]), "teacher": teacher_name, "seed": SEED, "data_quality": dq}),
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
                        f"liquid_ttm_{symbol.lower()}",
                        "liquid",
                        "v2.1",
                        nn_model_path,
                        json.dumps(
                            {
                                "train_mse": round(nn_mse, 9),
                                "teacher_mse": round(mse, 9),
                                "distill_gap": round(distill_gap, 9),
                                "val_directional_hit_rate": round(directional_hit, 9),
                                "samples": int(X.shape[0]),
                                "grad_acc_steps": GRAD_ACC_STEPS,
                                "best_val_loss": round(best_val, 9),
                                "best_epoch": int(best_epoch),
                                "teacher": teacher_name,
                                "seed": SEED,
                                "data_quality": dq,
                            }
                        ),
                    ),
                )

        return {
            "symbol": symbol,
            "status": "ok",
            "samples": int(X.shape[0]),
            "train_mse": round(mse, 9),
            "nn_mse": round(nn_mse, 9),
            "val_directional_hit_rate": round(directional_hit, 9),
            "distilled_artifact": nn_model_path,
            "teacher": teacher_name,
            "data_quality": dq,
        }

    def train_all(self) -> Dict:
        results = [self.train_symbol(symbol) for symbol in self.symbols]
        return {"status": "ok", "results": results}
