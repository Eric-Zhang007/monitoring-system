from __future__ import annotations

import json
import os
import random
import hashlib
from datetime import datetime
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from feature_pipeline import FeaturePipeline
from validation import evaluate_regression_oos, purged_kfold_slices, summarize_fold_metrics, walk_forward_slices

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
WF_TRAIN_WINDOW = int(os.getenv("LIQUID_WF_TRAIN_WINDOW", "512"))
WF_TEST_WINDOW = int(os.getenv("LIQUID_WF_TEST_WINDOW", "96"))
WF_PURGE_WINDOW = int(os.getenv("LIQUID_WF_PURGE_WINDOW", "12"))
WF_MIN_FOLDS = int(os.getenv("LIQUID_WF_MIN_FOLDS", "3"))
PKF_SPLITS = int(os.getenv("LIQUID_PURGED_KFOLD_SPLITS", "5"))
PKF_PURGE_WINDOW = int(os.getenv("LIQUID_PURGED_KFOLD_PURGE", "12"))
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "feature-store-v2.1")
DATA_VERSION = os.getenv("DATA_VERSION", "v1")


class MixerBlock(nn.Module):
    def __init__(self, n_tokens: int, n_channels: int, hidden_dim: int = 64):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(n_tokens, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_tokens),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(n_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_channels),
        )
        self.ln1 = nn.LayerNorm(n_channels)
        self.ln2 = nn.LayerNorm(n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, tokens, channels]
        y = self.ln1(x)
        y = y.transpose(1, 2)  # [batch, channels, tokens]
        y = self.token_mlp(y)
        y = y.transpose(1, 2)
        x = x + y
        z = self.ln2(x)
        z = self.channel_mlp(z)
        return x + z


class TSMixerLiquidModel(nn.Module):
    def __init__(self, n_tokens: int, n_channels: int, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([MixerBlock(n_tokens=n_tokens, n_channels=n_channels) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(n_channels),
            nn.Flatten(),
            nn.Linear(n_tokens * n_channels, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(-1)


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

    @staticmethod
    def _to_sequence(xn: np.ndarray) -> np.ndarray:
        if xn.shape[1] < 15:
            return xn[:, None, :]
        seq = np.stack(
            [
                np.stack([xn[:, 0], xn[:, 4], xn[:, 10]], axis=1),
                np.stack([xn[:, 1], xn[:, 5], xn[:, 9]], axis=1),
                np.stack([xn[:, 2], xn[:, 6], xn[:, 11]], axis=1),
                np.stack([xn[:, 3], xn[:, 7], xn[:, 12]], axis=1),
                np.stack([xn[:, 13], xn[:, 14], xn[:, 8]], axis=1),
            ],
            axis=1,
        )
        return seq.astype(np.float32)

    def _fit_lightgbm(self, x_train: np.ndarray, y_train: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, str, Dict]:
        if not HAS_LGB:
            w = np.linalg.pinv(x_train.T @ x_train + np.eye(x_train.shape[1]) * 1e-3) @ x_train.T @ y_train
            return (x_pred @ w).astype(np.float32), "ridge_fallback", {"model": "ridge_fallback", "weights": w.astype(np.float32).tolist()}
        try:
            reg = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=SEED,
            )
            reg.fit(x_train, y_train)
            booster = reg.booster_.model_to_string() if hasattr(reg, "booster_") and reg.booster_ is not None else ""
            return reg.predict(x_pred).astype(np.float32), "lightgbm", {"model": "lightgbm", "booster_model": booster}
        except Exception:
            w = np.linalg.pinv(x_train.T @ x_train + np.eye(x_train.shape[1]) * 1e-3) @ x_train.T @ y_train
            return (x_pred @ w).astype(np.float32), "ridge_fallback", {"model": "ridge_fallback", "weights": w.astype(np.float32).tolist()}

    def train_symbol(self, symbol: str) -> Dict:
        self._set_seed(SEED)
        dq = self.pipeline.check_data_quality(symbol, timeframe="5m", lookback_hours=72)
        dq_failed_reasons = []
        if float(dq.get("missing_rate", 0.0)) > float(os.getenv("DQ_MAX_MISSING_RATE", "0.02")):
            dq_failed_reasons.append("missing_rate_exceeded")
        if float(dq.get("invalid_price_rate", 0.0)) > float(os.getenv("DQ_MAX_INVALID_PRICE_RATE", "0.005")):
            dq_failed_reasons.append("invalid_price_rate_exceeded")
        if float(dq.get("duplicate_rate", 0.0)) > float(os.getenv("DQ_MAX_DUPLICATE_RATE", "0.02")):
            dq_failed_reasons.append("duplicate_rate_exceeded")
        if float(dq.get("stale_ratio", 0.0)) > float(os.getenv("DQ_MAX_STALE_RATIO", "0.1")):
            dq_failed_reasons.append("stale_ratio_exceeded")
        if dq.get("quality_passed", 0.0) < 0.5:
            return {
                "symbol": symbol,
                "status": "blocked_by_data_quality",
                "reason": ",".join(dq_failed_reasons) or "quality_gate_failed",
                "data_quality": dq,
            }

        batch = self.pipeline.load_liquid_training_batch(symbol=symbol, limit=4000)
        if batch.X.shape[0] == 0:
            return {"symbol": symbol, "status": "no_data"}

        X, y = batch.X, batch.y
        train_lineage_id = f"train-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{symbol.lower()}"
        feature_rows = [
            {
                "ret_1": float(r[0]),
                "ret_3": float(r[1]),
                "ret_12": float(r[2]),
                "ret_48": float(r[3]),
                "vol_3": float(r[4]),
                "vol_12": float(r[5]),
                "vol_48": float(r[6]),
                "vol_96": float(r[7]),
                "log_volume": float(r[8]),
                "vol_z": float(r[9]),
                "volume_impact": float(r[10]),
                "orderbook_imbalance": float(r[11]),
                "funding_rate": float(r[12]),
                "onchain_norm": float(r[13]),
                "event_decay": float(r[14]),
                "orderbook_missing_flag": float(r[15]) if len(r) > 15 else 1.0,
                "funding_missing_flag": float(r[16]) if len(r) > 16 else 1.0,
                "onchain_missing_flag": float(r[17]) if len(r) > 17 else 1.0,
                "feature_payload_schema_version": "v2.1",
            }
            for r in X
        ]
        self.pipeline.save_feature_snapshots_bulk(
            target=symbol.upper(),
            track="liquid",
            feature_rows=feature_rows,
            version=FEATURE_VERSION,
            lineage_id=train_lineage_id,
            data_version=DATA_VERSION,
        )
        label_1h = batch.extra_labels.get("fwd_ret_1h", y) if batch.extra_labels else y
        label_4h = batch.extra_labels.get("fwd_ret_4h", y) if batch.extra_labels else y
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
        lgb_all_pred, lgb_model_name, lgb_model_payload = self._fit_lightgbm(Xn[train_idx], y_train, Xn)
        lgb_mse = float(np.mean((lgb_all_pred[val_idx] - y_val) ** 2))
        wf_folds = walk_forward_slices(
            n_samples=n,
            train_window=WF_TRAIN_WINDOW,
            test_window=WF_TEST_WINDOW,
            purge_window=WF_PURGE_WINDOW,
        )
        wf_fold_metrics: List[Dict[str, float]] = []
        for fold_train_idx, fold_test_idx in wf_folds:
            if fold_train_idx.size < 64 or fold_test_idx.size < 16:
                continue
            fold_pred, _, _ = self._fit_lightgbm(Xn[fold_train_idx], y[fold_train_idx], Xn[fold_test_idx])
            wf_fold_metrics.append(
                evaluate_regression_oos(
                    y_true=y[fold_test_idx],
                    y_pred=fold_pred,
                    fee_bps=5.0,
                    slippage_bps=3.0,
                )
            )
        wf_metrics = summarize_fold_metrics(wf_fold_metrics)
        wf_ready = int(wf_metrics["folds"]) >= WF_MIN_FOLDS
        pkf_folds = purged_kfold_slices(n_samples=n, n_splits=PKF_SPLITS, purge_window=PKF_PURGE_WINDOW)
        pkf_metrics_per_fold: List[Dict[str, float]] = []
        for fold_train_idx, fold_test_idx in pkf_folds:
            if fold_train_idx.size < 64 or fold_test_idx.size < 16:
                continue
            fold_pred, _, _ = self._fit_lightgbm(Xn[fold_train_idx], y[fold_train_idx], Xn[fold_test_idx])
            pkf_metrics_per_fold.append(
                evaluate_regression_oos(
                    y_true=y[fold_test_idx],
                    y_pred=fold_pred,
                    fee_bps=5.0,
                    slippage_bps=3.0,
                )
            )
        pkf_metrics = summarize_fold_metrics(pkf_metrics_per_fold)
        teacher = lgb_all_pred.copy()
        teacher_name = lgb_model_name

        seq = self._to_sequence(Xn)
        n_tokens = int(seq.shape[1])
        n_channels = int(seq.shape[2])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TSMixerLiquidModel(n_tokens=n_tokens, n_channels=n_channels, n_blocks=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        mse_loss = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

        x_train = torch.tensor(seq[train_idx], dtype=torch.float32)
        x_val = torch.tensor(seq[val_idx], dtype=torch.float32)
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
            lgb_val_pred = lgb_all_pred[val_idx]
            denom = float(np.mean((pred_nn - lgb_val_pred) ** 2))
            if denom < 1e-12:
                ensemble_alpha = 0.5
            else:
                ensemble_alpha = float(np.mean((y_val - lgb_val_pred) * (pred_nn - lgb_val_pred)) / denom)
            ensemble_alpha = max(0.0, min(1.0, ensemble_alpha))
            ensemble_pred = ensemble_alpha * pred_nn + (1.0 - ensemble_alpha) * lgb_val_pred
            ensemble_metrics = evaluate_regression_oos(
                y_true=y_val,
                y_pred=ensemble_pred,
                fee_bps=5.0,
                slippage_bps=3.0,
            )

        model_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_lgbm_baseline_v2.json")
        model_name = f"liquid_baseline_{symbol.lower()}"
        model_version = "v2.0"
        created_at = datetime.utcnow().isoformat() + "Z"
        with open(model_path, "w", encoding="utf-8") as f:
            base_payload = {
                "model_name": model_name,
                "model_version": model_version,
                "track": "liquid",
                "type": "tabular_lightgbm" if lgb_model_name == "lightgbm" else "tabular_linear",
                "created_at": created_at,
                "feature_version": FEATURE_VERSION,
                "data_version": DATA_VERSION,
                "x_mean": x_mean.flatten().tolist(),
                "x_std": x_std.flatten().tolist(),
                "trained_at": created_at,
                "seed": SEED,
                "feature_dim": int(X.shape[1]),
            }
            base_payload.update(lgb_model_payload)
            json.dump(base_payload, f)
        nn_model_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_tsmixer_v2.pt")
        model_version_nn = "v2.1"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_tokens": n_tokens,
                "n_channels": n_channels,
                "trained_at": created_at,
                "type": "tsmixer_liquid",
                "seed": SEED,
                "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
                "ensemble_alpha": float(ensemble_alpha),
                "feature_payload_schema_version": "v2.1",
                "feature_version": FEATURE_VERSION,
                "data_version": DATA_VERSION,
                "train_report_hash": "",
            },
            nn_model_path,
        )

        train_manifest = {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "grad_acc_steps": GRAD_ACC_STEPS,
            "grad_clip": GRAD_CLIP,
            "val_ratio": VAL_RATIO,
            "early_stop_patience": PATIENCE,
            "wf_train_window": WF_TRAIN_WINDOW,
            "wf_test_window": WF_TEST_WINDOW,
            "wf_purge_window": WF_PURGE_WINDOW,
            "wf_min_folds": WF_MIN_FOLDS,
            "wf_ready": wf_ready,
            "wf_metrics": wf_metrics,
            "purged_kfold": pkf_metrics,
            "feature_dim": int(X.shape[1]),
            "samples": int(X.shape[0]),
            "created_at": created_at,
            "feature_version": FEATURE_VERSION,
            "data_version": DATA_VERSION,
            "feature_payload_schema_version": "v2.1",
        }
        train_report_hash = hashlib.sha256(
            json.dumps(train_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        train_manifest["train_report_hash"] = train_report_hash
        torch.save(
            {
                "state_dict": model.state_dict(),
                "n_tokens": n_tokens,
                "n_channels": n_channels,
                "trained_at": created_at,
                "type": "tsmixer_liquid",
                "seed": SEED,
                "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
                "ensemble_alpha": float(ensemble_alpha),
                "feature_payload_schema_version": "v2.1",
                "feature_version": FEATURE_VERSION,
                "data_version": DATA_VERSION,
                "train_report_hash": train_report_hash,
            },
            nn_model_path,
        )
        manifest_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_train_manifest_v2.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(train_manifest, f)
        nn_manifest_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_tsmixer_v2.manifest.json")
        with open(nn_manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": f"liquid_ttm_{symbol.lower()}",
                    "model_version": model_version_nn,
                    "track": "liquid",
                    "type": "tsmixer_liquid",
                    "created_at": created_at,
                    "feature_version": FEATURE_VERSION,
                    "data_version": DATA_VERSION,
                    "feature_payload_schema_version": "v2.1",
                    "n_tokens": n_tokens,
                    "n_channels": n_channels,
                    "ensemble_alpha": float(ensemble_alpha),
                    "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
                    "train_report_hash": train_report_hash,
                    "train_manifest_path": manifest_path,
                    "checkpoint_path": nn_model_path,
                },
                f,
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
                        model_name,
                        "liquid",
                        model_version,
                        model_path,
                        json.dumps(
                            {
                                "train_mse": round(lgb_mse, 9),
                                "samples": int(X.shape[0]),
                                "teacher": teacher_name,
                                "seed": SEED,
                                "walk_forward": wf_metrics,
                                "purged_kfold": pkf_metrics,
                                "data_quality": dq,
                            }
                        ),
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
                        model_version_nn,
                        nn_model_path,
                        json.dumps(
                            {
                                "train_mse": round(nn_mse, 9),
                                "teacher_mse": round(lgb_mse, 9),
                                "distill_gap": round(distill_gap, 9),
                                "val_directional_hit_rate": round(directional_hit, 9),
                                "ensemble_alpha": round(float(ensemble_alpha), 6),
                                "ensemble_metrics": {k: round(float(v), 9) for k, v in ensemble_metrics.items()},
                                "samples": int(X.shape[0]),
                                "grad_acc_steps": GRAD_ACC_STEPS,
                                "best_val_loss": round(best_val, 9),
                                "best_epoch": int(best_epoch),
                                "teacher": teacher_name,
                                "seed": SEED,
                                "walk_forward": wf_metrics,
                                "purged_kfold": pkf_metrics,
                                "train_manifest_path": manifest_path,
                                "train_report_hash": train_report_hash,
                                "checkpoint_manifest_path": nn_manifest_path,
                                "data_quality": dq,
                            }
                        ),
                    ),
                )

        return {
            "symbol": symbol,
            "status": "ok",
            "train_lineage_id": train_lineage_id,
            "samples": int(X.shape[0]),
            "train_mse": round(lgb_mse, 9),
            "nn_mse": round(nn_mse, 9),
            "val_directional_hit_rate": round(directional_hit, 9),
            "distilled_artifact": nn_model_path,
            "teacher": teacher_name,
            "walk_forward_ready": bool(wf_ready),
            "walk_forward": wf_metrics,
            "purged_kfold": pkf_metrics,
            "ensemble_alpha": round(float(ensemble_alpha), 6),
            "ensemble_metrics": {k: round(float(v), 9) for k, v in ensemble_metrics.items()},
            "labels": {
                "fwd_ret_1h_mean": round(float(np.mean(label_1h)), 9),
                "fwd_ret_4h_mean": round(float(np.mean(label_4h)), 9),
            },
            "train_manifest_path": manifest_path,
            "train_report_hash": train_report_hash,
            "checkpoint_manifest_path": nn_manifest_path,
            "data_quality": dq,
        }

    def train_all(self) -> Dict:
        results = [self.train_symbol(symbol) for symbol in self.symbols]
        return {"status": "ok", "results": results}
