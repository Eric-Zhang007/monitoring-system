from __future__ import annotations

import json
import os
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from feature_pipeline import FeaturePipeline, LIQUID_FEATURE_SCHEMA_VERSION
from validation import evaluate_regression_oos, purged_kfold_slices, summarize_fold_metrics, walk_forward_slices

try:
    import lightgbm as lgb  # type: ignore
    HAS_LGB = True
except Exception:
    HAS_LGB = False

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor")


def _default_model_dir() -> str:
    env_path = str(os.getenv("MODEL_DIR", "")).strip()
    if env_path:
        return env_path
    repo_models = Path(__file__).resolve().parents[1] / "backend" / "models"
    if repo_models.exists():
        return str(repo_models)
    return "/app/models"


MODEL_DIR = _default_model_dir()
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
TRAIN_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "4"))
TRAIN_PREFETCH_FACTOR = int(os.getenv("TRAIN_PREFETCH_FACTOR", "4"))
TRAIN_PIN_MEMORY = os.getenv("TRAIN_PIN_MEMORY", "1").lower() in {"1", "true", "yes", "y"}
LIQUID_SYMBOL_DDP = os.getenv("LIQUID_SYMBOL_DDP", "1").lower() in {"1", "true", "yes", "y"}
TRAIN_DQ_GATE_MODE = str(os.getenv("TRAIN_DQ_GATE_MODE", "soft")).strip().lower()
LIQUID_DQ_GATE_MODE = str(os.getenv("LIQUID_DQ_GATE_MODE", TRAIN_DQ_GATE_MODE)).strip().lower()
LIQUID_DQ_HARD_BLOCK = os.getenv("LIQUID_DQ_HARD_BLOCK", "").strip().lower() in {"1", "true", "yes", "y"}


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
    def __init__(
        self,
        pipeline: FeaturePipeline,
        symbols: List[str],
        db_url: str = DATABASE_URL,
        *,
        rank: int = 0,
        world_size: int = 1,
        local_rank: int = 0,
    ):
        self.pipeline = pipeline
        self.symbols = symbols
        self.db_url = db_url
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.local_rank = int(local_rank)
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

    def _resolve_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if 0 <= self.local_rank < torch.cuda.device_count():
            return torch.device(f"cuda:{self.local_rank}")
        return torch.device("cuda")

    def _is_distributed(self) -> bool:
        return bool(self.world_size > 1 and dist.is_available() and dist.is_initialized())

    def _ddp_enabled(self) -> bool:
        return bool(self._is_distributed() and LIQUID_SYMBOL_DDP)

    def _is_primary(self) -> bool:
        return int(self.rank) == 0

    def _barrier(self) -> None:
        if not self._is_distributed():
            return
        if torch.cuda.is_available() and self.local_rank >= 0:
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()

    def _maybe_allreduce_mean(self, val: float, count: int, device: torch.device) -> float:
        if not self._is_distributed():
            return float(val / max(1, count))
        total = torch.tensor(float(val), dtype=torch.float64, device=device)
        n = torch.tensor(float(count), dtype=torch.float64, device=device)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(n, op=dist.ReduceOp.SUM)
        denom = float(n.item())
        if denom <= 0:
            return float("inf")
        return float(total.item() / denom)

    def _loader_kwargs(self, device: torch.device) -> Dict[str, object]:
        workers = max(0, int(TRAIN_NUM_WORKERS))
        kwargs: Dict[str, object] = {
            "num_workers": workers,
            "pin_memory": bool(TRAIN_PIN_MEMORY and device.type == "cuda"),
        }
        if workers > 0:
            kwargs["prefetch_factor"] = max(2, int(TRAIN_PREFETCH_FACTOR))
            kwargs["persistent_workers"] = True
        return kwargs

    def _symbols_for_rank(self) -> List[str]:
        if self.world_size <= 1 or self._ddp_enabled():
            return list(self.symbols)
        return [symbol for idx, symbol in enumerate(self.symbols) if idx % self.world_size == self.rank]

    @staticmethod
    def _ridge_weights(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        reg = np.eye(x_train.shape[1], dtype=np.float64) * 1e-3
        return np.linalg.pinv(x_train.T @ x_train + reg) @ x_train.T @ y_train

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
        ridge_w = self._ridge_weights(x_train, y_train)
        if not HAS_LGB:
            return (
                (x_pred @ ridge_w).astype(np.float32),
                "ridge_fallback",
                {"model": "ridge_fallback", "weights": ridge_w.astype(np.float32).tolist()},
            )
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
            return (
                reg.predict(x_pred).astype(np.float32),
                "lightgbm",
                {
                    "model": "lightgbm",
                    "booster_model": booster,
                    # Keep a deterministic linear surrogate so runtime/backtest can still score
                    # when LightGBM native runtime is unavailable.
                    "weights": ridge_w.astype(np.float32).tolist(),
                    "runtime_fallback": "ridge_linear_surrogate",
                },
            )
        except Exception:
            return (
                (x_pred @ ridge_w).astype(np.float32),
                "ridge_fallback",
                {"model": "ridge_fallback", "weights": ridge_w.astype(np.float32).tolist()},
            )

    def train_symbol(self, symbol: str) -> Dict:
        self._set_seed(SEED + self.rank)
        tf_candidates: List[str] = []
        primary_tf = os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m").strip().lower()
        for tf in [primary_tf, "5m", "15m", "1h"]:
            cur_tf = tf.strip().lower()
            if cur_tf and cur_tf not in tf_candidates:
                tf_candidates.append(cur_tf)
        selected_tf = tf_candidates[0] if tf_candidates else "5m"
        dq: Dict = {}
        best_dq: Dict | None = None
        for tf in tf_candidates:
            check = self.pipeline.check_data_quality(symbol, timeframe=tf, lookback_hours=72)
            if (best_dq is None) or (float(check.get("total_rows", 0.0)) > float(best_dq.get("total_rows", 0.0))):
                best_dq = check
                selected_tf = tf
            if float(check.get("quality_passed", 0.0)) >= 0.5:
                dq = check
                selected_tf = tf
                break
        if not dq:
            dq = best_dq or {}
        dq_failed_reasons = []
        if float(dq.get("missing_rate", 0.0)) > float(os.getenv("DQ_MAX_MISSING_RATE", "0.02")):
            dq_failed_reasons.append("missing_rate_exceeded")
        if float(dq.get("invalid_price_rate", 0.0)) > float(os.getenv("DQ_MAX_INVALID_PRICE_RATE", "0.005")):
            dq_failed_reasons.append("invalid_price_rate_exceeded")
        if float(dq.get("duplicate_rate", 0.0)) > float(os.getenv("DQ_MAX_DUPLICATE_RATE", "0.02")):
            dq_failed_reasons.append("duplicate_rate_exceeded")
        if float(dq.get("stale_ratio", 0.0)) > float(os.getenv("DQ_MAX_STALE_RATIO", "0.1")):
            dq_failed_reasons.append("stale_ratio_exceeded")
        dq_degraded = False
        if dq.get("quality_passed", 0.0) < 0.5:
            if float(dq.get("total_rows", 0.0)) < float(dq.get("required_rows", 0.0)):
                dq_failed_reasons.append("insufficient_rows")
            gate_mode = LIQUID_DQ_GATE_MODE if LIQUID_DQ_GATE_MODE in {"soft", "hard"} else "soft"
            hard_block = bool(LIQUID_DQ_HARD_BLOCK or gate_mode == "hard")
            if hard_block:
                return {
                    "symbol": symbol,
                    "status": "blocked_by_data_quality",
                    "reason": ",".join(dq_failed_reasons) or "quality_gate_failed",
                    "timeframe": selected_tf,
                    "data_quality": dq,
                    "dq_gate_mode": "hard",
                }
            dq_degraded = True
            dq = {
                **dq,
                "dq_gate_mode": "soft",
                "dq_degraded": True,
                "dq_failed_reasons": dq_failed_reasons,
            }

        batch = self.pipeline.load_liquid_training_batch(symbol=symbol, limit=4000, timeframe=selected_tf)
        if batch.X.shape[0] == 0:
            return {"symbol": symbol, "status": "no_data", "timeframe": selected_tf, "data_quality": dq}

        X, y = batch.X, batch.y
        train_lineage_id = f"train-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{symbol.lower()}"
        feature_rows = [
            {
                **self.pipeline.vector_to_feature_payload(r),
                "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
            }
            for r in X
        ]
        if (not self._ddp_enabled()) or self._is_primary():
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
        device = self._resolve_device()
        model = TSMixerLiquidModel(n_tokens=n_tokens, n_channels=n_channels, n_blocks=2).to(device)
        if self._ddp_enabled():
            model = DDP(
                model,
                device_ids=[self.local_rank] if device.type == "cuda" else None,
                output_device=self.local_rank if device.type == "cuda" else None,
            )
        raw_model = model.module if isinstance(model, DDP) else model
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
        train_sampler = None
        val_sampler = None
        if self._ddp_enabled():
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                drop_last=False,
            )
            val_sampler = DistributedSampler(
                val_ds,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )
        loader_kwargs = self._loader_kwargs(device)
        train_bs = BATCH_SIZE
        ckpt_path = os.path.join(MODEL_DIR, f"liquid_{symbol.lower()}_checkpoint.pt")
        best_val = float("inf")
        best_epoch = 0
        bad_epochs = 0
        start_epoch = 0
        ckpt_payload = None
        if self._ddp_enabled():
            if self._is_primary() and os.path.exists(ckpt_path):
                ckpt_payload = torch.load(ckpt_path, map_location=device)
            payload_box = [ckpt_payload]
            dist.broadcast_object_list(payload_box, src=0)
            ckpt_payload = payload_box[0]
        elif os.path.exists(ckpt_path):
            ckpt_payload = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt_payload, dict):
            raw_model.load_state_dict(ckpt_payload["model_state"])
            opt.load_state_dict(ckpt_payload["optimizer_state"])
            start_epoch = int(ckpt_payload.get("epoch", 0)) + 1
            best_val = float(ckpt_payload.get("best_val", best_val))

        for epoch in range(start_epoch, EPOCHS):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_dl = DataLoader(
                train_ds,
                batch_size=train_bs,
                shuffle=bool(train_sampler is None),
                sampler=train_sampler,
                drop_last=False,
                **loader_kwargs,
            )
            val_dl = DataLoader(
                val_ds,
                batch_size=train_bs,
                shuffle=False,
                sampler=val_sampler,
                drop_last=False,
                **loader_kwargs,
            )
            model.train()
            opt.zero_grad(set_to_none=True)
            try:
                for step, (bx, by, bt) in enumerate(train_dl, start=1):
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    bt = bt.to(device, non_blocking=True)
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
                val_loss_sum = 0.0
                val_n = 0
                for bx, by, bt in val_dl:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    bt = bt.to(device, non_blocking=True)
                    pred = model(bx)
                    lv = 0.85 * mse_loss(pred, by) + 0.15 * mse_loss(pred, bt)
                    cur_n = int(bx.shape[0])
                    val_loss_sum += float(lv.item()) * max(1, cur_n)
                    val_n += max(1, cur_n)
                val_loss = self._maybe_allreduce_mean(val_loss_sum, val_n, device=device)
                scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_epochs = 0
                if (not self._ddp_enabled()) or self._is_primary():
                    torch.save(
                        {
                            "epoch": epoch,
                            "best_val": best_val,
                            "model_state": raw_model.state_dict(),
                            "optimizer_state": opt.state_dict(),
                        },
                        ckpt_path,
                    )
            else:
                bad_epochs += 1
            should_stop = int(bad_epochs >= PATIENCE)
            if self._is_distributed():
                stop_tensor = torch.tensor([should_stop], dtype=torch.int64, device=device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = int(stop_tensor.item())
            if should_stop:
                break

        final_ckpt = None
        if self._ddp_enabled():
            if self._is_primary() and os.path.exists(ckpt_path):
                final_ckpt = torch.load(ckpt_path, map_location=device)
            payload_box = [final_ckpt]
            dist.broadcast_object_list(payload_box, src=0)
            final_ckpt = payload_box[0]
        elif os.path.exists(ckpt_path):
            final_ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(final_ckpt, dict):
            raw_model.load_state_dict(final_ckpt["model_state"])

        if self._ddp_enabled() and (not self._is_primary()):
            return {
                "symbol": symbol,
                "status": "ddp_worker_done",
                "timeframe": selected_tf,
                "samples": int(X.shape[0]),
                "rank": self.rank,
                "world_size": self.world_size,
            }

        raw_model.eval()
        with torch.no_grad():
            pred_nn = raw_model(x_val.to(device, non_blocking=True)).detach().cpu().numpy()
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
                "state_dict": raw_model.state_dict(),
                "n_tokens": n_tokens,
                "n_channels": n_channels,
                "trained_at": created_at,
                "type": "tsmixer_liquid",
                "seed": SEED,
                "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
                "ensemble_alpha": float(ensemble_alpha),
                "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
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
            "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
            "dq_degraded": bool(dq_degraded),
            "dq_gate_mode": "hard" if (LIQUID_DQ_HARD_BLOCK or LIQUID_DQ_GATE_MODE == "hard") else "soft",
            "dq_failed_reasons": dq_failed_reasons,
        }
        train_report_hash = hashlib.sha256(
            json.dumps(train_manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        train_manifest["train_report_hash"] = train_report_hash
        torch.save(
            {
                "state_dict": raw_model.state_dict(),
                "n_tokens": n_tokens,
                "n_channels": n_channels,
                "trained_at": created_at,
                "type": "tsmixer_liquid",
                "seed": SEED,
                "normalization": {"x_mean": x_mean.flatten().tolist(), "x_std": x_std.flatten().tolist()},
                "ensemble_alpha": float(ensemble_alpha),
                "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
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
                    "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
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
                                "dq_degraded": bool(dq_degraded),
                                "dq_failed_reasons": dq_failed_reasons,
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
                                "dq_degraded": bool(dq_degraded),
                                "dq_failed_reasons": dq_failed_reasons,
                            }
                        ),
                    ),
                )

        return {
            "symbol": symbol,
            "status": "ok",
            "timeframe": selected_tf,
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
            "dq_degraded": bool(dq_degraded),
            "dq_failed_reasons": dq_failed_reasons,
        }

    def train_all(self) -> Dict:
        assigned = self._symbols_for_rank()
        if not assigned:
            return {
                "status": "no_symbols_assigned",
                "rank": self.rank,
                "world_size": self.world_size,
                "assigned_symbols": [],
                "results": [],
            }
        results: List[Dict] = []
        for symbol in assigned:
            result = self.train_symbol(symbol)
            if (not self._ddp_enabled()) or self._is_primary():
                results.append(result)
            self._barrier()
        return {
            "status": "ok",
            "rank": self.rank,
            "world_size": self.world_size,
            "assigned_symbols": assigned,
            "results": results,
        }
