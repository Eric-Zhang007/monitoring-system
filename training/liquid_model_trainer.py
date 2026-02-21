from __future__ import annotations

import json
import os
import random
import hashlib
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from feature_pipeline import FeaturePipeline, LIQUID_FEATURE_KEYS, LIQUID_FEATURE_SCHEMA_VERSION
from validation import (
    evaluate_regression_oos,
    purged_kfold_slices,
    resolve_validation_protocol,
    summarize_fold_metrics,
    validation_protocol_to_dict,
    walk_forward_slices,
)

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
    return "/opt/monitoring-system/models"


MODEL_DIR = _default_model_dir()
GRAD_ACC_STEPS = int(os.getenv("TRAIN_GRAD_ACC_STEPS", "4"))
EPOCHS = int(os.getenv("LIQUID_EPOCHS", "14"))
BATCH_SIZE = int(os.getenv("LIQUID_BATCH_SIZE", "128"))
SEED = int(os.getenv("TRAIN_SEED", "42"))
VAL_RATIO = float(os.getenv("LIQUID_VAL_RATIO", "0.2"))
PATIENCE = int(os.getenv("LIQUID_EARLY_STOP_PATIENCE", "3"))
GRAD_CLIP = float(os.getenv("TRAIN_GRAD_CLIP", "1.0"))
VALIDATION_PROTOCOL = resolve_validation_protocol(prefix="LIQUID")
FEATURE_VERSION = os.getenv("FEATURE_VERSION", "feature-store-v2.1")
DATA_VERSION = os.getenv("DATA_VERSION", "v1")
TRAIN_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "4"))
TRAIN_PREFETCH_FACTOR = int(os.getenv("TRAIN_PREFETCH_FACTOR", "4"))
TRAIN_PIN_MEMORY = os.getenv("TRAIN_PIN_MEMORY", "1").lower() in {"1", "true", "yes", "y"}
LIQUID_SYMBOL_DDP = os.getenv("LIQUID_SYMBOL_DDP", "1").lower() in {"1", "true", "yes", "y"}
TRAIN_DQ_GATE_MODE = str(os.getenv("TRAIN_DQ_GATE_MODE", "soft")).strip().lower()
LIQUID_DQ_GATE_MODE = str(os.getenv("LIQUID_DQ_GATE_MODE", TRAIN_DQ_GATE_MODE)).strip().lower()
LIQUID_DQ_HARD_BLOCK = os.getenv("LIQUID_DQ_HARD_BLOCK", "").strip().lower() in {"1", "true", "yes", "y"}
LIQUID_TEXT_DROPOUT_PROB = float(os.getenv("LIQUID_TEXT_DROPOUT_PROB", os.getenv("MULTIMODAL_TEXT_DROPOUT_PROB", "0.1")))


def _safe_env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return int(default)


LIQUID_TRAIN_LIMIT = _safe_env_int("LIQUID_TRAIN_LIMIT", 4000)
LIQUID_TRAIN_MAX_SAMPLES = _safe_env_int("LIQUID_TRAIN_MAX_SAMPLES", 0)
LIQUID_TRAIN_SAMPLE_MODE = str(os.getenv("LIQUID_TRAIN_SAMPLE_MODE", "uniform")).strip().lower() or "uniform"
LIQUID_TRAIN_START = str(os.getenv("LIQUID_TRAIN_START", "")).strip()
LIQUID_TRAIN_END = str(os.getenv("LIQUID_TRAIN_END", "")).strip()
LIQUID_TRAIN_LOOKBACK_DAYS = _safe_env_int("LIQUID_TRAIN_LOOKBACK_DAYS", 365)
LIQUID_TRAIN_MODE = str(os.getenv("LIQUID_TRAIN_MODE", "production")).strip().lower() or "production"
LIQUID_DATA_MODE = str(os.getenv("LIQUID_DATA_MODE", "production")).strip().lower() or "production"
LIQUID_RESEARCH_MAX_MISSING_FLAGS = _safe_env_int("LIQUID_RESEARCH_MAX_MISSING_FLAGS", 2)
LIQUID_RESEARCH_MAX_SAMPLES = _safe_env_int("LIQUID_RESEARCH_MAX_SAMPLES", 50000)
LIQUID_MULTI_HORIZON_TRAIN_MODE = str(
    os.getenv("LIQUID_MULTI_HORIZON_TRAIN_MODE", "single_model_multihead")
).strip().lower() or "single_model_multihead"
LIQUID_HORIZONS: List[str] = ["1h", "4h", "1d", "7d"]


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
        train_start: str | datetime | None = None,
        train_end: str | datetime | None = None,
        train_limit: int | None = None,
        train_max_samples: int | None = None,
        train_sample_mode: str | None = None,
        train_data_mode: str | None = None,
    ):
        self.pipeline = pipeline
        self.symbols = symbols
        self.db_url = db_url
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.local_rank = int(local_rank)
        self.train_start = self._parse_optional_utc(train_start if train_start is not None else LIQUID_TRAIN_START)
        self.train_end = self._parse_optional_utc(train_end if train_end is not None else LIQUID_TRAIN_END)
        self.train_mode = str(os.getenv("LIQUID_TRAIN_MODE", LIQUID_TRAIN_MODE)).strip().lower() or "production"
        self.train_lookback_days = max(1, int(os.getenv("LIQUID_TRAIN_LOOKBACK_DAYS", str(LIQUID_TRAIN_LOOKBACK_DAYS)) or LIQUID_TRAIN_LOOKBACK_DAYS))
        requested_limit = max(0, int(LIQUID_TRAIN_LIMIT if train_limit is None else train_limit))
        # Explicit train_limit always wins; otherwise production defaults to window-based sampling.
        if train_limit is None:
            self.train_limit = int(requested_limit if self.train_mode == "fast" else 0)
        else:
            self.train_limit = int(requested_limit)
        self.train_max_samples = max(0, int(LIQUID_TRAIN_MAX_SAMPLES if train_max_samples is None else train_max_samples))
        self.train_sample_mode = str(train_sample_mode if train_sample_mode is not None else LIQUID_TRAIN_SAMPLE_MODE).strip().lower() or "uniform"
        self.train_data_mode = str(train_data_mode if train_data_mode is not None else LIQUID_DATA_MODE).strip().lower() or "production"
        if self.train_end is None:
            self.train_end = datetime.now(timezone.utc)
        if self.train_start is None and isinstance(self.train_end, datetime):
            self.train_start = self.train_end - timedelta(days=max(1, int(self.train_lookback_days)))
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
    def _parse_optional_utc(raw: str | datetime | None) -> datetime | None:
        if raw is None:
            return None
        if isinstance(raw, datetime):
            dt = raw
        else:
            text = str(raw or "").strip()
            if not text:
                return None
            text = text.replace(" ", "T")
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                dt = datetime.fromisoformat(text)
            except Exception:
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

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
    def _resolve_horizon_labels(
        y_default: np.ndarray,
        extra_labels: Dict[str, np.ndarray] | None,
    ) -> Dict[str, np.ndarray]:
        extra = extra_labels or {}
        out: Dict[str, np.ndarray] = {}
        alias = {
            "1h": ["net_ret_1h", "fwd_ret_1h"],
            "4h": ["net_ret_4h", "fwd_ret_4h"],
            "1d": ["net_ret_1d"],
            "7d": ["net_ret_7d"],
        }
        for horizon in LIQUID_HORIZONS:
            arr = None
            for key in alias.get(horizon, []):
                cur = extra.get(key)
                if cur is not None:
                    arr = np.asarray(cur, dtype=np.float32).reshape(-1)
                    break
            if arr is None or arr.size != y_default.shape[0]:
                arr = np.asarray(y_default, dtype=np.float32).reshape(-1)
            out[horizon] = arr
        return out

    @staticmethod
    def _resolve_horizon_vol_targets(
        y_by_horizon: Dict[str, np.ndarray],
        extra_labels: Dict[str, np.ndarray] | None,
    ) -> Dict[str, np.ndarray]:
        extra = extra_labels or {}
        out: Dict[str, np.ndarray] = {}
        for horizon in LIQUID_HORIZONS:
            key = f"vol_target_{horizon}"
            cur = extra.get(key)
            if cur is None:
                cur_arr = np.abs(np.asarray(y_by_horizon[horizon], dtype=np.float32))
            else:
                cur_arr = np.asarray(cur, dtype=np.float32).reshape(-1)
            out[horizon] = cur_arr
        return out

    @staticmethod
    def _safe_sigmoid(x: np.ndarray) -> np.ndarray:
        xx = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-xx))

    def _fit_horizon_heads(
        self,
        Xn_train_drop: np.ndarray,
        Xn_full: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        y_by_horizon: Dict[str, np.ndarray],
        vol_targets: Dict[str, np.ndarray],
    ) -> Dict[str, Dict[str, object]]:
        out: Dict[str, Dict[str, object]] = {}
        for horizon in LIQUID_HORIZONS:
            y_h = np.asarray(y_by_horizon[horizon], dtype=np.float64)
            vol_h = np.asarray(vol_targets[horizon], dtype=np.float64)
            w = self._ridge_weights(Xn_train_drop.astype(np.float64), y_h[train_idx])
            pred_all = (Xn_full.astype(np.float64) @ w).astype(np.float64)
            pred_val = pred_all[val_idx]
            y_val = y_h[val_idx]
            resid_val = y_val - pred_val
            resid_std = float(np.std(resid_val)) if resid_val.size else 0.0

            wv = self._ridge_weights(Xn_train_drop.astype(np.float64), vol_h[train_idx])
            vol_pred_all = np.clip((Xn_full.astype(np.float64) @ wv).astype(np.float64), 1e-6, None)
            vol_pred_val = vol_pred_all[val_idx]
            z_val = pred_val / np.clip(vol_pred_val, 1e-6, None)
            hit_val = (np.sign(pred_val) == np.sign(y_val)).astype(np.float64) if y_val.size else np.zeros((0,), dtype=np.float64)

            conf_a = 0.85
            conf_b = -0.1
            conf_val = self._safe_sigmoid(conf_a * np.abs(z_val) + conf_b)
            order = np.argsort(conf_val)
            bins = max(1, min(5, int(conf_val.size // 16) if conf_val.size > 0 else 1))
            calib_bins: List[Dict[str, float]] = []
            if conf_val.size > 0:
                for b in np.array_split(order, bins):
                    if b.size <= 0:
                        continue
                    calib_bins.append(
                        {
                            "count": int(b.size),
                            "conf_mean": float(np.mean(conf_val[b])),
                            "hit_rate": float(np.mean(hit_val[b])) if hit_val.size else 0.0,
                            "score_abs_mean": float(np.mean(np.abs(pred_val[b]))),
                        }
                    )

            out[horizon] = {
                "weights": w.astype(np.float32).tolist(),
                "vol_weights": wv.astype(np.float32).tolist(),
                "residual_std": float(max(1e-6, resid_std)),
                "calibration": {
                    "method": "sigmoid_abs_z",
                    "a": float(conf_a),
                    "b": float(conf_b),
                    "bins": calib_bins,
                },
                "metrics": {
                    "mse": float(np.mean((pred_val - y_val) ** 2)) if y_val.size else 0.0,
                    "mae": float(np.mean(np.abs(pred_val - y_val))) if y_val.size else 0.0,
                    "hit_rate": float(np.mean(hit_val)) if hit_val.size else 0.0,
                },
            }
        return out

    def _fit_horizon_models_with_meta(
        self,
        Xn_train_drop: np.ndarray,
        Xn_full: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        y_by_horizon: Dict[str, np.ndarray],
        vol_targets: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        # Multi-model mode: one submodel per horizon + a lightweight linear meta-aggregator.
        heads = self._fit_horizon_heads(
            Xn_train_drop=Xn_train_drop,
            Xn_full=Xn_full,
            train_idx=train_idx,
            val_idx=val_idx,
            y_by_horizon=y_by_horizon,
            vol_targets=vol_targets,
        )
        ordered_h = [h for h in LIQUID_HORIZONS if h in heads]
        if not ordered_h:
            return {"horizon_models": {}, "meta_aggregator": {}}

        pred_cols: List[np.ndarray] = []
        for h in ordered_h:
            ww = np.asarray((heads.get(h) or {}).get("weights") or [], dtype=np.float64).reshape(-1)
            if ww.size <= 0:
                continue
            pred_cols.append((Xn_full.astype(np.float64) @ ww).reshape(-1))
        if not pred_cols:
            return {"horizon_models": {}, "meta_aggregator": {}}
        pred_mat = np.stack(pred_cols, axis=1).astype(np.float64)
        y_meta = np.asarray(y_by_horizon.get("1h"), dtype=np.float64).reshape(-1)
        meta_x = pred_mat[train_idx]
        meta_y = y_meta[train_idx]
        meta_w = self._ridge_weights(meta_x, meta_y).astype(np.float64)
        meta_val = pred_mat[val_idx] @ meta_w
        meta_y_val = y_meta[val_idx]
        hit = (np.sign(meta_val) == np.sign(meta_y_val)).astype(np.float64) if meta_y_val.size else np.zeros((0,), dtype=np.float64)

        horizon_models: Dict[str, Dict[str, Any]] = {}
        for h in ordered_h:
            item = dict(heads.get(h) or {})
            item["model_type"] = "ridge_submodel"
            horizon_models[h] = item

        return {
            "horizon_models": horizon_models,
            "meta_aggregator": {
                "model_type": "linear_meta",
                "horizons": ordered_h,
                "weights": meta_w.astype(np.float32).tolist(),
                "target_horizon": "1h",
                "metrics": {
                    "mse": float(np.mean((meta_val - meta_y_val) ** 2)) if meta_y_val.size else 0.0,
                    "mae": float(np.mean(np.abs(meta_val - meta_y_val))) if meta_y_val.size else 0.0,
                    "hit_rate": float(np.mean(hit)) if hit.size else 0.0,
                },
            },
        }

    @staticmethod
    def _infer_text_feature_indices(keys: List[str]) -> List[int]:
        out: List[int] = []
        for idx, key in enumerate(keys):
            k = str(key or "").strip().lower()
            if not k:
                continue
            if k.startswith("social_") or k.startswith("event_") or k.startswith("source_") or k.startswith("sentiment_"):
                out.append(idx)
                continue
            if k in {"comment_skew", "cross_source_consensus", "novelty_confidence_blend"}:
                out.append(idx)
                continue
            m = re.fullmatch(r"latent_(\d+)", k)
            if m:
                latent_id = int(m.group(1))
                if latent_id >= 64:
                    out.append(idx)
        return sorted(set(out))

    @staticmethod
    def _apply_feature_dropout(
        X: np.ndarray,
        indices: List[int],
        prob: float,
        seed: int,
    ) -> np.ndarray:
        if X.size == 0 or not indices:
            return X
        p = max(0.0, min(0.95, float(prob)))
        if p <= 1e-9:
            return X
        out = X.copy()
        rng = np.random.default_rng(int(seed))
        mask = rng.random((out.shape[0], len(indices))) < p
        selected = out[:, indices]
        selected[mask] = 0.0
        out[:, indices] = selected
        return out

    @staticmethod
    def _apply_index_mask(batch: "SampleBatch", idx: np.ndarray) -> "SampleBatch":
        index = np.array(idx, dtype=np.int64)
        extra = batch.extra_labels or {}
        filtered_extra = {
            k: np.asarray(v)[index]
            for k, v in extra.items()
        }
        meta = list(batch.meta or [])
        filtered_meta = [meta[i] for i in index.tolist()] if meta else []
        return type(batch)(
            X=np.asarray(batch.X)[index],
            y=np.asarray(batch.y)[index],
            meta=filtered_meta,
            extra_labels=filtered_extra,
            sampling=dict(batch.sampling or {}),
        )

    def _apply_research_mode_guardrails(self, batch: "SampleBatch") -> tuple["SampleBatch", Dict[str, object]]:
        if str(self.train_data_mode) != "research":
            return batch, {"enabled": False}
        n_rows = int(batch.X.shape[0]) if hasattr(batch, "X") else 0
        if n_rows <= 0:
            return batch, {"enabled": True, "rows_before": 0, "rows_after": 0}

        missing_idx = [
            i for i, key in enumerate(LIQUID_FEATURE_KEYS)
            if str(key).endswith("_missing_flag")
        ]
        keep = np.ones((n_rows,), dtype=bool)
        max_missing_allowed = max(0, int(LIQUID_RESEARCH_MAX_MISSING_FLAGS))
        if missing_idx:
            miss = np.asarray(batch.X[:, missing_idx], dtype=np.float64)
            miss_cnt = np.sum(miss > 0.5, axis=1)
            keep = miss_cnt <= max_missing_allowed
        kept_idx = np.where(keep)[0].astype(np.int64)
        guarded = self._apply_index_mask(batch, kept_idx)

        rows_after_missing = int(guarded.X.shape[0])
        cap = int(self.train_max_samples) if int(self.train_max_samples) > 0 else int(LIQUID_RESEARCH_MAX_SAMPLES)
        downsampled = False
        if cap > 0 and rows_after_missing > cap:
            ds_idx = np.linspace(0, rows_after_missing - 1, cap).astype(np.int64)
            guarded = self._apply_index_mask(guarded, ds_idx)
            downsampled = True
        guard = {
            "enabled": True,
            "rows_before": n_rows,
            "rows_after_missing_filter": rows_after_missing,
            "rows_after": int(guarded.X.shape[0]),
            "missing_flag_count": int(len(missing_idx)),
            "max_missing_flags_allowed": int(max_missing_allowed),
            "downsampled": bool(downsampled),
            "target_cap": int(cap),
        }
        sampling = dict(guarded.sampling or {})
        sampling["data_mode"] = "research"
        sampling["research_guardrails"] = guard
        guarded.sampling = sampling
        return guarded, guard

    @staticmethod
    def _to_sequence(xn: np.ndarray, n_channels: int = 5) -> np.ndarray:
        if xn.ndim != 2:
            raise ValueError("expected 2D feature matrix")
        n, d = xn.shape
        ch = max(1, int(n_channels))
        tok = int(max(1, int(np.ceil(float(d) / float(ch)))))
        target_dim = tok * ch
        if target_dim > d:
            pad = np.zeros((n, target_dim - d), dtype=np.float32)
            xpad = np.concatenate([xn.astype(np.float32), pad], axis=1)
        else:
            xpad = xn[:, :target_dim].astype(np.float32)
        return xpad.reshape(n, tok, ch)

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

        batch = self.pipeline.load_liquid_training_batch(
            symbol=symbol,
            limit=self.train_limit,
            timeframe=selected_tf,
            start=self.train_start,
            end=self.train_end,
            max_samples=self.train_max_samples,
            sample_mode=self.train_sample_mode,
        )
        batch, research_guard = self._apply_research_mode_guardrails(batch)
        if batch.X.shape[0] == 0:
            return {
                "symbol": symbol,
                "status": "no_data",
                "timeframe": selected_tf,
                "data_quality": dq,
                "sampling_strategy": {
                    **(batch.sampling or {}),
                    "data_mode": str(self.train_data_mode),
                    "research_guardrails": research_guard,
                },
            }

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
            row_times = []
            for m in (batch.meta or []):
                if isinstance(m, dict) and isinstance(m.get("as_of_ts"), datetime):
                    row_times.append(m["as_of_ts"])
            self.pipeline.save_feature_snapshots_bulk(
                target=symbol.upper(),
                track="liquid",
                feature_rows=feature_rows,
                version=FEATURE_VERSION,
                lineage_id=train_lineage_id,
                data_version=DATA_VERSION,
                row_times=row_times,
            )
        y_by_horizon = self._resolve_horizon_labels(y, batch.extra_labels)
        vol_targets = self._resolve_horizon_vol_targets(y_by_horizon, batch.extra_labels)
        n = X.shape[0]
        val_n = max(64, int(n * VAL_RATIO))
        if val_n >= n:
            val_n = max(1, n // 4)
        train_idx = np.arange(0, n - val_n)
        val_idx = np.arange(n - val_n, n)

        x_mean = np.mean(X[train_idx], axis=0, keepdims=True)
        x_std = np.clip(np.std(X[train_idx], axis=0, keepdims=True), 1e-6, None)
        Xn = (X - x_mean) / x_std
        text_feature_indices = self._infer_text_feature_indices(LIQUID_FEATURE_KEYS)
        Xn_train_drop = self._apply_feature_dropout(
            Xn[train_idx],
            indices=text_feature_indices,
            prob=LIQUID_TEXT_DROPOUT_PROB,
            seed=SEED + self.rank,
        )
        Xn_seq = Xn.copy()
        Xn_seq[train_idx] = Xn_train_drop

        y_train = y_by_horizon["1h"][train_idx]
        y_val = y[val_idx]
        lgb_all_pred, lgb_model_name, lgb_model_payload = self._fit_lightgbm(Xn_train_drop, y_train, Xn)
        lgb_mse = float(np.mean((lgb_all_pred[val_idx] - y_val) ** 2))
        train_mode = str(LIQUID_MULTI_HORIZON_TRAIN_MODE).strip().lower()
        horizon_models: Dict[str, Dict[str, Any]] = {}
        meta_aggregator: Dict[str, Any] = {}
        if train_mode in {"multi_model_meta", "multi_model", "per_horizon_submodels"}:
            multi_model_pack = self._fit_horizon_models_with_meta(
                Xn_train_drop=Xn_train_drop,
                Xn_full=Xn,
                train_idx=train_idx,
                val_idx=val_idx,
                y_by_horizon=y_by_horizon,
                vol_targets=vol_targets,
            )
            horizon_models = dict(multi_model_pack.get("horizon_models") or {})
            meta_aggregator = dict(multi_model_pack.get("meta_aggregator") or {})
            # Keep backward compatibility: expose horizon heads even when trained as multi-model.
            horizon_heads = {
                h: {k: v for k, v in dict(item).items() if k != "model_type"}
                for h, item in horizon_models.items()
            }
        else:
            horizon_heads = self._fit_horizon_heads(
                Xn_train_drop=Xn_train_drop,
                Xn_full=Xn,
                train_idx=train_idx,
                val_idx=val_idx,
                y_by_horizon=y_by_horizon,
                vol_targets=vol_targets,
            )
        wf_folds = walk_forward_slices(
            n_samples=n,
            train_window=VALIDATION_PROTOCOL.wf_train_window,
            test_window=VALIDATION_PROTOCOL.wf_test_window,
            purge_window=VALIDATION_PROTOCOL.wf_purge_window,
            step_window=VALIDATION_PROTOCOL.wf_step_window,
        )
        wf_fold_metrics: List[Dict[str, float]] = []
        for fold_train_idx, fold_test_idx in wf_folds:
            if (
                fold_train_idx.size < VALIDATION_PROTOCOL.min_train_points
                or fold_test_idx.size < VALIDATION_PROTOCOL.min_test_points
            ):
                continue
            fold_x_train = self._apply_feature_dropout(
                Xn[fold_train_idx],
                indices=text_feature_indices,
                prob=LIQUID_TEXT_DROPOUT_PROB,
                seed=SEED + self.rank + int(fold_train_idx[0]) if fold_train_idx.size > 0 else SEED + self.rank,
            )
            fold_pred, _, _ = self._fit_lightgbm(fold_x_train, y_by_horizon["1h"][fold_train_idx], Xn[fold_test_idx])
            wf_fold_metrics.append(
                evaluate_regression_oos(
                    y_true=y_by_horizon["1h"][fold_test_idx],
                    y_pred=fold_pred,
                    fee_bps=5.0,
                    slippage_bps=3.0,
                )
            )
        wf_metrics = summarize_fold_metrics(wf_fold_metrics)
        wf_ready = int(wf_metrics["folds"]) >= VALIDATION_PROTOCOL.wf_min_folds
        pkf_folds = purged_kfold_slices(
            n_samples=n,
            n_splits=VALIDATION_PROTOCOL.pkf_splits,
            purge_window=VALIDATION_PROTOCOL.pkf_purge_window,
        )
        pkf_metrics_per_fold: List[Dict[str, float]] = []
        for fold_train_idx, fold_test_idx in pkf_folds:
            if (
                fold_train_idx.size < VALIDATION_PROTOCOL.min_train_points
                or fold_test_idx.size < VALIDATION_PROTOCOL.min_test_points
            ):
                continue
            fold_x_train = self._apply_feature_dropout(
                Xn[fold_train_idx],
                indices=text_feature_indices,
                prob=LIQUID_TEXT_DROPOUT_PROB,
                seed=SEED + self.rank + int(fold_train_idx[0]) if fold_train_idx.size > 0 else SEED + self.rank,
            )
            fold_pred, _, _ = self._fit_lightgbm(fold_x_train, y_by_horizon["1h"][fold_train_idx], Xn[fold_test_idx])
            pkf_metrics_per_fold.append(
                evaluate_regression_oos(
                    y_true=y_by_horizon["1h"][fold_test_idx],
                    y_pred=fold_pred,
                    fee_bps=5.0,
                    slippage_bps=3.0,
                )
            )
        pkf_metrics = summarize_fold_metrics(pkf_metrics_per_fold)
        teacher = lgb_all_pred.copy()
        teacher_name = lgb_model_name

        seq_channels = int(os.getenv("LIQUID_SEQUENCE_CHANNELS", "5"))
        seq = self._to_sequence(Xn_seq, n_channels=seq_channels)
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
                "default_horizon": "1h",
                "horizons": list(LIQUID_HORIZONS),
                "multi_horizon_train_mode": str(LIQUID_MULTI_HORIZON_TRAIN_MODE),
                "horizon_heads": horizon_heads,
                "horizon_models": horizon_models,
                "meta_aggregator": meta_aggregator,
                "cost_config": dict((batch.sampling or {}).get("cost_config") or {}),
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
            "wf_train_window": VALIDATION_PROTOCOL.wf_train_window,
            "wf_test_window": VALIDATION_PROTOCOL.wf_test_window,
            "wf_purge_window": VALIDATION_PROTOCOL.wf_purge_window,
            "wf_step_window": VALIDATION_PROTOCOL.wf_step_window,
            "wf_min_folds": VALIDATION_PROTOCOL.wf_min_folds,
            "pkf_splits": VALIDATION_PROTOCOL.pkf_splits,
            "pkf_purge_window": VALIDATION_PROTOCOL.pkf_purge_window,
            "validation_min_train_points": VALIDATION_PROTOCOL.min_train_points,
            "validation_min_test_points": VALIDATION_PROTOCOL.min_test_points,
            "validation_protocol": validation_protocol_to_dict(VALIDATION_PROTOCOL),
            "text_dropout_prob": float(max(0.0, min(0.95, LIQUID_TEXT_DROPOUT_PROB))),
            "text_feature_count": int(len(text_feature_indices)),
            "wf_ready": wf_ready,
            "wf_metrics": wf_metrics,
            "purged_kfold": pkf_metrics,
            "horizons": list(LIQUID_HORIZONS),
            "multi_horizon_train_mode": str(LIQUID_MULTI_HORIZON_TRAIN_MODE),
            "horizon_head_metrics": {h: dict((horizon_heads.get(h) or {}).get("metrics") or {}) for h in LIQUID_HORIZONS},
            "meta_aggregator_metrics": dict((meta_aggregator or {}).get("metrics") or {}),
            "feature_dim": int(X.shape[1]),
            "samples": int(X.shape[0]),
            "created_at": created_at,
            "feature_version": FEATURE_VERSION,
            "data_version": DATA_VERSION,
            "feature_payload_schema_version": LIQUID_FEATURE_SCHEMA_VERSION,
            "dq_degraded": bool(dq_degraded),
            "dq_gate_mode": "hard" if (LIQUID_DQ_HARD_BLOCK or LIQUID_DQ_GATE_MODE == "hard") else "soft",
            "dq_failed_reasons": dq_failed_reasons,
            "sampling_strategy": {
                **(batch.sampling or {}),
                "requested_train_start": self.train_start.isoformat().replace("+00:00", "Z") if isinstance(self.train_start, datetime) else "",
                "requested_train_end": self.train_end.isoformat().replace("+00:00", "Z") if isinstance(self.train_end, datetime) else "",
                "requested_limit": int(self.train_limit),
                "requested_max_samples": int(self.train_max_samples),
                "requested_sample_mode": str(self.train_sample_mode),
                "train_mode": str(self.train_mode),
                "train_lookback_days": int(self.train_lookback_days),
                "data_mode": str(self.train_data_mode),
                "research_guardrails": research_guard,
            },
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
                "net_ret_1h_mean": round(float(np.mean(y_by_horizon["1h"])), 9),
                "net_ret_4h_mean": round(float(np.mean(y_by_horizon["4h"])), 9),
                "net_ret_1d_mean": round(float(np.mean(y_by_horizon["1d"])), 9),
                "net_ret_7d_mean": round(float(np.mean(y_by_horizon["7d"])), 9),
            },
            "train_manifest_path": manifest_path,
            "train_report_hash": train_report_hash,
            "checkpoint_manifest_path": nn_manifest_path,
            "data_quality": dq,
            "dq_degraded": bool(dq_degraded),
            "dq_failed_reasons": dq_failed_reasons,
            "sampling_strategy": train_manifest.get("sampling_strategy", {}),
            "data_mode": str(self.train_data_mode),
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
