#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from inference.liquid_feature_contract import LIQUID_FEATURE_KEYS
from validation import (
    evaluate_regression_oos,
    purged_kfold_slices,
    resolve_validation_protocol,
    summarize_fold_metrics,
    validation_protocol_to_dict,
    walk_forward_slices,
)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except Exception:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    HAS_TORCH = False


@dataclass
class ExperimentConfig:
    database_url: str
    start: datetime
    end: datetime
    symbols: List[str]
    lookback_steps: int
    horizon_steps: int
    max_samples: int
    backbones: List[str]
    l2: float
    seed: int
    epochs: int
    batch_size: int
    lr: float
    out: str


SUPPORTED_BACKBONES = {"ridge", "itransformer", "patchtst", "tft"}
BACKBONE_ALIASES = {
    "tftlite": "tft",
    "tft_lite": "tft",
    "temporal_fusion_transformer": "tft",
}


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_backbones(backbones: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in backbones:
        cur = str(raw or "").strip().lower()
        if not cur:
            continue
        cur = BACKBONE_ALIASES.get(cur, cur)
        if cur not in SUPPORTED_BACKBONES:
            continue
        if cur in seen:
            continue
        seen.add(cur)
        out.append(cur)
    return out


def _load_rows(
    db_url: str,
    start_dt: datetime,
    end_dt: datetime,
    symbols: Sequence[str],
) -> List[Dict[str, object]]:
    with psycopg2.connect(db_url, cursor_factory=RealDictCursor) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, as_of_ts, features
                FROM feature_matrix_main
                WHERE as_of_ts >= %s AND as_of_ts <= %s
                  AND symbol = ANY(%s)
                ORDER BY symbol ASC, as_of_ts ASC
                """,
                (start_dt, end_dt, list(symbols)),
            )
            return [dict(r) for r in cur.fetchall()]


def build_sequence_dataset(
    rows: Sequence[Dict[str, object]],
    *,
    lookback_steps: int,
    horizon_steps: int,
    feature_keys: Sequence[str] = LIQUID_FEATURE_KEYS,
    max_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    by_symbol: Dict[str, List[Tuple[datetime, Dict[str, float]]]] = {}
    for row in rows:
        sym = str(row.get("symbol") or "").upper().strip()
        ts = row.get("as_of_ts")
        payload = row.get("features") if isinstance(row.get("features"), dict) else {}
        if not sym or not isinstance(ts, datetime):
            continue
        by_symbol.setdefault(sym, []).append((ts, payload))

    L = max(2, int(lookback_steps))
    H = max(1, int(horizon_steps))
    X_seq: List[np.ndarray] = []
    y: List[float] = []
    for _, seq in by_symbol.items():
        if len(seq) < L + H:
            continue
        for i in range(L - 1, len(seq) - H):
            window = seq[i - L + 1 : i + 1]
            tgt_payload = seq[i + H][1]
            mat = np.array(
                [
                    [float(payload.get(k, 0.0) or 0.0) for k in feature_keys]
                    for (_, payload) in window
                ],
                dtype=np.float32,
            )
            X_seq.append(mat)
            y.append(float(tgt_payload.get("ret_1", 0.0) or 0.0))
    if not X_seq:
        return (
            np.zeros((0, L, len(feature_keys)), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    X = np.stack(X_seq, axis=0).astype(np.float32)
    yy = np.array(y, dtype=np.float32)
    if int(max_samples) > 0 and X.shape[0] > int(max_samples):
        idx = np.linspace(0, X.shape[0] - 1, int(max_samples)).astype(np.int64)
        X = X[idx]
        yy = yy[idx]
    return X, yy


def _standardize_by_train(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(X_train, axis=(0, 1), keepdims=True)
    sigma = np.clip(np.std(X_train, axis=(0, 1), keepdims=True), 1e-6, None)
    return (X_train - mu) / sigma, (X_test - mu) / sigma


def _fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    l2: float,
) -> np.ndarray:
    xtr = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    xte = X_test.reshape(X_test.shape[0], -1).astype(np.float64)
    d = xtr.shape[1]
    w = np.linalg.solve(xtr.T @ xtr + float(l2) * np.eye(d), xtr.T @ y_train.astype(np.float64))
    return (xte @ w).astype(np.float64)


if HAS_TORCH:
    class ITransformerLite(nn.Module):
        def __init__(self, lookback_steps: int, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.var_proj = nn.Linear(int(lookback_steps), int(d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(d_model) * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(int(d_model)),
                nn.Linear(int(d_model), int(d_model) // 2),
                nn.GELU(),
                nn.Linear(int(d_model) // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D] -> variable tokens [B, D, L]
            tokens = x.transpose(1, 2)
            h = self.var_proj(tokens)
            h = self.encoder(h)
            pooled = h.mean(dim=1)
            return self.head(pooled).squeeze(-1)


    class PatchTSTLite(nn.Module):
        def __init__(
            self,
            lookback_steps: int,
            n_features: int,
            patch_len: int = 8,
            patch_stride: int = 4,
            d_model: int = 64,
            n_heads: int = 4,
            n_layers: int = 2,
        ):
            super().__init__()
            p = max(2, min(int(patch_len), int(lookback_steps)))
            s = max(1, int(patch_stride))
            self.patch_len = p
            self.patch_stride = s
            self.patch_proj = nn.Linear(p, int(d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(d_model) * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(int(d_model)),
                nn.Linear(int(d_model), int(d_model) // 2),
                nn.GELU(),
                nn.Linear(int(d_model) // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D] -> [B, D, L]
            ch = x.transpose(1, 2)
            patches = ch.unfold(dimension=2, size=self.patch_len, step=self.patch_stride)
            # [B, D, Np, P] -> project P -> d_model, then aggregate channels
            tok = self.patch_proj(patches)
            tok = tok.mean(dim=1)
            h = self.encoder(tok)
            pooled = h.mean(dim=1)
            return self.head(pooled).squeeze(-1)


    class TFTLite(nn.Module):
        def __init__(self, lookback_steps: int, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
            super().__init__()
            self.input_proj = nn.Linear(int(n_features), int(d_model))
            self.temporal_gate = nn.Sequential(
                nn.Linear(int(d_model), int(d_model)),
                nn.GELU(),
                nn.Linear(int(d_model), int(d_model)),
                nn.Sigmoid(),
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=int(d_model),
                nhead=int(n_heads),
                dim_feedforward=int(d_model) * 2,
                dropout=0.1,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
            self.head = nn.Sequential(
                nn.LayerNorm(int(d_model)),
                nn.Linear(int(d_model), int(d_model) // 2),
                nn.GELU(),
                nn.Linear(int(d_model) // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, L, D]
            h = self.input_proj(x)
            g = self.temporal_gate(h)
            h = h * g
            h = self.encoder(h)
            pooled = h[:, -1, :]
            return self.head(pooled).squeeze(-1)


def _fit_predict_torch(
    backbone: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Tuple[np.ndarray, str]:
    if not HAS_TORCH:
        return np.zeros((X_test.shape[0],), dtype=np.float64), "torch_missing"
    if X_train.shape[0] < 32:
        return np.zeros((X_test.shape[0],), dtype=np.float64), "insufficient_train_rows"

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, D = X_train.shape
    if str(backbone) == "itransformer":
        model = ITransformerLite(lookback_steps=L, n_features=D)
    elif str(backbone) == "patchtst":
        model = PatchTSTLite(lookback_steps=L, n_features=D)
    elif str(backbone) == "tft":
        model = TFTLite(lookback_steps=L, n_features=D)
    else:
        return np.zeros((X_test.shape[0],), dtype=np.float64), "unsupported_backbone"
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=max(16, int(batch_size)), shuffle=True, drop_last=False)
    model.train()
    for _ in range(max(1, int(epochs))):
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X_test, dtype=torch.float32, device=device)
        pred = model(xt).detach().cpu().numpy().astype(np.float64)
    return pred, "ok"


def _fold_metrics(
    X: np.ndarray,
    y: np.ndarray,
    folds: Iterable[Tuple[np.ndarray, np.ndarray]],
    *,
    backbone: str,
    l2: float,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    min_train_points: int,
    min_test_points: int,
) -> Dict[str, object]:
    metrics: List[Dict[str, float]] = []
    basic: List[Dict[str, float]] = []
    blocked_reason = ""
    used_folds = 0
    for tr, te in folds:
        if tr.size < int(min_train_points) or te.size < int(min_test_points):
            continue
        Xtr_raw = X[tr]
        Xte_raw = X[te]
        Xtr, Xte = _standardize_by_train(Xtr_raw, Xte_raw)
        if backbone == "ridge":
            pred = _fit_predict_ridge(Xtr, y[tr], Xte, l2=float(l2))
            status = "ok"
        elif backbone in {"itransformer", "patchtst", "tft"}:
            pred, status = _fit_predict_torch(
                backbone,
                Xtr,
                y[tr],
                Xte,
                seed=int(seed) + int(used_folds),
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
            )
        else:
            pred = np.zeros((te.size,), dtype=np.float64)
            status = "unsupported_backbone"
        if status != "ok":
            blocked_reason = status
            break
        used_folds += 1
        cur_y = y[te].astype(np.float64)
        mse = float(np.mean((pred - cur_y) ** 2))
        mae = float(np.mean(np.abs(pred - cur_y)))
        hit_rate = float(np.mean(np.sign(pred) == np.sign(cur_y)))
        basic.append({"mse": mse, "mae": mae, "hit_rate": hit_rate})
        metrics.append(
            evaluate_regression_oos(
                y_true=cur_y,
                y_pred=pred,
                fee_bps=5.0,
                slippage_bps=3.0,
            )
        )
    if blocked_reason:
        return {
            "status": "blocked",
            "reason": blocked_reason,
            "folds": int(used_folds),
        }
    return {
        "status": "ok",
        "folds": int(used_folds),
        "basic": {
            "mse": float(np.mean([m["mse"] for m in basic])) if basic else 0.0,
            "mae": float(np.mean([m["mae"] for m in basic])) if basic else 0.0,
            "hit_rate": float(np.mean([m["hit_rate"] for m in basic])) if basic else 0.0,
        },
        "oos": summarize_fold_metrics(metrics),
    }


def run_experiment_suite(config: ExperimentConfig) -> Dict[str, object]:
    rows = _load_rows(config.database_url, config.start, config.end, config.symbols)
    X, y = build_sequence_dataset(
        rows,
        lookback_steps=config.lookback_steps,
        horizon_steps=config.horizon_steps,
        max_samples=config.max_samples,
    )
    if X.shape[0] < 512:
        raise RuntimeError("insufficient_rows_for_backbone_experiments")

    protocol = resolve_validation_protocol(prefix="LIQUID")
    wf = walk_forward_slices(
        n_samples=X.shape[0],
        train_window=protocol.wf_train_window,
        test_window=protocol.wf_test_window,
        purge_window=protocol.wf_purge_window,
        step_window=protocol.wf_step_window,
    )
    pkf = purged_kfold_slices(
        n_samples=X.shape[0],
        n_splits=protocol.pkf_splits,
        purge_window=protocol.pkf_purge_window,
    )

    results: List[Dict[str, object]] = []
    normalized = normalize_backbones(config.backbones)
    for backbone in normalized:
        wf_out = _fold_metrics(
            X,
            y,
            wf,
            backbone=backbone,
            l2=config.l2,
            seed=config.seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            min_train_points=protocol.min_train_points,
            min_test_points=protocol.min_test_points,
        )
        pkf_out = _fold_metrics(
            X,
            y,
            pkf,
            backbone=backbone,
            l2=config.l2,
            seed=config.seed + 1000,
            epochs=config.epochs,
            batch_size=config.batch_size,
            lr=config.lr,
            min_train_points=protocol.min_train_points,
            min_test_points=protocol.min_test_points,
        )
        results.append(
            {
                "backbone": backbone,
                "walk_forward": wf_out,
                "purged_kfold": pkf_out,
                "ready": bool(
                    wf_out.get("status") == "ok"
                    and int(wf_out.get("folds", 0)) >= int(protocol.wf_min_folds)
                ),
            }
        )

    ready_backbones = [r for r in results if bool(r.get("ready"))]
    out: Dict[str, object] = {
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "config": {
            "symbols": config.symbols,
            "lookback_steps": int(config.lookback_steps),
            "horizon_steps": int(config.horizon_steps),
            "max_samples": int(config.max_samples),
            "backbones": normalized,
            "seed": int(config.seed),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "lr": float(config.lr),
            "l2": float(config.l2),
        },
        "validation_protocol": validation_protocol_to_dict(protocol),
        "dataset": {
            "rows": int(X.shape[0]),
            "lookback_steps": int(X.shape[1]),
            "feature_dim": int(X.shape[2]),
        },
        "results": results,
        "ready_backbones": [str(r.get("backbone")) for r in ready_backbones],
        "supported_backbones": sorted(SUPPORTED_BACKBONES),
        "torch_available": bool(HAS_TORCH),
    }
    return out


def _config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    start = _parse_dt(args.start)
    end = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    symbols = [s.strip().upper() for s in str(args.symbols).split(",") if s.strip()]
    backbones = normalize_backbones([s.strip().lower() for s in str(args.backbones).split(",") if s.strip()])
    return ExperimentConfig(
        database_url=str(args.database_url),
        start=start,
        end=end,
        symbols=symbols,
        lookback_steps=max(2, int(args.lookback_steps)),
        horizon_steps=max(1, int(args.horizon_steps)),
        max_samples=max(0, int(args.max_samples)),
        backbones=backbones or ["ridge", "itransformer", "patchtst", "tft"],
        l2=float(args.l2),
        seed=int(args.seed),
        epochs=max(1, int(args.epochs)),
        batch_size=max(16, int(args.batch_size)),
        lr=float(args.lr),
        out=str(args.out),
    )


def run_experiment_suite_from_env() -> Dict[str, object]:
    cfg = ExperimentConfig(
        database_url=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"),
        start=_parse_dt(os.getenv("BACKBONE_EXP_START", "2018-01-01T00:00:00Z")),
        end=_parse_dt(os.getenv("BACKBONE_EXP_END", datetime.now(timezone.utc).isoformat())),
        symbols=[s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL").split(",") if s.strip()],
        lookback_steps=int(os.getenv("BACKBONE_EXP_LOOKBACK_STEPS", "32")),
        horizon_steps=int(os.getenv("MULTIMODAL_TARGET_HORIZON_STEPS", "1")),
        max_samples=int(os.getenv("BACKBONE_EXP_MAX_SAMPLES", "30000")),
        backbones=normalize_backbones([s.strip().lower() for s in os.getenv("BACKBONE_EXP_BACKBONES", "ridge,itransformer,patchtst,tft").split(",") if s.strip()]),
        l2=float(os.getenv("MULTIMODAL_L2", "0.05")),
        seed=int(os.getenv("TRAIN_SEED", "42")),
        epochs=int(os.getenv("BACKBONE_EXP_EPOCHS", "6")),
        batch_size=int(os.getenv("BACKBONE_EXP_BATCH_SIZE", "256")),
        lr=float(os.getenv("BACKBONE_EXP_LR", "0.001")),
        out=os.getenv("BACKBONE_EXP_OUT", "artifacts/experiments/backbone_suite_latest.json"),
    )
    out = run_experiment_suite(cfg)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unified liquid backbone experiments (ridge/itransformer/patchtst/tft)")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    parser.add_argument("--start", default="2018-01-01T00:00:00Z")
    parser.add_argument("--end", default="")
    parser.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    parser.add_argument("--lookback-steps", type=int, default=int(os.getenv("BACKBONE_EXP_LOOKBACK_STEPS", "32")))
    parser.add_argument("--horizon-steps", type=int, default=int(os.getenv("MULTIMODAL_TARGET_HORIZON_STEPS", "1")))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("BACKBONE_EXP_MAX_SAMPLES", "30000")))
    parser.add_argument("--backbones", default=os.getenv("BACKBONE_EXP_BACKBONES", "ridge,itransformer,patchtst,tft"))
    parser.add_argument("--l2", type=float, default=float(os.getenv("MULTIMODAL_L2", "0.05")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("TRAIN_SEED", "42")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("BACKBONE_EXP_EPOCHS", "6")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BACKBONE_EXP_BATCH_SIZE", "256")))
    parser.add_argument("--lr", type=float, default=float(os.getenv("BACKBONE_EXP_LR", "0.001")))
    parser.add_argument("--out", default=os.getenv("BACKBONE_EXP_OUT", "artifacts/experiments/backbone_suite_latest.json"))
    args = parser.parse_args()

    cfg = _config_from_args(args)
    out = run_experiment_suite(cfg)
    out_path = Path(cfg.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "out": str(out_path),
                "rows": int(out["dataset"]["rows"]),
                "ready_backbones": out.get("ready_backbones", []),
                "torch_available": bool(out.get("torch_available", False)),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
