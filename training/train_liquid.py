#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from artifacts.pack import pack_model_artifact
from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, SCHEMA_HASH
from models.multimodal_gate import ResidualGateHead
from models.patchtst import PatchTSTBackbone
from training.datasets.liquid_sequence_dataset import LiquidSequenceDataset, load_training_samples


@dataclass
class TrainConfig:
    db_url: str
    symbols: List[str]
    start: datetime
    end: datetime
    lookback: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    d_model: int
    n_layers: int
    dropout: float
    max_samples_per_symbol: int
    out_dir: Path
    model_id: str


class LiquidSequenceModel(nn.Module):
    def __init__(self, *, lookback: int, feature_dim: int, d_model: int, n_layers: int, dropout: float):
        super().__init__()
        self.backbone = PatchTSTBackbone(
            feature_dim=feature_dim,
            lookback=lookback,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.text_indices = [FEATURE_INDEX[k] for k in sorted([k for k in FEATURE_INDEX if k.startswith("text_emb_")])]
        self.quality_indices = [FEATURE_INDEX[k] for k in ("text_item_count", "text_unique_authors", "text_dup_ratio", "text_disagreement", "text_avg_lag_sec", "text_coverage") if k in FEATURE_INDEX]
        self.head = ResidualGateHead(hidden_dim=d_model, text_dim=len(self.text_indices), quality_dim=len(self.quality_indices), out_dim=4)

    def forward(self, x_values: torch.Tensor, x_mask: torch.Tensor):
        h = self.backbone(x_values, x_mask)
        last_values = x_values[:, -1, :]
        text_vec = last_values[:, self.text_indices]
        quality_vec = last_values[:, self.quality_indices]
        return self.head(h, text_vec, quality_vec)


def _parse_dt(raw: str) -> datetime:
    text = str(raw or "").strip().replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _evaluate(model: LiquidSequenceModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    gates: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xv = batch["x_values"].to(device)
            xm = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            pred, gate = model(xv, xm)
            preds.append(_to_numpy(pred))
            ys.append(_to_numpy(y))
            gates.append(_to_numpy(gate))

    if not preds:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "hit_rate": 0.0,
            "sharpe_proxy": 0.0,
            "gate_mean": 0.0,
            "gate_q10": 0.0,
            "gate_q50": 0.0,
            "gate_q90": 0.0,
        }

    p = np.concatenate(preds, axis=0)
    y = np.concatenate(ys, axis=0)
    g = np.concatenate(gates, axis=0).reshape(-1)

    err = p - y
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    hit = float(np.mean((np.sign(p) == np.sign(y)).astype(np.float32)))

    # cost-after proxy on 1h horizon target
    pnl = np.sign(p[:, 0]) * y[:, 0]
    sharpe_proxy = float(np.mean(pnl) / max(1e-9, np.std(pnl)))

    return {
        "mse": mse,
        "mae": mae,
        "hit_rate": hit,
        "sharpe_proxy": sharpe_proxy,
        "gate_mean": float(np.mean(g)),
        "gate_q10": float(np.quantile(g, 0.1)),
        "gate_q50": float(np.quantile(g, 0.5)),
        "gate_q90": float(np.quantile(g, 0.9)),
    }


def _build_config() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train strict liquid sequence model")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--start", default=os.getenv("LIQUID_TRAIN_START", "2025-01-01T00:00:00Z"))
    ap.add_argument("--end", default=os.getenv("LIQUID_TRAIN_END", ""))
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "96")))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("LIQUID_EPOCHS", "6")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("LIQUID_BATCH", "32")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LIQUID_LR", "1e-3")))
    ap.add_argument("--weight-decay", type=float, default=float(os.getenv("LIQUID_WD", "1e-4")))
    ap.add_argument("--d-model", type=int, default=int(os.getenv("LIQUID_D_MODEL", "128")))
    ap.add_argument("--n-layers", type=int, default=int(os.getenv("LIQUID_N_LAYERS", "2")))
    ap.add_argument("--dropout", type=float, default=float(os.getenv("LIQUID_DROPOUT", "0.1")))
    ap.add_argument("--max-samples-per-symbol", type=int, default=int(os.getenv("LIQUID_MAX_SAMPLES_PER_SYMBOL", "0")))
    ap.add_argument("--out-dir", default=os.getenv("LIQUID_ARTIFACT_DIR", "artifacts/models/liquid_main"))
    ap.add_argument("--model-id", default=os.getenv("LIQUID_MODEL_ID", "liquid_main"))
    args = ap.parse_args()

    end = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    return TrainConfig(
        db_url=str(args.database_url),
        symbols=[s.strip().upper() for s in str(args.symbols).split(",") if s.strip()],
        start=_parse_dt(args.start),
        end=end,
        lookback=max(8, int(args.lookback)),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        d_model=max(32, int(args.d_model)),
        n_layers=max(1, int(args.n_layers)),
        dropout=float(args.dropout),
        max_samples_per_symbol=max(0, int(args.max_samples_per_symbol)),
        out_dir=Path(args.out_dir),
        model_id=str(args.model_id),
    )


def main() -> int:
    cfg = _build_config()
    samples = load_training_samples(
        db_url=cfg.db_url,
        symbols=cfg.symbols,
        start_ts=cfg.start,
        end_ts=cfg.end,
        lookback=cfg.lookback,
        max_samples_per_symbol=cfg.max_samples_per_symbol,
    )
    if len(samples) < 64:
        raise RuntimeError(f"insufficient_samples:{len(samples)}")

    split = int(len(samples) * 0.8)
    train_ds = LiquidSequenceDataset(samples[:split])
    oos_ds = LiquidSequenceDataset(samples[split:])

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    oos_loader = DataLoader(oos_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiquidSequenceModel(
        lookback=cfg.lookback,
        feature_dim=FEATURE_DIM,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for _epoch in range(cfg.epochs):
        for batch in train_loader:
            xv = batch["x_values"].to(device)
            xm = batch["x_mask"].to(device)
            y = batch["y"].to(device)
            pred, _gate = model(xv, xm)
            loss = loss_fn(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

    train_metrics = _evaluate(model, train_loader, device)
    oos_metrics = _evaluate(model, oos_loader, device)

    state = {
        "state_dict": model.state_dict(),
        "lookback": int(cfg.lookback),
        "feature_dim": int(FEATURE_DIM),
        "schema_hash": str(SCHEMA_HASH),
        "text_indices": model.text_indices,
        "quality_indices": model.quality_indices,
        "d_model": int(cfg.d_model),
        "n_layers": int(cfg.n_layers),
        "dropout": float(cfg.dropout),
    }

    metrics_summary = {
        "train": train_metrics,
        "oos": oos_metrics,
    }

    manifest = pack_model_artifact(
        model_dir=cfg.out_dir,
        model_id=cfg.model_id,
        state_dict=state,
        schema_path="schema/liquid_feature_schema.yaml",
        lookback=cfg.lookback,
        feature_dim=FEATURE_DIM,
        data_version="main",
        metrics_summary=metrics_summary,
        extra_payload={
            "train_samples": len(train_ds),
            "oos_samples": len(oos_ds),
            "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "oos_cost_after_metrics": oos_metrics,
            "gate_distribution": {
                "mean": oos_metrics["gate_mean"],
                "q10": oos_metrics["gate_q10"],
                "q50": oos_metrics["gate_q50"],
                "q90": oos_metrics["gate_q90"],
            },
        },
    )

    print(json.dumps({"status": "ok", "manifest": manifest, "schema_hash": SCHEMA_HASH}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
