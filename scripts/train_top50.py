#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from features.feature_contract import FEATURE_DIM, SCHEMA_HASH
from mlops_artifacts.pack import pack_model_artifact
from models.liquid_model import DEFAULT_HORIZONS, DEFAULT_QUANTILES, LiquidModel, LiquidModelConfig
from training.calibration.calibrate import build_calibration_bundle
from training.datasets.liquid_panel_cache_dataset import LiquidPanelCacheDataset
from training.losses.trading_losses import compose_liquid_loss


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train top50 panel model with cache-only pipeline")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", default=os.getenv("LIQUID_ARTIFACT_DIR", "artifacts/models/liquid_main"))
    ap.add_argument("--model-id", default=os.getenv("LIQUID_MODEL_ID", "liquid_main"))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("LIQUID_EPOCHS", "4")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("LIQUID_BATCH", "64")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LIQUID_LR", "1e-3")))
    ap.add_argument("--weight-decay", type=float, default=float(os.getenv("LIQUID_WD", "1e-4")))
    ap.add_argument("--d-model", type=int, default=int(os.getenv("LIQUID_D_MODEL", "128")))
    ap.add_argument("--n-layers", type=int, default=int(os.getenv("LIQUID_N_LAYERS", "2")))
    ap.add_argument("--n-heads", type=int, default=int(os.getenv("LIQUID_N_HEADS", "4")))
    ap.add_argument("--dropout", type=float, default=float(os.getenv("LIQUID_DROPOUT", "0.1")))
    ap.add_argument("--patch-len", type=int, default=int(os.getenv("LIQUID_PATCH_LEN", "16")))
    ap.add_argument("--backbone", choices=["patchtst", "itransformer", "tft"], default=os.getenv("LIQUID_BACKBONE", "patchtst"))
    # Trend setup default: 5m bars * 2016 ~= 7 days context.
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "2016")))
    ap.add_argument("--num-workers", type=int, default=int(os.getenv("LIQUID_DATALOADER_WORKERS", "2")))
    ap.add_argument("--prefetch-factor", type=int, default=int(os.getenv("LIQUID_DATALOADER_PREFETCH", "2")))
    ap.add_argument("--pin-memory", type=int, default=int(os.getenv("LIQUID_DATALOADER_PIN_MEMORY", "1")))
    ap.add_argument("--cost-profile", default=os.getenv("LIQUID_COST_PROFILE", "standard"))
    ap.add_argument("--sample-stride-buckets", type=int, default=int(os.getenv("LIQUID_SAMPLE_STRIDE_BUCKETS", "3")))
    ap.add_argument("--max-samples-per-symbol", type=int, default=int(os.getenv("LIQUID_MAX_SAMPLES_PER_SYMBOL", "0")))
    ap.add_argument("--balanced-sampling", type=int, default=int(os.getenv("LIQUID_BALANCED_SAMPLING", "1")))
    ap.add_argument("--primary-timeframe", default=os.getenv("LIQUID_PRIMARY_TIMEFRAME", "5m"))
    return ap.parse_args()


def _build_model(cfg: argparse.Namespace, symbol_count: int, regime_dim: int) -> LiquidModel:
    model_cfg = LiquidModelConfig(
        backbone_name=str(cfg.backbone),
        lookback=int(cfg.lookback),
        feature_dim=FEATURE_DIM,
        d_model=int(cfg.d_model),
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        dropout=float(cfg.dropout),
        patch_len=int(cfg.patch_len),
        horizons=list(DEFAULT_HORIZONS),
        quantiles=list(DEFAULT_QUANTILES),
        text_indices=[],
        quality_indices=[],
        num_symbols=max(1, int(symbol_count)),
        symbol_emb_dim=16,
        regime_dim=max(1, int(regime_dim)),
        sparse_topk=2,
    )
    return LiquidModel(model_cfg)


def _loader(ds: LiquidPanelCacheDataset, indices: np.ndarray, cfg: argparse.Namespace, shuffle: bool) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": Subset(ds, indices.tolist()),
        "batch_size": int(cfg.batch_size),
        "shuffle": bool(shuffle),
        "drop_last": False,
        "num_workers": max(0, int(cfg.num_workers)),
        "pin_memory": bool(int(cfg.pin_memory)),
    }
    if int(cfg.num_workers) > 0:
        kwargs["prefetch_factor"] = max(2, int(cfg.prefetch_factor))
    return DataLoader(**kwargs)


def _train_epoch(model: LiquidModel, loader: DataLoader, device: torch.device, opt: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    logs: Dict[str, List[float]] = {
        "total": [],
        "student_t_nll": [],
        "quantile": [],
        "direction": [],
        "load_balance": [],
        "router_entropy": [],
        "horizon_smoothness": [],
        "vol_monotonic": [],
    }
    seen = 0
    started = time.perf_counter()
    for batch in loader:
        xv = batch["x_values"].to(device)
        xm = batch["x_mask"].to(device)
        sid = batch["symbol_id"].to(device)
        regime = batch["regime_features"].to(device)
        regime_m = batch["regime_mask"].to(device)
        y = batch["y_net"].to(device)
        cost_bps = batch["cost_bps"].to(device)
        out = model(xv, xm, symbol_id=sid, regime_features=regime, regime_mask=regime_m)
        losses = compose_liquid_loss(
            mu=out.mu,
            log_sigma=out.log_sigma,
            q=out.q,
            direction_logit=out.direction_logit,
            gate=out.aux.get("gate") if isinstance(out.aux, dict) else None,
            df=out.df,
            expert_weights=out.expert_weights,
            y=y,
            cost_bps=cost_bps,
            quantiles=list(DEFAULT_QUANTILES),
            w_nll=1.0,
            w_quantile=0.3,
            w_direction=0.2,
            w_gate=0.05,
            w_calibration=0.05,
            w_load_balance=0.02,
            w_router_entropy=0.01,
            w_horizon_smoothness=0.02,
            w_vol_monotonic=0.02,
        )
        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        seen += int(xv.shape[0])
        for k in logs:
            logs[k].append(float(losses.get(k, losses["total"]).detach().cpu().item()))
    elapsed = max(1e-9, time.perf_counter() - started)
    out = {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}
    out["samples_per_sec"] = float(seen / elapsed)
    out["samples"] = int(seen)
    return out


def _evaluate_router(model: LiquidModel, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    ws: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x_values"].to(device),
                batch["x_mask"].to(device),
                symbol_id=batch["symbol_id"].to(device),
                regime_features=batch["regime_features"].to(device),
                regime_mask=batch["regime_mask"].to(device),
            )
            if out.expert_weights is not None:
                ws.append(out.expert_weights.detach().cpu().numpy())
    if not ws:
        raise RuntimeError("router_weights_missing")
    w = np.concatenate(ws, axis=0)
    usage = np.mean(w, axis=0)
    entropy = float(np.mean(-np.sum(np.clip(w, 1e-8, 1.0) * np.log(np.clip(w, 1e-8, 1.0)), axis=1)))
    collapse = float(np.max(usage))
    if collapse >= 0.95:
        raise RuntimeError(f"router_collapse_detected:{collapse:.6f}")
    return {"usage": usage.tolist(), "entropy": entropy, "collapse_max": collapse}


def _collect_for_calibration(model: LiquidModel, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    out_mu: List[np.ndarray] = []
    out_sigma: List[np.ndarray] = []
    out_dir: List[np.ndarray] = []
    out_y: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x_values"].to(device),
                batch["x_mask"].to(device),
                symbol_id=batch["symbol_id"].to(device),
                regime_features=batch["regime_features"].to(device),
                regime_mask=batch["regime_mask"].to(device),
            )
            out_mu.append(out.mu.detach().cpu().numpy())
            out_sigma.append(torch.exp(out.log_sigma).detach().cpu().numpy())
            if out.direction_logit is not None:
                out_dir.append(out.direction_logit.detach().cpu().numpy())
            out_y.append(batch["y_net"].detach().cpu().numpy())
    if not out_mu or not out_sigma or not out_y:
        raise RuntimeError("calibration_validation_pack_empty")
    mu = np.concatenate(out_mu, axis=0)
    sigma = np.concatenate(out_sigma, axis=0)
    y = np.concatenate(out_y, axis=0)
    direction_logit = np.concatenate(out_dir, axis=0) if out_dir else np.zeros_like(mu, dtype=np.float32)
    return {"mu": mu, "sigma": sigma, "y": y, "direction_logit": direction_logit}


def _select_indices(ds: LiquidPanelCacheDataset, cfg: argparse.Namespace) -> np.ndarray:
    stride = max(1, int(cfg.sample_stride_buckets))
    cap = max(0, int(cfg.max_samples_per_symbol))
    keep: List[int] = []
    counts: Dict[int, int] = {}
    for i, ref in enumerate(ds._refs):  # pylint: disable=protected-access
        if int(ref.t_idx) % stride != 0:
            continue
        sid = int(ref.symbol_id)
        if cap > 0 and int(counts.get(sid, 0)) >= cap:
            continue
        keep.append(i)
        counts[sid] = int(counts.get(sid, 0)) + 1
    if bool(int(cfg.balanced_sampling)) and keep:
        min_count = min(counts.values()) if counts else 0
        if min_count > 0:
            used: Dict[int, int] = {}
            out: List[int] = []
            for i in keep:
                sid = int(ds._refs[i].symbol_id)  # pylint: disable=protected-access
                if int(used.get(sid, 0)) >= int(min_count):
                    continue
                out.append(i)
                used[sid] = int(used.get(sid, 0)) + 1
            keep = out
    return np.asarray(keep, dtype=np.int64)


def main() -> int:
    cfg = _parse_args()
    ds = LiquidPanelCacheDataset(
        cache_dir=cfg.cache_dir,
        lookback=int(cfg.lookback),
        horizons=DEFAULT_HORIZONS,
        cost_profile_name=str(cfg.cost_profile),
        require_cache=True,
    )
    primary_tf = str(cfg.primary_timeframe).strip().lower() or "5m"
    cache_bar = str(ds.manifest.get("bar_size") or "").strip().lower()
    if cache_bar != primary_tf:
        raise RuntimeError(f"primary_timeframe_mismatch:{cache_bar}:{primary_tf}")
    selected = _select_indices(ds, cfg)
    if selected.size == 0:
        raise RuntimeError("selected_samples_empty")
    n = int(selected.size)
    if n < 512:
        raise RuntimeError(f"insufficient_cache_samples:{n}")
    cut = int(n * 0.8)
    train_idx = selected[:cut]
    val_idx = selected[cut:]
    if val_idx.size == 0:
        raise RuntimeError("validation_split_empty")

    tr = _loader(ds, train_idx, cfg, shuffle=True)
    va = _loader(ds, val_idx, cfg, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    regime_dim = int(len(ds.manifest.get("regime_feature_names") or []) + len(ds.manifest.get("multi_tf_feature_names") or []))
    model = _build_model(cfg, symbol_count=len(ds.symbols), regime_dim=regime_dim)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    train_trace: List[Dict[str, Any]] = []
    for _ in range(max(1, int(cfg.epochs))):
        train_trace.append(_train_epoch(model, tr, device, opt))
    router_report = _evaluate_router(model, va, device)
    cal_pack = _collect_for_calibration(model, va, device)
    calibration = build_calibration_bundle(
        direction_logit=cal_pack["direction_logit"],
        y_net=cal_pack["y"],
        mu=cal_pack["mu"],
        sigma=cal_pack["sigma"],
    )

    report = {
        "status": "ok",
        "cache_dir": str(cfg.cache_dir),
        "cache_hash": ds.manifest.get("cache_hash"),
        "universe_snapshot_hash": ds.manifest.get("universe_snapshot_hash"),
        "feature_contract_hash": ds.manifest.get("feature_contract_hash"),
        "symbol_count": len(ds.symbols),
        "sample_count": int(n),
        "sample_stride_buckets": int(cfg.sample_stride_buckets),
        "max_samples_per_symbol": int(cfg.max_samples_per_symbol),
        "balanced_sampling": bool(int(cfg.balanced_sampling)),
        "train_trace": train_trace,
        "router_report": router_report,
        "cost_profile": str(cfg.cost_profile),
        "calibration": calibration,
        "primary_timeframe": primary_tf,
        "context_timeframes": list(ds.manifest.get("context_timeframes") or []),
    }

    state = {
        "state_dict": model.state_dict(),
        "schema_hash": str(SCHEMA_HASH),
        "cost_profile": str(cfg.cost_profile),
        "symbol_to_id": dict(ds.symbol_to_id),
        "cache_hash": ds.manifest.get("cache_hash"),
        "universe_snapshot_hash": ds.manifest.get("universe_snapshot_hash"),
        "context_timeframes": list(ds.manifest.get("context_timeframes") or []),
        "calibration": calibration,
        **model.export_meta(),
    }
    out_dir = Path(str(cfg.out_dir))
    manifest = pack_model_artifact(
        model_dir=out_dir,
        model_id=str(cfg.model_id),
        state_dict=state,
        schema_path="schema/liquid_feature_schema.yaml",
        lookback=int(cfg.lookback),
        feature_dim=FEATURE_DIM,
        data_version="main",
        metrics_summary={"train": {"sample_count": n}},
        extra_payload=report,
    )
    # Persist schema snapshot used by panel training.
    (out_dir / "schema_snapshot_ext.json").write_text(
        json.dumps(
            {
                "feature_contract_hash": ds.manifest.get("feature_contract_hash"),
                "universe_snapshot_hash": ds.manifest.get("universe_snapshot_hash"),
                "symbol_to_id": ds.symbol_to_id,
                "cache_hash": ds.manifest.get("cache_hash"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "manifest": manifest, "router_report": router_report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
