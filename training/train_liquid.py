#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from backend.v2_repository import V2Repository
from mlops_artifacts.pack import pack_model_artifact
from features.feature_contract import FEATURE_DIM, FEATURE_INDEX, SCHEMA_HASH
from models.liquid_model import DEFAULT_HORIZONS, DEFAULT_QUANTILES, LiquidModel, LiquidModelConfig
from training.calibration.calibrate import build_calibration_bundle
from training.datasets.liquid_sequence_dataset import HORIZONS, LiquidSequenceDataset, load_training_samples
from training.losses.trading_losses import compose_liquid_loss
from training.metrics.liquid_metrics import evaluate_liquid_metrics
from training.splits.walkforward_purged import WalkForwardPurgedConfig, build_walkforward_purged_splits


@dataclass
class TrainConfig:
    db_url: str
    symbols: List[str]
    universe_track: str
    use_universe_snapshot: bool
    start: datetime
    end: datetime
    lookback: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    d_model: int
    n_layers: int
    n_heads: int
    dropout: float
    patch_len: int
    backbone: str
    max_samples_per_symbol: int
    out_dir: Path
    model_id: str
    cost_profile: str
    train_days: int
    val_days: int
    test_days: int
    purge_gap_hours: int
    step_days: int
    force_purged: bool


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


def _build_cfg() -> TrainConfig:
    ap = argparse.ArgumentParser(description="Train strict liquid model with walk-forward purged validation")
    ap.add_argument("--database-url", default=os.getenv("DATABASE_URL", "postgresql://monitor@localhost:5432/monitor"))
    ap.add_argument("--symbols", default=os.getenv("LIQUID_SYMBOLS", "BTC,ETH,SOL"))
    ap.add_argument("--universe-track", default=os.getenv("LIQUID_UNIVERSE_TRACK", "liquid"))
    ap.add_argument("--use-universe-snapshot", type=int, default=int(os.getenv("LIQUID_USE_UNIVERSE_SNAPSHOT", "1")))
    ap.add_argument("--start", default=os.getenv("LIQUID_TRAIN_START", "2025-01-01T00:00:00Z"))
    ap.add_argument("--end", default=os.getenv("LIQUID_TRAIN_END", ""))
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LIQUID_LOOKBACK", "96")))
    ap.add_argument("--epochs", type=int, default=int(os.getenv("LIQUID_EPOCHS", "6")))
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("LIQUID_BATCH", "32")))
    ap.add_argument("--lr", type=float, default=float(os.getenv("LIQUID_LR", "1e-3")))
    ap.add_argument("--weight-decay", type=float, default=float(os.getenv("LIQUID_WD", "1e-4")))
    ap.add_argument("--d-model", type=int, default=int(os.getenv("LIQUID_D_MODEL", "128")))
    ap.add_argument("--n-layers", type=int, default=int(os.getenv("LIQUID_N_LAYERS", "2")))
    ap.add_argument("--n-heads", type=int, default=int(os.getenv("LIQUID_N_HEADS", "4")))
    ap.add_argument("--dropout", type=float, default=float(os.getenv("LIQUID_DROPOUT", "0.1")))
    ap.add_argument("--patch-len", type=int, default=int(os.getenv("LIQUID_PATCH_LEN", "16")))
    ap.add_argument("--backbone", choices=["patchtst", "itransformer", "tft"], default=os.getenv("LIQUID_BACKBONE", "patchtst"))
    ap.add_argument("--max-samples-per-symbol", type=int, default=int(os.getenv("LIQUID_MAX_SAMPLES_PER_SYMBOL", "0")))
    ap.add_argument("--out-dir", default=os.getenv("LIQUID_ARTIFACT_DIR", "artifacts/models/liquid_main"))
    ap.add_argument("--model-id", default=os.getenv("LIQUID_MODEL_ID", "liquid_main"))
    ap.add_argument("--cost-profile", default=os.getenv("LIQUID_COST_PROFILE", "standard"))
    ap.add_argument("--train-days", type=int, default=int(os.getenv("LIQUID_WF_TRAIN_DAYS", "60")))
    ap.add_argument("--val-days", type=int, default=int(os.getenv("LIQUID_WF_VAL_DAYS", "7")))
    ap.add_argument("--test-days", type=int, default=int(os.getenv("LIQUID_WF_TEST_DAYS", "7")))
    ap.add_argument("--purge-gap-hours", type=int, default=int(os.getenv("LIQUID_WF_PURGE_GAP_HOURS", "24")))
    ap.add_argument("--step-days", type=int, default=int(os.getenv("LIQUID_WF_STEP_DAYS", "7")))
    ap.add_argument("--force-purged", action="store_true", default=str(os.getenv("LIQUID_FORCE_PURGED", "1")).lower() in {"1", "true", "yes", "on"})
    args = ap.parse_args()

    end = _parse_dt(args.end) if str(args.end).strip() else datetime.now(timezone.utc)
    return TrainConfig(
        db_url=str(args.database_url),
        symbols=[s.strip().upper() for s in str(args.symbols).split(",") if s.strip()],
        universe_track=str(args.universe_track).strip().lower() or "liquid",
        use_universe_snapshot=bool(int(args.use_universe_snapshot)),
        start=_parse_dt(args.start),
        end=end,
        lookback=max(8, int(args.lookback)),
        epochs=max(1, int(args.epochs)),
        batch_size=max(1, int(args.batch_size)),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        d_model=max(32, int(args.d_model)),
        n_layers=max(1, int(args.n_layers)),
        n_heads=max(1, int(args.n_heads)),
        dropout=float(args.dropout),
        patch_len=max(2, int(args.patch_len)),
        backbone=str(args.backbone),
        max_samples_per_symbol=max(0, int(args.max_samples_per_symbol)),
        out_dir=Path(args.out_dir),
        model_id=str(args.model_id),
        cost_profile=str(args.cost_profile),
        train_days=max(1, int(args.train_days)),
        val_days=max(1, int(args.val_days)),
        test_days=max(1, int(args.test_days)),
        purge_gap_hours=max(0, int(args.purge_gap_hours)),
        step_days=max(1, int(args.step_days)),
        force_purged=bool(args.force_purged),
    )


def _normalize_symbols(symbols: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in symbols:
        sym = str(raw or "").strip().upper()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    out.sort()
    return out


def _resolve_training_universe(cfg: TrainConfig) -> Dict[str, Any]:
    fallback = _normalize_symbols(cfg.symbols)
    if not fallback:
        raise RuntimeError("empty_fallback_symbols")
    if not bool(cfg.use_universe_snapshot):
        return {
            "track": str(cfg.universe_track),
            "as_of": cfg.start.isoformat(),
            "symbols": fallback,
            "source": "cli_symbols",
            "universe_version": "manual",
            "snapshot_at": None,
        }

    repo = V2Repository(cfg.db_url)
    resolved = repo.resolve_asset_universe_asof(
        track=str(cfg.universe_track),
        as_of=cfg.start,
        fallback_targets=fallback,
    )
    symbols = _normalize_symbols(list(resolved.get("symbols") or []))
    if not symbols:
        raise RuntimeError("resolved_universe_empty")
    universe_version = str(resolved.get("universe_version") or "runtime_resolved")
    source = str(resolved.get("source") or "snapshot")
    repo.upsert_asset_universe_snapshot(
        track=str(cfg.universe_track),
        as_of=cfg.start,
        symbols=symbols,
        universe_version=universe_version,
        source=source,
    )
    return {
        "track": str(cfg.universe_track),
        "as_of": cfg.start.isoformat(),
        "symbols": symbols,
        "source": source,
        "universe_version": universe_version,
        "snapshot_at": resolved.get("snapshot_at"),
    }


def _attach_universe_to_training_report(training_report: Dict[str, Any], universe_meta: Dict[str, Any]) -> None:
    training_report["universe"] = {
        "track": str(universe_meta.get("track") or "liquid"),
        "source": str(universe_meta.get("source") or "unknown"),
        "universe_version": str(universe_meta.get("universe_version") or "unknown"),
        "snapshot_at": universe_meta.get("snapshot_at"),
        "as_of": universe_meta.get("as_of"),
        "symbols": list(universe_meta.get("symbols") or []),
    }


def _build_model(cfg: TrainConfig) -> LiquidModel:
    text_indices = [FEATURE_INDEX[k] for k in sorted(k for k in FEATURE_INDEX if k.startswith("text_emb_"))]
    quality_indices = [
        FEATURE_INDEX[k]
        for k in ("text_item_count", "text_unique_authors", "text_dup_ratio", "text_disagreement", "text_avg_lag_sec", "text_coverage")
        if k in FEATURE_INDEX
    ]
    model_cfg = LiquidModelConfig(
        backbone_name=cfg.backbone,
        lookback=cfg.lookback,
        feature_dim=FEATURE_DIM,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
        patch_len=cfg.patch_len,
        horizons=list(DEFAULT_HORIZONS),
        quantiles=list(DEFAULT_QUANTILES),
        text_indices=text_indices,
        quality_indices=quality_indices,
    )
    return LiquidModel(model_cfg)


def _train_one_epoch(model: LiquidModel, loader: DataLoader, device: torch.device, opt: torch.optim.Optimizer) -> Dict[str, float]:
    model.train()
    logs: Dict[str, List[float]] = {"total": [], "gaussian_nll": [], "quantile": [], "direction": [], "gate": []}
    for batch in loader:
        xv = batch["x_values"].to(device)
        xm = batch["x_mask"].to(device)
        y = batch["y_net"].to(device)
        cost_bps = batch["cost_bps"].to(device)

        out = model(xv, xm)
        losses = compose_liquid_loss(
            mu=out.mu,
            log_sigma=out.log_sigma,
            q=out.q,
            direction_logit=out.direction_logit,
            gate=out.aux.get("gate") if isinstance(out.aux, dict) else None,
            y=y,
            cost_bps=cost_bps,
            quantiles=list(DEFAULT_QUANTILES),
            w_nll=1.0,
            w_quantile=0.3,
            w_direction=0.2,
            w_gate=0.05,
        )

        opt.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        for k in logs:
            logs[k].append(float(losses[k].detach().cpu().item()))

    return {k: float(np.mean(v)) if v else 0.0 for k, v in logs.items()}


def _collect_predictions(model: LiquidModel, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    out_mu: List[np.ndarray] = []
    out_sigma: List[np.ndarray] = []
    out_q: List[np.ndarray] = []
    out_dir: List[np.ndarray] = []
    out_gate: List[np.ndarray] = []
    y_raw: List[np.ndarray] = []
    y_net: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            xv = batch["x_values"].to(device)
            xm = batch["x_mask"].to(device)
            y_raw.append(_to_numpy(batch["y_raw"]))
            y_net.append(_to_numpy(batch["y_net"]))

            pred = model(xv, xm)
            out_mu.append(_to_numpy(pred.mu))
            out_sigma.append(_to_numpy(torch.exp(pred.log_sigma)))
            if pred.q is not None:
                out_q.append(_to_numpy(pred.q))
            if pred.direction_logit is not None:
                out_dir.append(_to_numpy(pred.direction_logit))
            gate_t = pred.aux.get("gate") if isinstance(pred.aux, dict) else None
            if isinstance(gate_t, torch.Tensor):
                out_gate.append(_to_numpy(gate_t))

    pack: Dict[str, np.ndarray] = {
        "mu": np.concatenate(out_mu, axis=0) if out_mu else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "sigma": np.concatenate(out_sigma, axis=0) if out_sigma else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "y_raw": np.concatenate(y_raw, axis=0) if y_raw else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "y_net": np.concatenate(y_net, axis=0) if y_net else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "direction_logit": np.concatenate(out_dir, axis=0) if out_dir else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "gate": np.concatenate(out_gate, axis=0) if out_gate else np.zeros((0, len(HORIZONS)), dtype=np.float32),
        "q": np.concatenate(out_q, axis=0) if out_q else np.zeros((0, len(HORIZONS), len(DEFAULT_QUANTILES)), dtype=np.float32),
    }
    return pack


def _run_fold(
    *,
    cfg: TrainConfig,
    samples: Sequence,
    fold: Dict[str, np.ndarray],
    device: torch.device,
) -> Dict[str, object]:
    ds = LiquidSequenceDataset(samples)
    tr = DataLoader(Subset(ds, fold["train"].tolist()), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    va = DataLoader(Subset(ds, fold["val"].tolist()), batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    te = DataLoader(Subset(ds, fold["test"].tolist()), batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_state = None
    best_val = float("inf")
    loss_trace: List[Dict[str, float]] = []
    for _ in range(cfg.epochs):
        train_loss = _train_one_epoch(model, tr, device, opt)
        val_pack = _collect_predictions(model, va, device)
        val_nll = float(np.mean(((val_pack["y_net"] - val_pack["mu"]) / np.clip(val_pack["sigma"], 1e-6, None)) ** 2))
        loss_trace.append({**train_loss, "val_nll_proxy": val_nll})
        if val_nll < best_val:
            best_val = val_nll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("fold_training_failed_no_state")
    model.load_state_dict(best_state)

    test_pack = _collect_predictions(model, te, device)
    metrics = evaluate_liquid_metrics(
        horizons=list(HORIZONS),
        y_raw=test_pack["y_raw"],
        y_net=test_pack["y_net"],
        mu=test_pack["mu"],
        sigma=test_pack["sigma"],
        direction_logit=test_pack["direction_logit"],
        q=test_pack["q"],
        feature_context={},
        cost_profile_name=cfg.cost_profile,
    )
    return {
        "best_state": best_state,
        "best_val": best_val,
        "loss_trace": loss_trace,
        "metrics": metrics,
        "pred": test_pack,
    }


def _hard_gate(agg_metrics: Dict[str, Dict[str, float]], gate_mean: float) -> None:
    for h, m in agg_metrics.items():
        if np.isnan(float(m.get("spearman_ic", 0.0))) or np.isnan(float(m.get("sharpe", 0.0))):
            raise RuntimeError(f"gate_nan_metric:{h}")
        if float(m.get("ece", 0.0)) > 0.60:
            raise RuntimeError(f"gate_ece_exploded:{h}:{m.get('ece')}")
    if gate_mean >= 0.98:
        raise RuntimeError(f"gate_over_open:{gate_mean}")


def _avg_metrics(folds: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for h in HORIZONS:
        rows = [f[h] for f in folds if h in f]
        if not rows:
            out[h] = {}
            continue
        keys = sorted(rows[0].keys())
        out[h] = {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}
    return out


def main() -> int:
    cfg = _build_cfg()
    universe_meta = _resolve_training_universe(cfg)
    symbols_for_training = _normalize_symbols(list(universe_meta.get("symbols") or []))
    if not symbols_for_training:
        raise RuntimeError("training_universe_empty")
    samples = load_training_samples(
        db_url=cfg.db_url,
        symbols=symbols_for_training,
        start_ts=cfg.start,
        end_ts=cfg.end,
        lookback=cfg.lookback,
        max_samples_per_symbol=cfg.max_samples_per_symbol,
        cost_profile_name=cfg.cost_profile,
    )
    if len(samples) < 128:
        raise RuntimeError(f"insufficient_samples:{len(samples)}")

    times = [s.end_ts for s in samples]
    split_cfg = WalkForwardPurgedConfig(
        train_days=cfg.train_days,
        val_days=cfg.val_days,
        test_days=cfg.test_days,
        purge_gap_hours=cfg.purge_gap_hours,
        step_days=cfg.step_days,
    )
    folds = build_walkforward_purged_splits(times, split_cfg)
    if cfg.force_purged and len(folds) == 0:
        raise RuntimeError("walkforward_purged_required_no_fold")
    if len(folds) == 0:
        raise RuntimeError("no_training_folds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_reports: List[Dict[str, object]] = []
    fold_metrics: List[Dict[str, Dict[str, float]]] = []
    oof_mu: List[np.ndarray] = []
    oof_sigma: List[np.ndarray] = []
    oof_y: List[np.ndarray] = []
    oof_dir: List[np.ndarray] = []
    oof_gate: List[np.ndarray] = []

    for i, fold in enumerate(folds):
        fr = _run_fold(cfg=cfg, samples=samples, fold=fold, device=device)
        f_metrics = dict(fr["metrics"])
        fold_metrics.append(f_metrics)
        pred = dict(fr["pred"])
        oof_mu.append(pred["mu"])
        oof_sigma.append(pred["sigma"])
        oof_y.append(pred["y_net"])
        oof_dir.append(pred["direction_logit"])
        oof_gate.append(pred["gate"])

        fold_reports.append(
            {
                "fold": i,
                "time_range": {
                    "train_start": fold["train_start"].isoformat(),
                    "train_end": fold["train_end"].isoformat(),
                    "val_start": fold["val_start"].isoformat(),
                    "val_end": fold["val_end"].isoformat(),
                    "test_start": fold["test_start"].isoformat(),
                    "test_end": fold["test_end"].isoformat(),
                },
                "best_val_nll_proxy": float(fr["best_val"]),
                "loss_trace": fr["loss_trace"],
                "metrics": f_metrics,
            }
        )

    # final fit on all samples with fold-selected config
    full_ds = LiquidSequenceDataset(samples)
    full_loader = DataLoader(full_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    model = _build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    final_loss_trace: List[Dict[str, float]] = []
    for _ in range(cfg.epochs):
        final_loss_trace.append(_train_one_epoch(model, full_loader, device, opt))

    oof_mu_arr = np.concatenate(oof_mu, axis=0)
    oof_sigma_arr = np.concatenate(oof_sigma, axis=0)
    oof_y_arr = np.concatenate(oof_y, axis=0)
    oof_dir_arr = np.concatenate(oof_dir, axis=0) if oof_dir else np.zeros_like(oof_mu_arr)
    oof_gate_arr = np.concatenate(oof_gate, axis=0) if oof_gate else np.zeros_like(oof_mu_arr)

    calibrator = build_calibration_bundle(
        direction_logit=oof_dir_arr,
        y_net=oof_y_arr,
        mu=oof_mu_arr,
        sigma=oof_sigma_arr,
    )

    agg = _avg_metrics(fold_metrics)
    gate_mean = float(np.mean(oof_gate_arr)) if oof_gate_arr.size else 0.0
    _hard_gate(agg, gate_mean)

    training_report = {
        "status": "ok",
        "strict_mode": True,
        "backbone": cfg.backbone,
        "horizons": list(HORIZONS),
        "quantiles": list(DEFAULT_QUANTILES),
        "cost_profile": cfg.cost_profile,
        "walkforward": {
            "train_days": cfg.train_days,
            "val_days": cfg.val_days,
            "test_days": cfg.test_days,
            "purge_gap_hours": cfg.purge_gap_hours,
            "step_days": cfg.step_days,
            "folds": len(folds),
        },
        "folds": fold_reports,
        "aggregate_oos": agg,
        "gate_distribution": {
            "mean": gate_mean,
            "q10": float(np.quantile(oof_gate_arr, 0.1)) if oof_gate_arr.size else 0.0,
            "q50": float(np.quantile(oof_gate_arr, 0.5)) if oof_gate_arr.size else 0.0,
            "q90": float(np.quantile(oof_gate_arr, 0.9)) if oof_gate_arr.size else 0.0,
        },
        "calibration": calibrator,
        "final_fit_loss": final_loss_trace,
    }
    _attach_universe_to_training_report(training_report, universe_meta)

    state = {
        "state_dict": model.state_dict(),
        "schema_hash": str(SCHEMA_HASH),
        "calibration": calibrator,
        "cost_profile": cfg.cost_profile,
        **model.export_meta(),
    }

    manifest = pack_model_artifact(
        model_dir=cfg.out_dir,
        model_id=cfg.model_id,
        state_dict=state,
        schema_path="schema/liquid_feature_schema.yaml",
        lookback=cfg.lookback,
        feature_dim=FEATURE_DIM,
        data_version="main",
        metrics_summary={"oos": agg},
        extra_payload=training_report,
    )

    print(json.dumps({"status": "ok", "manifest": manifest, "schema_hash": SCHEMA_HASH, "training_report": training_report}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
