#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from cost.cost_profile import compute_cost_breakdown_bps, cost_profile_snapshot, load_cost_profile
from mlops_artifacts.validate import validate_manifest_dir
from models.liquid_model import build_liquid_model_from_checkpoint
from training.datasets.liquid_panel_cache_dataset import LiquidPanelCacheDataset


def _parse_ts(raw: str) -> datetime:
    text = str(raw or "").strip()
    if not text:
        return datetime.now(timezone.utc)
    text = text.replace(" ", "T")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _ece(prob: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    if prob.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (prob >= lo) & (prob < hi if i < bins - 1 else prob <= hi)
        if not np.any(m):
            continue
        out += abs(float(np.mean(prob[m])) - float(np.mean(labels[m]))) * (float(np.sum(m)) / float(prob.size))
    return float(out)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    sx = float(np.std(rx))
    sy = float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _collect(model, loader: DataLoader, device: torch.device, horizons: List[str]) -> Dict[str, Any]:
    model.eval()
    y = []
    mu = []
    sigma = []
    q = []
    d = []
    expert = []
    regime = []
    end_ts = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["x_values"].to(device),
                batch["x_mask"].to(device),
                symbol_id=batch["symbol_id"].to(device),
                regime_features=batch["regime_features"].to(device),
                regime_mask=batch["regime_mask"].to(device),
            )
            y.append(batch["y_net"].cpu().numpy())
            mu.append(out.mu.cpu().numpy())
            sigma.append(torch.exp(out.log_sigma).cpu().numpy())
            q.append(out.q.cpu().numpy() if out.q is not None else np.zeros((out.mu.shape[0], len(horizons), 3), dtype=np.float32))
            d.append(torch.sigmoid(out.direction_logit).cpu().numpy() if out.direction_logit is not None else np.zeros_like(out.mu.cpu().numpy()))
            expert.append(out.expert_weights.cpu().numpy() if out.expert_weights is not None else np.zeros((out.mu.shape[0], 4), dtype=np.float32))
            regime.append(batch["regime_features"].cpu().numpy())
            end_ts.extend([int(x) for x in batch["end_ts"]])
    return {
        "y": np.concatenate(y, axis=0),
        "mu": np.concatenate(mu, axis=0),
        "sigma": np.concatenate(sigma, axis=0),
        "q": np.concatenate(q, axis=0),
        "direction_prob": np.concatenate(d, axis=0),
        "expert_weights": np.concatenate(expert, axis=0),
        "regime_features": np.concatenate(regime, axis=0),
        "end_ts": np.asarray(end_ts, dtype=np.int64),
    }


def _metrics(pack: Dict[str, Any], horizons: List[str]) -> Dict[str, Any]:
    y = pack["y"]
    mu = pack["mu"]
    sigma = np.clip(pack["sigma"], 1e-6, None)
    q = pack["q"]
    dprob = pack["direction_prob"]
    out: Dict[str, Any] = {"global": {}, "per_horizon": {}}
    sharpe_rows = []
    for i, h in enumerate(horizons):
        yh = y[:, i]
        mh = mu[:, i]
        sh = sigma[:, i]
        prob = dprob[:, i]
        hit = float(np.mean(np.sign(yh) == np.sign(mh)))
        ic = _spearman(mh, yh)
        ece = _ece(prob, (yh >= 0).astype(np.float64))
        cover = float(np.mean((yh >= q[:, i, 0]) & (yh <= q[:, i, -1])))
        turn = float(np.mean(np.abs(np.diff(np.clip(mh / sh, -1.0, 1.0), prepend=0.0))))
        pnl = np.clip(mh / sh, -1.0, 1.0) * yh
        sharpe = float(np.mean(pnl) / max(1e-9, np.std(pnl)))
        mdd = float(abs(np.min(np.cumprod(1.0 + pnl) / np.maximum.accumulate(np.cumprod(1.0 + pnl)) - 1.0)))
        sharpe_rows.append(sharpe)
        out["per_horizon"][h] = {
            "hit_rate": hit,
            "ic": ic,
            "ece": ece,
            "coverage": cover,
            "turnover": turn,
            "sharpe": sharpe,
            "mdd": mdd,
        }
    out["global"] = {
        "sharpe": float(np.mean(sharpe_rows)),
        "turnover": float(np.mean([out["per_horizon"][h]["turnover"] for h in horizons])),
        "hit_rate": float(np.mean([out["per_horizon"][h]["hit_rate"] for h in horizons])),
        "ic": float(np.mean([out["per_horizon"][h]["ic"] for h in horizons])),
    }
    return out


def _bucket_perf(pack: Dict[str, Any], mask: np.ndarray, h_idx: int = 0) -> Dict[str, float]:
    if not np.any(mask):
        return {"count": 0, "hit_rate": 0.0, "ic": 0.0, "pnl": 0.0}
    y = pack["y"][mask, h_idx]
    mu = pack["mu"][mask, h_idx]
    sigma = np.clip(pack["sigma"][mask, h_idx], 1e-6, None)
    sig = np.clip(mu / sigma, -1.0, 1.0)
    pnl = float(np.mean(sig * y))
    return {
        "count": int(np.sum(mask)),
        "hit_rate": float(np.mean(np.sign(y) == np.sign(mu))),
        "ic": _spearman(mu, y),
        "pnl": pnl,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate top50 panel artifact with stratified gating outputs")
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--universe-snapshot", required=True)
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--strategy-config", default="")
    ap.add_argument("--cost-profile", default="standard")
    ap.add_argument("--out-dir", default="artifacts/eval/top50_latest")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = validate_manifest_dir(Path(str(args.artifact_dir)))
    ckpt = torch.load(Path(str(args.artifact_dir)) / str(manifest["files"]["weights"]), map_location="cpu")
    model = build_liquid_model_from_checkpoint(ckpt)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ds = LiquidPanelCacheDataset(cache_dir=args.cache_dir, lookback=int(ckpt.get("lookback") or 96), horizons=("1h", "4h", "1d", "7d"))
    audit_path = Path(args.cache_dir) / "data_audit.json"
    if not audit_path.exists():
        raise RuntimeError(f"cache_audit_missing:{audit_path}")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if not bool((audit_payload.get("asof_leakage_check") or {}).get("passed", False)):
        raise RuntimeError("gate_leakage_check_failed")

    start_ts = _parse_ts(args.start).timestamp() if str(args.start).strip() else None
    end_ts = _parse_ts(args.end).timestamp() if str(args.end).strip() else None
    idx = []
    for i, ref in enumerate(ds._refs):  # pylint: disable=protected-access
        ts = int(ref.end_ts)
        if start_ts is not None and ts < int(start_ts):
            continue
        if end_ts is not None and ts > int(end_ts):
            continue
        idx.append(i)
    if not idx:
        raise RuntimeError("eval_window_empty")
    loader = DataLoader(Subset(ds, idx), batch_size=128, shuffle=False, num_workers=0)

    horizons = [str(h) for h in (ckpt.get("horizons") or ["1h", "4h", "1d", "7d"])]
    pack = _collect(model, loader, device, horizons)
    metrics = _metrics(pack, horizons)

    expert = pack["expert_weights"]
    expert_usage = np.mean(expert, axis=0)
    expert_entropy = float(np.mean(-np.sum(np.clip(expert, 1e-8, 1.0) * np.log(np.clip(expert, 1e-8, 1.0)), axis=1)))
    collapse = float(np.max(expert_usage))
    if collapse >= 0.95:
        raise RuntimeError(f"gate_router_collapse:{collapse:.6f}")
    router_report = {
        "expert_usage": expert_usage.tolist(),
        "entropy": expert_entropy,
        "collapse_max": collapse,
    }

    rf = pack["regime_features"]
    stratified = {
        "funding_extreme": _bucket_perf(pack, np.abs(rf[:, 7]) > 2.0),
        "high_vol": _bucket_perf(pack, rf[:, 0] > np.median(rf[:, 0])),
        "low_depth": _bucket_perf(pack, rf[:, 4] < np.median(rf[:, 4])),
        "wide_spread": _bucket_perf(pack, rf[:, 3] > np.median(rf[:, 3])),
        "oi_drop": _bucket_perf(pack, rf[:, 10] < 0.0),
    }

    cal = {
        "ece": {h: float(metrics["per_horizon"][h]["ece"]) for h in horizons},
        "coverage": {h: float(metrics["per_horizon"][h]["coverage"]) for h in horizons},
    }
    if any(abs(float(cal["coverage"][h]) - 0.8) > 0.25 for h in horizons):
        raise RuntimeError("gate_calibration_coverage_deviation")

    cost_snap = cost_profile_snapshot(str(args.cost_profile))
    if str(ckpt.get("cost_profile") or str(args.cost_profile)) != str(args.cost_profile):
        raise RuntimeError("gate_cost_profile_name_mismatch")

    pnl_csv = out_dir / "pnl_timeseries.csv"
    profile = load_cost_profile(str(args.cost_profile))
    with pnl_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["end_ts", "horizon", "pnl_raw", "fee", "slippage", "impact", "funding", "infra", "total_cost"])
        for i in range(pack["y"].shape[0]):
            for h_idx, h in enumerate(horizons):
                bd = compute_cost_breakdown_bps(
                    horizon=h,
                    profile=profile,
                    market_state={"realized_vol": float(pack["regime_features"][i, 0]), "funding_rate": float(pack["regime_features"][i, 6])},
                    liquidity_features={"orderbook_depth_total": float(np.expm1(pack["regime_features"][i, 4])), "spread_bps": float(pack["regime_features"][i, 3])},
                )
                pnl_raw = float(np.clip(pack["mu"][i, h_idx] / max(1e-8, pack["sigma"][i, h_idx]), -1.0, 1.0) * pack["y"][i, h_idx])
                w.writerow(
                    [
                        int(pack["end_ts"][i]),
                        h,
                        pnl_raw,
                        float(bd["fee_bps"]) / 10000.0,
                        float(bd["slippage_bps"]) / 10000.0,
                        float(bd["impact_bps"]) / 10000.0,
                        float(bd["funding_bps"]) / 10000.0,
                        float(bd["infra_bps"]) / 10000.0,
                        float(bd["total_bps"]) / 10000.0,
                    ]
                )

    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "stratified_metrics.json").write_text(json.dumps(stratified, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "calibration.json").write_text(json.dumps(cal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "router_report.json").write_text(json.dumps(router_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "cost_profile_snapshot.json").write_text(json.dumps(cost_snap, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(
        "\n".join(
            [
                "# Top50 Evaluation Report",
                "",
                f"- samples: {pack['y'].shape[0]}",
                f"- horizons: {', '.join(horizons)}",
                f"- sharpe: {metrics['global']['sharpe']:.6f}",
                f"- router collapse max: {collapse:.6f}",
                f"- cache leakage check: {bool((audit_payload.get('asof_leakage_check') or {}).get('passed', False))}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "out_dir": str(out_dir),
                "metrics": str(out_dir / "metrics.json"),
                "stratified_metrics": str(out_dir / "stratified_metrics.json"),
                "calibration": str(out_dir / "calibration.json"),
                "pnl_timeseries": str(pnl_csv),
                "router_report": str(out_dir / "router_report.json"),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
