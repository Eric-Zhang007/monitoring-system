from __future__ import annotations

import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException

from schemas_v2 import (
    AutoGateEvaluateRequest,
    AutoGateEvaluateResponse,
    BacktestRunRequest,
    BacktestRunResponse,
    DataQualityAuditUpdate,
    DataQualitySampleRequest,
    DataQualityStatsResponse,
    DriftEvaluateRequest,
    DriftEvaluateResponse,
    ExecuteOrdersRequest,
    ExecuteOrdersResponse,
    ExecutionOrderStatusResponse,
    ModelGateRequest,
    ModelGateResponse,
    IngestEventsRequest,
    IngestEventsResponse,
    LiquidPredictRequest,
    PnLAttributionResponse,
    PortfolioRebalanceRequest,
    PortfolioRebalanceResponse,
    PortfolioScoreRequest,
    PortfolioScoreResponse,
    RebalancePosition,
    RiskCheckRequest,
    RiskCheckResponse,
    RiskLimitsResponse,
    RollbackCheckRequest,
    RollbackCheckResponse,
    SignalGenerateRequest,
    SignalGenerateResponse,
    SubmitExecutionOrdersRequest,
    SubmitExecutionOrdersResponse,
    VCPredictRequest,
)
from execution_engine import ExecutionEngine
from v2_repository import V2Repository

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://monitor:change_me_please@localhost:5432/monitor")

router = APIRouter(prefix="/api/v2", tags=["v2"])
repo = V2Repository(DATABASE_URL)
exec_engine = ExecutionEngine()


def _sigmoid(x: float) -> float:
    if x < -30:
        return 0.0
    if x > 30:
        return 1.0
    return 1.0 / (1.0 + (2.718281828 ** (-x)))


def _risk_limits() -> Dict[str, float]:
    return {
        "max_single_weight": float(os.getenv("RISK_MAX_SINGLE_WEIGHT", "0.2")),
        "max_gross_exposure": float(os.getenv("RISK_MAX_GROSS_EXPOSURE", "1.0")),
        "max_turnover_per_rebalance": float(os.getenv("RISK_MAX_TURNOVER", "0.35")),
        "max_realized_drawdown": float(os.getenv("RISK_MAX_DRAWDOWN", "0.2")),
        "max_sector_exposure": float(os.getenv("RISK_MAX_SECTOR_EXPOSURE", "0.45")),
        "max_style_exposure": float(os.getenv("RISK_MAX_STYLE_EXPOSURE", "0.55")),
    }


def _bucket_limits() -> Dict[str, float]:
    return {
        "trend": float(os.getenv("RISK_BUCKET_TREND", "0.5")),
        "event": float(os.getenv("RISK_BUCKET_EVENT", "0.4")),
        "mean_reversion": float(os.getenv("RISK_BUCKET_MEAN_REVERSION", "0.35")),
    }


def _position_map(positions: List[RebalancePosition]) -> Dict[Tuple[str, str], float]:
    return {(p.target.upper(), p.track): float(p.weight) for p in positions}


def _fallback_sector(target: str, track: str) -> str:
    t = target.upper()
    if track == "vc":
        return "private"
    if t in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        return "crypto"
    return "equity"


def _fallback_style(target: str, track: str) -> str:
    t = target.upper()
    if track == "vc":
        return "venture"
    if t in {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE"}:
        return "crypto_beta"
    return "core"


def _resolve_position_profiles(positions: List[RebalancePosition]) -> List[RebalancePosition]:
    if not positions:
        return positions
    missing_targets = [p.target.upper() for p in positions if not p.sector or not p.style_bucket]
    try:
        target_profiles = repo.get_target_profiles(missing_targets) if missing_targets else {}
    except Exception:
        target_profiles = {}
    resolved: List[RebalancePosition] = []
    for p in positions:
        key = p.target.upper()
        profile = target_profiles.get(key, {})
        sector = p.sector or profile.get("sector") or _fallback_sector(key, p.track)
        style = p.style_bucket or profile.get("style_bucket") or _fallback_style(key, p.track)
        resolved.append(RebalancePosition(target=key, track=p.track, weight=float(p.weight), sector=sector, style_bucket=style))
    return resolved


def _evaluate_risk(
    proposed: List[RebalancePosition],
    current: List[RebalancePosition],
    realized_drawdown: float,
    max_sector_exposure_override: Optional[float] = None,
    max_style_exposure_override: Optional[float] = None,
) -> Tuple[List[RebalancePosition], List[str], float, float]:
    limits = _risk_limits()
    if max_sector_exposure_override is not None:
        limits["max_sector_exposure"] = max_sector_exposure_override
    if max_style_exposure_override is not None:
        limits["max_style_exposure"] = max_style_exposure_override
    violations: List[str] = []
    proposed = _resolve_position_profiles(proposed)
    current = _resolve_position_profiles(current)

    capped: List[RebalancePosition] = []
    for p in proposed:
        w = float(p.weight)
        max_single = limits["max_single_weight"]
        if abs(w) > max_single:
            violations.append(f"single_weight_exceeded:{p.target}")
            w = max_single if w > 0 else -max_single
        capped.append(
            RebalancePosition(
                target=p.target,
                track=p.track,
                weight=w,
                sector=p.sector,
                style_bucket=p.style_bucket,
            )
        )

    gross = sum(abs(p.weight) for p in capped)
    if gross > limits["max_gross_exposure"] and gross > 0:
        violations.append("gross_exposure_exceeded")
        scale = limits["max_gross_exposure"] / gross
        capped = [
            RebalancePosition(
                target=p.target,
                track=p.track,
                weight=p.weight * scale,
                sector=p.sector,
                style_bucket=p.style_bucket,
            )
            for p in capped
        ]
        gross = limits["max_gross_exposure"]

    sector_exposure: Dict[str, float] = {}
    for p in capped:
        k = str(p.sector or _fallback_sector(p.target, p.track)).lower()
        sector_exposure[k] = sector_exposure.get(k, 0.0) + abs(float(p.weight))
    for sector, exp in sector_exposure.items():
        if exp > limits["max_sector_exposure"] and exp > 0:
            violations.append(f"sector_exposure_exceeded:{sector}")
            scale = limits["max_sector_exposure"] / exp
            capped = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=p.weight * scale if str(p.sector or _fallback_sector(p.target, p.track)).lower() == sector else p.weight,
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in capped
            ]

    style_exposure: Dict[str, float] = {}
    for p in capped:
        k = str(p.style_bucket or _fallback_style(p.target, p.track)).lower()
        style_exposure[k] = style_exposure.get(k, 0.0) + abs(float(p.weight))
    for style, exp in style_exposure.items():
        if exp > limits["max_style_exposure"] and exp > 0:
            violations.append(f"style_exposure_exceeded:{style}")
            scale = limits["max_style_exposure"] / exp
            capped = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=p.weight * scale if str(p.style_bucket or _fallback_style(p.target, p.track)).lower() == style else p.weight,
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in capped
            ]

    gross = sum(abs(p.weight) for p in capped)

    current_map = _position_map(current)
    next_map = _position_map(capped)
    universe = set(current_map.keys()) | set(next_map.keys())
    turnover = sum(abs(next_map.get(k, 0.0) - current_map.get(k, 0.0)) for k in universe)
    if turnover > limits["max_turnover_per_rebalance"]:
        violations.append("turnover_exceeded")

    if realized_drawdown > limits["max_realized_drawdown"]:
        violations.append("drawdown_exceeded")

    return capped, violations, gross, turnover


def _build_vc_prediction(company_name: str, horizon_months: int) -> Dict[str, object]:
    context = repo.recent_event_context(company_name, limit=8)

    funding_hits = sum(1 for e in context if e["event_type"] == "funding")
    product_hits = sum(1 for e in context if e["event_type"] == "product")
    regulatory_hits = sum(1 for e in context if e["event_type"] == "regulatory")
    score_raw = funding_hits * 0.9 + product_hits * 0.4 - regulatory_hits * 0.6

    horizon_adj = {6: 0.1, 12: 0.0, 24: -0.05}[horizon_months]
    p_next_round = max(0.01, min(0.99, _sigmoid(score_raw * 0.7 + horizon_adj)))
    p_exit = max(0.01, min(0.95, _sigmoid(score_raw * 0.3 - 0.4)))
    expected_moic = {
        "p10": round(0.6 + p_next_round * 0.8, 2),
        "p50": round(1.0 + p_next_round * 1.6, 2),
        "p90": round(1.2 + p_next_round * 3.5 + p_exit * 2.0, 2),
    }

    explanation = {
        "top_event_contributors": [
            {
                "event_id": e["id"],
                "event_type": e["event_type"],
                "title": e["title"],
                "weight": round(0.2 + e.get("confidence_score", 0.5) * 0.6, 3),
            }
            for e in context[:5]
        ],
        "top_feature_contributors": [
            {"feature": "recent_funding_event_count", "value": funding_hits, "contribution": round(funding_hits * 0.9, 3)},
            {"feature": "recent_product_event_count", "value": product_hits, "contribution": round(product_hits * 0.4, 3)},
            {"feature": "recent_regulatory_event_count", "value": regulatory_hits, "contribution": round(-regulatory_hits * 0.6, 3)},
        ],
        "evidence_links": [e["source_url"] for e in context if e.get("source_url")][:5],
        "model_version": "vc-survival-baseline-v2",
        "feature_version": "feature-store-v2.0",
    }

    outputs = {
        "p_next_round": {
            "6m": round(max(0.01, min(0.99, p_next_round + 0.05)), 4),
            "12m": round(p_next_round, 4),
            "24m": round(max(0.01, min(0.99, p_next_round - 0.08)), 4),
        },
        "p_exit_24m": round(p_exit, 4),
        "expected_moic_distribution": expected_moic,
        "as_of": datetime.utcnow().isoformat(),
    }

    return {
        "target": company_name,
        "score": float(round(p_next_round, 4)),
        "confidence": float(round(min(0.95, 0.55 + len(context) * 0.04), 4)),
        "outputs": outputs,
        "explanation": explanation,
    }


def _build_liquid_prediction(symbol: str, horizon: str) -> Dict[str, object]:
    price_row = repo.latest_price_snapshot(symbol)
    if not price_row:
        raise HTTPException(status_code=404, detail=f"no price snapshot for {symbol}")

    context = repo.recent_event_context(symbol, limit=8)
    market_events = sum(1 for e in context if e["event_type"] == "market")
    regulatory_events = sum(1 for e in context if e["event_type"] == "regulatory")

    horizon_scale = {"1h": 0.3, "1d": 1.0, "7d": 2.1}[horizon]
    base_return = (market_events * 0.002 - regulatory_events * 0.003) * horizon_scale
    vol_forecast = max(0.01, 0.02 + (market_events + regulatory_events) * 0.0025)
    confidence = max(0.3, min(0.95, 0.55 + len(context) * 0.03))

    outputs = {
        "expected_return": round(base_return, 6),
        "vol_forecast": round(vol_forecast, 6),
        "signal_confidence": round(confidence, 4),
        "current_price": float(price_row["price"]),
        "horizon": horizon,
        "as_of": datetime.utcnow().isoformat(),
    }

    explanation = {
        "top_event_contributors": [
            {
                "event_id": e["id"],
                "event_type": e["event_type"],
                "title": e["title"],
                "weight": round(0.1 + e.get("confidence_score", 0.5) * 0.7, 3),
            }
            for e in context[:5]
        ],
        "top_feature_contributors": [
            {"feature": "recent_market_event_count", "value": market_events, "contribution": round(market_events * 0.002, 6)},
            {"feature": "recent_regulatory_event_count", "value": regulatory_events, "contribution": round(-regulatory_events * 0.003, 6)},
            {"feature": "price_level", "value": float(price_row["price"]), "contribution": 0.0},
        ],
        "evidence_links": [e["source_url"] for e in context if e.get("source_url")][:5],
        "model_version": "liquid-tsfm-adapter-baseline-v2",
        "feature_version": "feature-store-v2.0",
    }

    return {
        "target": symbol.upper(),
        "score": float(round(base_return, 6)),
        "confidence": float(round(confidence, 4)),
        "outputs": outputs,
        "explanation": explanation,
    }


def _walk_forward_metrics(price_rows: List[Dict], train_days: int, test_days: int, fee_bps: float, slippage_bps: float) -> Dict[str, float]:
    filtered_rows = []
    for r in price_rows:
        ts = r.get("timestamp")
        if hasattr(ts, "weekday") and ts.weekday() >= 5:
            continue
        filtered_rows.append(r)

    if len(filtered_rows) < (train_days + test_days + 2):
        return {
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }

    prices = np.array([float(r["price"]) for r in filtered_rows], dtype=np.float64)
    volumes = np.array([float(r.get("volume") or 0.0) for r in filtered_rows], dtype=np.float64)
    rets = np.diff(prices) / np.clip(prices[:-1], 1e-12, None)

    scores: List[float] = []
    future_rets: List[float] = []
    equity_curve: List[float] = [1.0]
    wins = 0
    trades = 0
    turnover = 0.0
    prev_signal = 0.0

    start = train_days
    while start < len(rets) - 1:
        end = min(start + test_days, len(rets) - 1)
        train_slice = rets[start - train_days : start]
        mu = float(np.mean(train_slice)) if train_slice.size else 0.0
        sigma = float(np.std(train_slice)) if train_slice.size else 0.0
        threshold = max(1e-6, sigma * 0.2)

        for i in range(start, end):
            mom = float(rets[i - 1] - mu)
            signal = 1.0 if mom > threshold else -1.0 if mom < -threshold else 0.0
            # capacity/impact proxy: cap exposure by recent realized volatility
            cap = float(max(0.2, min(1.0, 0.8 - sigma * 20.0)))
            signal = signal * cap
            nxt = float(rets[i])
            vol = float(max(1.0, volumes[i]))
            impact_bps = min(25.0, 1200.0 / np.sqrt(vol))
            cost = ((fee_bps + slippage_bps + impact_bps) / 10000.0) * abs(signal - prev_signal)
            pnl = signal * nxt - cost

            eq = equity_curve[-1] * (1.0 + pnl)
            equity_curve.append(eq)

            scores.append(signal)
            future_rets.append(nxt)
            turnover += abs(signal - prev_signal)
            trades += 1
            if signal * nxt > 0:
                wins += 1
            prev_signal = signal

        start += test_days

    if not scores:
        return {
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
        }

    s = np.array(scores, dtype=np.float64)
    y = np.array(future_rets, dtype=np.float64)
    ic = float(np.corrcoef(s, y)[0, 1]) if np.std(s) > 0 and np.std(y) > 0 else 0.0
    hit_rate = float(wins / max(1, trades))
    pnl_after_cost = float(equity_curve[-1] - 1.0)

    peak = equity_curve[0]
    max_dd = 0.0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (peak - v) / max(peak, 1e-12)
        max_dd = max(max_dd, dd)

    return {
        "samples": int(len(scores)),
        "ic": round(ic, 6),
        "hit_rate": round(hit_rate, 6),
        "turnover": round(float(turnover), 6),
        "pnl_after_cost": round(pnl_after_cost, 6),
        "max_drawdown": round(float(max_dd), 6),
    }


def _default_model_by_track(track: str) -> Tuple[str, str]:
    if track == "liquid":
        return "liquid_ttm_ensemble", "v2.1"
    return "vc_survival_model", "v2.1"


def _evaluate_gate(
    track: str,
    min_ic: float,
    min_pnl_after_cost: float,
    max_drawdown: float,
    windows: int,
) -> Tuple[bool, str, Dict[str, float], int]:
    runs = repo.list_recent_backtest_runs(track=track, limit=windows)
    usable = [r for r in runs if isinstance(r.get("metrics"), dict) and r["metrics"].get("status") == "completed"]
    if len(usable) < windows:
        return False, "insufficient_windows", {"ic": 0.0, "pnl_after_cost": 0.0, "max_drawdown": 1.0}, len(usable)

    ic_vals = [float(r["metrics"].get("ic", 0.0)) for r in usable[:windows]]
    pnl_vals = [float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable[:windows]]
    dd_vals = [float(r["metrics"].get("max_drawdown", 1.0)) for r in usable[:windows]]

    summary = {
        "ic": round(float(np.mean(ic_vals)), 6),
        "pnl_after_cost": round(float(np.mean(pnl_vals)), 6),
        "max_drawdown": round(float(np.mean(dd_vals)), 6),
    }
    passed = summary["ic"] >= min_ic and summary["pnl_after_cost"] >= min_pnl_after_cost and summary["max_drawdown"] <= max_drawdown
    reason = "passed" if passed else "threshold_failed"
    return passed, reason, summary, windows


def _strategy_bucket(track: str, score: float, confidence: float, target: str) -> str:
    if track == "vc":
        return "event"
    px = repo.latest_price_snapshot(target)
    if not px:
        return "event"
    price = float(px.get("price") or 0.0)
    volume = float(px.get("volume") or 0.0)
    fast_signal = np.tanh(np.log1p(max(volume, 0.0)) * 0.05)
    slow_signal = np.tanh(np.log1p(max(price, 0.0)) * 0.03 + score * 2.0)
    mixed = 0.6 * slow_signal + 0.4 * fast_signal
    if abs(mixed) > 0.35 and confidence > 0.55:
        return "trend"
    if abs(mixed) < 0.1 and confidence > 0.6:
        return "mean_reversion"
    return "event"


def _ks_statistic(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    x = np.sort(np.array(a, dtype=np.float64))
    y = np.sort(np.array(b, dtype=np.float64))
    all_vals = np.sort(np.unique(np.concatenate([x, y])))
    cdf_x = np.searchsorted(x, all_vals, side="right") / max(1, len(x))
    cdf_y = np.searchsorted(y, all_vals, side="right") / max(1, len(y))
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _psi(reference: List[float], current: List[float], bins: int = 10) -> float:
    if not reference or not current:
        return 0.0
    ref = np.array(reference, dtype=np.float64)
    cur = np.array(current, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, None)
    cur_pct = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


@router.post("/ingest/events", response_model=IngestEventsResponse)
async def ingest_events(payload: IngestEventsRequest) -> IngestEventsResponse:
    accepted, inserted, deduplicated, event_ids = repo.ingest_events(payload.events)
    return IngestEventsResponse(
        accepted=accepted,
        inserted=inserted,
        deduplicated=deduplicated,
        event_ids=event_ids,
    )


@router.get("/entities/{entity_id}")
async def get_entity(entity_id: int):
    entity = repo.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="entity not found")
    return entity


@router.post("/predict/vc")
async def predict_vc(payload: VCPredictRequest):
    result = _build_vc_prediction(payload.company_name, payload.horizon_months)
    prediction_id = repo.insert_prediction(
        track="vc",
        target=payload.company_name,
        score=result["score"],
        confidence=result["confidence"],
        outputs=result["outputs"],
        explanation=result["explanation"],
        horizon=f"{payload.horizon_months}m",
        feature_set_id="feature-store-v2.0",
    )

    return {
        "prediction_id": prediction_id,
        "track": "vc",
        "target": payload.company_name,
        "outputs": result["outputs"],
        "explanation": result["explanation"],
    }


@router.post("/predict/liquid")
async def predict_liquid(payload: LiquidPredictRequest):
    result = _build_liquid_prediction(payload.symbol, payload.horizon)
    prediction_id = repo.insert_prediction(
        track="liquid",
        target=payload.symbol.upper(),
        score=result["score"],
        confidence=result["confidence"],
        outputs=result["outputs"],
        explanation=result["explanation"],
        horizon=payload.horizon,
        feature_set_id="feature-store-v2.0",
    )

    return {
        "prediction_id": prediction_id,
        "track": "liquid",
        "target": payload.symbol.upper(),
        "outputs": result["outputs"],
        "explanation": result["explanation"],
    }


@router.post("/portfolio/score", response_model=PortfolioScoreResponse)
async def portfolio_score(payload: PortfolioScoreRequest) -> PortfolioScoreResponse:
    if not payload.positions:
        raise HTTPException(status_code=400, detail="positions cannot be empty")

    weighted_return = sum(p.score * (1.0 / max(p.risk, 0.01)) for p in payload.positions)
    risk_proxy = sum(p.risk for p in payload.positions) / len(payload.positions)

    alpha_score = weighted_return / len(payload.positions)
    expected_vol = min(2.5, risk_proxy / payload.risk_budget)

    recommendations: List[Dict] = []
    for p in payload.positions:
        action = "hold"
        if p.score > 0 and p.risk < payload.risk_budget:
            action = "increase"
        elif p.score < 0:
            action = "reduce"
        recommendations.append({"target": p.target, "action": action, "track": p.track})

    return PortfolioScoreResponse(
        alpha_score=round(alpha_score, 6),
        expected_return=round(alpha_score * 0.8, 6),
        expected_volatility=round(expected_vol, 6),
        recommendations=recommendations,
    )


@router.get("/predictions/{prediction_id}/explanation")
async def get_prediction_explanation(prediction_id: int):
    row = repo.get_prediction_explanation(prediction_id)
    if not row:
        raise HTTPException(status_code=404, detail="prediction not found")
    return row


@router.post("/backtest/run", response_model=BacktestRunResponse)
async def run_backtest(payload: BacktestRunRequest) -> BacktestRunResponse:
    targets = payload.targets
    if not targets:
        if payload.track == "liquid":
            targets = [s.strip().upper() for s in os.getenv("LIQUID_SYMBOLS", "BTC,ETH,AAPL,TSLA,NVDA,SPY").split(",") if s.strip()]
        else:
            targets = ["OpenAI", "Anthropic", "Scale AI"]

    run_name = f"{payload.track}-wf-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    config = payload.model_dump()
    config["targets"] = targets
    run_id = repo.create_backtest_run(run_name=run_name, track=payload.track, config=config)

    all_metrics: List[Dict[str, float]] = []
    for target in targets:
        rows = repo.load_price_history(target, lookback_days=payload.lookback_days)
        m = _walk_forward_metrics(
            rows,
            train_days=payload.train_days,
            test_days=payload.test_days,
            fee_bps=payload.fee_bps,
            slippage_bps=payload.slippage_bps,
        )
        all_metrics.append(m)

    samples = sum(m["samples"] for m in all_metrics)
    if samples == 0:
        agg = {
            "status": "failed",
            "reason": "no_price_data",
            "targets": targets,
            "samples": 0,
            "ic": 0.0,
            "hit_rate": 0.0,
            "turnover": 0.0,
            "pnl_after_cost": 0.0,
            "max_drawdown": 0.0,
            "gate_passed": False,
        }
    else:
        agg = {
            "status": "completed",
            "targets": targets,
            "samples": int(samples),
            "ic": round(float(np.mean([m["ic"] for m in all_metrics])), 6),
            "hit_rate": round(float(np.mean([m["hit_rate"] for m in all_metrics])), 6),
            "turnover": round(float(np.mean([m["turnover"] for m in all_metrics])), 6),
            "pnl_after_cost": round(float(np.mean([m["pnl_after_cost"] for m in all_metrics])), 6),
            "max_drawdown": round(float(np.mean([m["max_drawdown"] for m in all_metrics])), 6),
        }
        agg["gate_passed"] = bool(agg["ic"] > 0 and agg["pnl_after_cost"] > 0 and agg["max_drawdown"] < 0.2)

    repo.finish_backtest_run(run_id, agg)

    model_name, model_version = _default_model_by_track(payload.track)
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=0.0,
        min_pnl_after_cost=0.0,
        max_drawdown=0.2,
        windows=3,
    )
    repo.promote_model(
        track=payload.track,
        model_name=model_name,
        model_version=model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked},
        gate_reason=reason,
    )
    current = repo.get_active_model_state(payload.track)
    prev_name = current["active_model_name"] if current else None
    prev_ver = current["active_model_version"] if current else None
    if passed:
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=model_name,
            active_model_version=model_version,
            previous_model_name=prev_name,
            previous_model_version=prev_ver,
            status="active",
            metadata={"source": "auto_gate", "summary": summary},
        )

    return BacktestRunResponse(
        run_id=run_id,
        run_name=run_name,
        track=payload.track,
        status=agg["status"],
        metrics=agg,
        config=config,
    )


@router.get("/backtest/{run_id}")
async def get_backtest(run_id: int):
    row = repo.get_backtest_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail="backtest run not found")
    return row


@router.post("/signals/generate", response_model=SignalGenerateResponse)
async def generate_signal(payload: SignalGenerateRequest) -> SignalGenerateResponse:
    decision_id = uuid.uuid4().hex

    if payload.track == "liquid":
        pred = _build_liquid_prediction(payload.target, payload.horizon)
        model_version = "liquid-tsfm-adapter-baseline-v2"
    else:
        pred = _build_vc_prediction(payload.target, 12)
        model_version = "vc-survival-baseline-v2"

    action = "hold"
    score = float(pred["score"])
    confidence = float(pred["confidence"])
    bucket = _strategy_bucket(payload.track, score, confidence, payload.target)

    if payload.policy == "ensemble-v1":
        # slow model: historical/event score, fast model: recent price-volume pulse
        px = repo.latest_price_snapshot(payload.target)
        fast = 0.0
        if px:
            p = float(px.get("price") or 0.0)
            v = float(px.get("volume") or 0.0)
            fast = float(np.tanh(np.log1p(max(v, 0.0)) * 0.03 + np.log1p(max(p, 0.0)) * 0.01))
        score = round(0.7 * score + 0.3 * fast, 6)
        confidence = round(min(0.95, confidence + 0.03), 4)

    if confidence >= payload.min_confidence:
        if score > 0.01:
            action = "buy"
        elif score < -0.01:
            action = "sell"

    reason = f"policy={payload.policy};model={model_version};score={round(score, 6)}"
    signal_id = repo.insert_signal_candidate(
        track=payload.track,
        target=payload.target.upper(),
        horizon=payload.horizon,
        score=score,
        confidence=confidence,
        action=action,
        policy=payload.policy,
        decision_id=decision_id,
        metadata={
            "reason": reason,
            "min_confidence": payload.min_confidence,
            "strategy_id": payload.strategy_id,
            "cost_profile": payload.cost_profile,
            "risk_profile": payload.risk_profile,
        },
    )

    return SignalGenerateResponse(
        signal_id=signal_id,
        track=payload.track,
        target=payload.target.upper(),
        horizon=payload.horizon,
        action=action,
        score=score,
        confidence=confidence,
        reason=reason,
        policy=payload.policy,
        strategy_bucket=bucket,
        created_at=datetime.utcnow(),
    )


@router.post("/models/gate/evaluate", response_model=ModelGateResponse)
async def evaluate_model_gate(payload: ModelGateRequest) -> ModelGateResponse:
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=payload.min_ic,
        min_pnl_after_cost=payload.min_pnl_after_cost,
        max_drawdown=payload.max_drawdown,
        windows=payload.windows,
    )
    repo.promote_model(
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked},
        gate_reason=reason,
    )
    if passed:
        current = repo.get_active_model_state(payload.track)
        prev_name = current["active_model_name"] if current else None
        prev_ver = current["active_model_version"] if current else None
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=payload.model_name,
            active_model_version=payload.model_version,
            previous_model_name=prev_name,
            previous_model_version=prev_ver,
            status="active",
            metadata={"source": "manual_gate", "summary": summary},
        )

    return ModelGateResponse(
        passed=passed,
        track=payload.track,
        model_name=payload.model_name,
        model_version=payload.model_version,
        reason=reason,
        windows_checked=checked,
        metrics_summary=summary,
    )


@router.post("/models/rollback/check", response_model=RollbackCheckResponse)
async def check_model_rollback(payload: RollbackCheckRequest) -> RollbackCheckResponse:
    runs = repo.list_recent_backtest_runs(track=payload.track, limit=payload.max_recent_losses)
    usable = [r for r in runs if isinstance(r.get("metrics"), dict) and r["metrics"].get("status") == "completed"]
    if not usable:
        return RollbackCheckResponse(
            rollback_triggered=False,
            reason="no_recent_metrics",
            from_model=f"{payload.model_name}:{payload.model_version}",
            to_model=f"{payload.model_name}:{payload.model_version}",
            metrics={},
        )

    hit_rate = float(np.mean([float(r["metrics"].get("hit_rate", 0.0)) for r in usable]))
    dd = float(np.mean([float(r["metrics"].get("max_drawdown", 1.0)) for r in usable]))
    pnl = float(np.mean([float(r["metrics"].get("pnl_after_cost", 0.0)) for r in usable]))

    trigger = hit_rate < payload.min_recent_hit_rate or dd > payload.max_recent_drawdown or pnl < 0
    metrics = {"hit_rate": round(hit_rate, 6), "max_drawdown": round(dd, 6), "pnl_after_cost": round(pnl, 6)}
    from_model = f"{payload.model_name}:{payload.model_version}"
    to_model = from_model
    reason = "stable"

    current = repo.get_active_model_state(payload.track)
    if trigger and current:
        prev_name = current.get("previous_model_name")
        prev_ver = current.get("previous_model_version")
        if prev_name and prev_ver:
            repo.upsert_active_model_state(
                track=payload.track,
                active_model_name=prev_name,
                active_model_version=prev_ver,
                previous_model_name=current.get("active_model_name"),
                previous_model_version=current.get("active_model_version"),
                status="rolled_back",
                metadata={"source": "rollback_check", "metrics": metrics},
            )
            to_model = f"{prev_name}:{prev_ver}"
            reason = "rollback_to_previous_model"
        else:
            reason = "rollback_triggered_but_no_previous_model"

        repo.save_risk_event(
            decision_id=f"rollback-{uuid.uuid4().hex[:12]}",
            severity="critical",
            code="model_kpi_degradation",
            message=reason,
            payload=metrics,
        )

    return RollbackCheckResponse(
        rollback_triggered=trigger,
        reason=reason,
        from_model=from_model,
        to_model=to_model,
        metrics=metrics,
    )


@router.get("/risk/limits", response_model=RiskLimitsResponse)
async def get_risk_limits() -> RiskLimitsResponse:
    limits = _risk_limits()
    return RiskLimitsResponse(updated_at=datetime.utcnow(), **limits)


@router.post("/risk/check", response_model=RiskCheckResponse)
async def risk_check(payload: RiskCheckRequest) -> RiskCheckResponse:
    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed=payload.proposed_positions,
        current=payload.current_positions,
        realized_drawdown=payload.realized_drawdown,
        max_sector_exposure_override=payload.max_sector_exposure_override,
        max_style_exposure_override=payload.max_style_exposure_override,
    )
    approved = len(violations) == 0
    hard_block = "drawdown_exceeded" in violations
    kill_switch_state = "triggered" if hard_block else "armed"
    risk_budget_used = round(float(gross), 6)

    if violations:
        decision_id = f"risk-check-{uuid.uuid4().hex[:12]}"
        for code in violations:
            repo.save_risk_event(
                decision_id=decision_id,
                severity="warning",
                code=code,
                message="risk check violation",
                payload={"gross_exposure": gross, "turnover": turnover},
            )

    return RiskCheckResponse(
        approved=approved,
        violations=violations,
        adjusted_positions=adjusted,
        gross_exposure=round(gross, 6),
        expected_turnover=round(turnover, 6),
        hard_block=hard_block,
        kill_switch_state=kill_switch_state,
        risk_budget_used=risk_budget_used,
    )


@router.post("/portfolio/rebalance", response_model=PortfolioRebalanceResponse)
async def portfolio_rebalance(payload: PortfolioRebalanceRequest) -> PortfolioRebalanceResponse:
    if not payload.signals:
        raise HTTPException(status_code=400, detail="signals cannot be empty")

    raw: List[Tuple[str, str, float, str]] = []
    for sig in payload.signals:
        side = 0.0
        if sig.action == "buy":
            side = 1.0
        elif sig.action == "sell":
            side = -1.0
        strength = side * abs(sig.score) * max(0.1, sig.confidence)
        bucket = _strategy_bucket(sig.track, float(sig.score), float(sig.confidence), sig.target)
        raw.append((sig.target.upper(), sig.track, strength, bucket))

    gross_strength = sum(abs(x[2]) for x in raw)
    if gross_strength <= 0:
        raise HTTPException(status_code=400, detail="no actionable signals")

    bucket_caps = _bucket_limits()
    target_positions: List[RebalancePosition] = []
    pre_bucket_exposure: Dict[str, float] = {}
    for target, track, strength, bucket in raw:
        weight = (strength / gross_strength) * payload.risk_budget
        pre_bucket_exposure[bucket] = pre_bucket_exposure.get(bucket, 0.0) + abs(weight)
        target_positions.append(RebalancePosition(target=target, track=track, weight=weight, style_bucket=bucket))

    bucket_violations: List[str] = []
    for bucket, exposure in pre_bucket_exposure.items():
        cap = payload.risk_budget * bucket_caps.get(bucket, 1.0)
        if cap > 0 and exposure > cap:
            scale = cap / exposure
            bucket_violations.append(f"bucket_exposure_exceeded:{bucket}")
            target_positions = [
                RebalancePosition(
                    target=p.target,
                    track=p.track,
                    weight=(p.weight * scale if p.style_bucket == bucket else p.weight),
                    sector=p.sector,
                    style_bucket=p.style_bucket,
                )
                for p in target_positions
            ]

    adjusted, violations, gross, turnover = _evaluate_risk(
        proposed=target_positions,
        current=payload.current_positions,
        realized_drawdown=0.0,
    )
    violations = bucket_violations + violations

    current_map = _position_map(payload.current_positions)
    new_map = _position_map(adjusted)
    universe = set(current_map.keys()) | set(new_map.keys())

    orders: List[Dict] = []
    decision_id = uuid.uuid4().hex
    for target, track in universe:
        delta = new_map.get((target, track), 0.0) - current_map.get((target, track), 0.0)
        if abs(delta) < 1e-6:
            continue
        side = "buy" if delta > 0 else "sell"
        px = repo.latest_price_snapshot(target)
        est_price = float(px["price"]) if px else None
        orders.append(
            {
                "target": target,
                "track": track,
                "side": side,
                "quantity": round(abs(delta) * payload.capital, 8),
                "est_price": est_price,
                "est_cost_bps": 8.0,
                "status": "simulated",
                "metadata": {"weight_delta": delta},
            }
        )

    repo.save_positions_snapshot(decision_id, [p.model_dump() for p in adjusted], reason="rebalance_v2")
    repo.save_orders_sim(decision_id, orders)
    for code in violations:
        repo.save_risk_event(
            decision_id=decision_id,
            severity="warning",
            code=code,
            message="rebalance risk violation",
            payload={"gross_exposure": gross, "turnover": turnover},
        )

    return PortfolioRebalanceResponse(
        decision_id=decision_id,
        target_positions=adjusted,
        expected_turnover=round(turnover, 6),
        orders=orders,
        risk_ok=len(violations) == 0,
        risk_violations=violations,
    )


@router.post("/execution/orders", response_model=SubmitExecutionOrdersResponse)
async def submit_execution_orders(payload: SubmitExecutionOrdersRequest) -> SubmitExecutionOrdersResponse:
    if not payload.orders:
        raise HTTPException(status_code=400, detail="orders cannot be empty")
    decision_id = uuid.uuid4().hex
    order_payloads = [
        {
            "target": o.target.upper(),
            "track": o.track,
            "side": o.side,
            "quantity": o.quantity,
            "est_price": o.est_price,
            "strategy_id": o.strategy_id,
            "metadata": o.metadata,
        }
        for o in payload.orders
    ]
    order_ids = repo.create_execution_orders(
        decision_id=decision_id,
        adapter=payload.adapter,
        venue=payload.venue,
        time_in_force=payload.time_in_force,
        max_slippage_bps=payload.max_slippage_bps,
        orders=order_payloads,
    )
    return SubmitExecutionOrdersResponse(
        decision_id=decision_id,
        adapter=payload.adapter,
        venue=payload.venue,
        accepted_orders=len(order_ids),
        order_ids=order_ids,
    )


@router.get("/execution/orders/{order_id}", response_model=ExecutionOrderStatusResponse)
async def get_execution_order(order_id: int) -> ExecutionOrderStatusResponse:
    row = repo.get_order_by_id(order_id)
    if not row:
        raise HTTPException(status_code=404, detail="order not found")
    return ExecutionOrderStatusResponse(
        order_id=int(row["id"]),
        decision_id=str(row["decision_id"]),
        target=str(row["target"]),
        side=str(row["side"]),
        quantity=float(row["quantity"]),
        status=str(row["status"]),
        track=str(row["track"]),
        venue=str(row.get("venue") or "coinbase"),
        adapter=str(row.get("adapter") or "paper"),
        created_at=row["created_at"],
        metadata=row.get("metadata") or {},
    )


@router.post("/execution/run", response_model=ExecuteOrdersResponse)
async def run_execution(payload: ExecuteOrdersRequest) -> ExecuteOrdersResponse:
    orders = repo.fetch_orders_for_decision(payload.decision_id, limit=payload.max_orders)
    if not orders:
        raise HTTPException(status_code=404, detail="no orders for decision")
    results = exec_engine.run(
        payload.adapter,
        orders,
        context={
            "time_in_force": payload.time_in_force,
            "max_slippage_bps": payload.max_slippage_bps,
            "venue": payload.venue,
        },
    )
    filled = 0
    rejected = 0
    merged = []
    for order, res in zip(orders, results):
        status = res["status"]
        if status == "filled":
            filled += 1
        elif status == "rejected":
            rejected += 1
        repo.update_order_execution(order["id"], status=status, metadata={"execution": res})
        merged.append({**order, "execution": res})

    return ExecuteOrdersResponse(
        decision_id=payload.decision_id,
        adapter=payload.adapter,
        total=len(orders),
        filled=filled,
        rejected=rejected,
        orders=merged,
    )


@router.post("/models/drift/evaluate", response_model=DriftEvaluateResponse)
async def evaluate_model_drift(payload: DriftEvaluateRequest) -> DriftEvaluateResponse:
    current_scores = repo.get_recent_prediction_scores(payload.track, payload.lookback_hours)
    reference_scores = repo.get_prediction_scores_window(
        payload.track,
        offset_hours=payload.lookback_hours,
        window_hours=payload.reference_hours,
    )
    slippage = repo.get_execution_slippage_samples(payload.track, payload.lookback_hours)

    psi_val = _psi(reference_scores, current_scores)
    ks_val = _ks_statistic(reference_scores, current_scores)
    slippage_abs_mean = float(np.mean(np.abs(slippage))) if slippage else 0.0
    drift = psi_val >= payload.psi_threshold or ks_val >= payload.ks_threshold
    action = "rollback_check" if drift else ("warn" if psi_val > payload.psi_threshold * 0.7 else "keep")
    reason = "distribution_shift_detected" if drift else "stable"

    return DriftEvaluateResponse(
        track=payload.track,
        drift_detected=drift,
        action=action,
        metrics={
            "psi": round(psi_val, 6),
            "ks": round(ks_val, 6),
            "execution_slippage_abs_mean": round(slippage_abs_mean, 8),
            "current_samples": float(len(current_scores)),
            "reference_samples": float(len(reference_scores)),
        },
        reason=reason,
        evaluated_at=datetime.utcnow(),
    )


@router.post("/models/gate/auto-evaluate", response_model=AutoGateEvaluateResponse)
async def auto_evaluate_model_gate(payload: AutoGateEvaluateRequest) -> AutoGateEvaluateResponse:
    default_name, default_version = _default_model_by_track(payload.track)
    model_name = payload.model_name or default_name
    model_version = payload.model_version or default_version
    passed, reason, summary, checked = _evaluate_gate(
        track=payload.track,
        min_ic=payload.min_ic,
        min_pnl_after_cost=payload.min_pnl_after_cost,
        max_drawdown=payload.max_drawdown,
        windows=payload.windows,
    )
    repo.promote_model(
        track=payload.track,
        model_name=model_name,
        model_version=model_version,
        passed=passed,
        metrics={**summary, "windows_checked": checked, "source": "auto_gate"},
        gate_reason=reason,
    )
    promoted = False
    if passed and payload.auto_promote:
        current = repo.get_active_model_state(payload.track)
        repo.upsert_active_model_state(
            track=payload.track,
            active_model_name=model_name,
            active_model_version=model_version,
            previous_model_name=current.get("active_model_name") if current else None,
            previous_model_version=current.get("active_model_version") if current else None,
            status="active",
            metadata={"source": "auto_gate", "summary": summary},
        )
        promoted = True
    return AutoGateEvaluateResponse(
        passed=passed,
        promoted=promoted,
        track=payload.track,
        model_name=model_name,
        model_version=model_version,
        reason=reason,
        windows_checked=checked,
        metrics_summary=summary,
    )


@router.get("/metrics/pnl-attribution", response_model=PnLAttributionResponse)
async def get_pnl_attribution(track: str = "liquid", lookback_hours: int = 24 * 7) -> PnLAttributionResponse:
    if track not in {"liquid", "vc"}:
        raise HTTPException(status_code=400, detail="track must be liquid or vc")
    attribution = repo.build_pnl_attribution(track=track, lookback_hours=lookback_hours)
    return PnLAttributionResponse(
        track=track,  # type: ignore[arg-type]
        lookback_hours=lookback_hours,
        totals=attribution["totals"],
        by_target=attribution["by_target"],
        generated_at=datetime.utcnow(),
    )


@router.post("/data-quality/sample")
async def sample_data_quality(payload: DataQualitySampleRequest):
    return {
        "items": repo.sample_data_quality(limit=payload.limit, min_quality_score=payload.min_quality_score),
        "limit": payload.limit,
    }


@router.get("/data-quality/stats", response_model=DataQualityStatsResponse)
async def data_quality_stats(lookback_days: int = 7) -> DataQualityStatsResponse:
    lookback_days = max(1, min(90, int(lookback_days)))
    stats = repo.get_data_quality_stats(lookback_days=lookback_days)
    return DataQualityStatsResponse(
        lookback_days=lookback_days,
        totals=stats.get("totals", {}),
        by_source=stats.get("by_source", []),
        generated_at=datetime.utcnow(),
    )


@router.post("/data-quality/audit")
async def update_data_quality_audit(payload: DataQualityAuditUpdate):
    repo.update_data_quality_audit(
        audit_id=payload.audit_id,
        reviewer=payload.reviewer,
        verdict=payload.verdict,
        note=payload.note,
    )
    return {"status": "ok", "audit_id": payload.audit_id}
